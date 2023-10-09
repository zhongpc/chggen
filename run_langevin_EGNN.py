"""Main file for running annealed Langevin dynamics for new material sampling."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from pymatgen.core import Structure, Lattice, Element

from chggen.pl_data.dataset import CHGNetDataset
from chggen.pl_modules.model_egnn import CHGGen
from chggen.common.data_utils import get_scaler


# Load dataset for obtaining lattice_scaler for scaling in Niggli reduction.
dataset = CHGNetDataset(
    path='/home/xzdai/ceder_group/material_dircovery/chggen_old/data/perov_5/test_zpc.csv',
    name = 'A_good_name',
    prop_list = ['heat_all'],
)

lattice_scaler = get_scaler(dataset= dataset)

model_hparams ={'latent_dim': 64, 'hidden_dim': 128, 
                'predict_property': True, 'property_dim': 1,  
                'load_pretrain': True, 
                'fc_num_layers': 1, 
                'sigma_F_begin': 10.0, 'sigma_F_end': 0.01, 
                'sigma_L_begin': 1.0, 'sigma_L_end': 0.01, 
                'type_sigma_begin': 5.0, 'type_sigma_end': 0.01,
                'max_atoms': 20, 
                'num_noise_level': 1, 
                'lattice_scale_method': 'scale_length', 
                'cost_natom': 1.0, 'cost_coord': 10.0, 'cost_type': 1.0, 'cost_lattice': 10.0, 'cost_composition': 1.0, 'cost_edge': 10.0, 'cost_property': 1.0,
                'beta': 0.01,
                'teacher_forcing_lattice': True,
                'teacher_forcing_max_epoch': 1000,
                'decoder': 'egnn'}
device = torch.device('cpu')

chggen = CHGGen(
    hparams_dict = model_hparams, lattice_scaler = lattice_scaler, 
)
chggen.to(device = device)

# langevin dynamics
ld_kwargs = SimpleNamespace(
    n_step_each = 10,
    step_lr = 1e-4,
    min_sigma = 0,
    save_traj = False,
    disable_bar = False,
    compute_force = True,
    beta_c = 0,         # property update rate
    beta_f = 0,         # atomic force update rate                          
)

z = torch.rand(5, 64, requires_grad = True, device = device)
results = chggen.langevin_dynamics_guidance(
    z = z, 
    prop_guidance = torch.tensor(-0.05, device = device), 
    ld_kwargs = ld_kwargs
)

lattices = results['lattices']
num_atoms = results['num_atoms']        
num_atoms = num_atoms[num_atoms.nonzero().squeeze()]        # Consider num_atom is zero
frac_coords = results['frac_coords']
atom_types = results['atom_types']

# Save final structures.
s_list = chggen.get_pymatgen_structure_from_lattice(
    lattices = lattices[num_atoms.cumsum(dim=0)-1],         # Convert lattices shape to (num of structures, 3, 3)
    num_atoms = num_atoms,
    frac_coords = frac_coords,
    atom_types = atom_types,
)

for structure_id in range(len(num_atoms)):
    s_list[structure_id].to(filename='./test_models/EGNN_structures/prop_guidance_' + str(structure_id) + '.cif')

print("Done")