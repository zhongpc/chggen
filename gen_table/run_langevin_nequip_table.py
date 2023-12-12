"""Main file for running annealed Langevin dynamics for new material sampling."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from chggen.pl_data.dataset import CHGNetDataset
from chggen.pl_modules.model_table import CHGGen
from chggen.common.data_utils import get_scaler, mkdir



mkdir('./test_models/structures/')

# Load dataset for obtaining lattice_scaler for scaling in Niggli reduction.
dataset = CHGNetDataset(
    path='/home/xzdai/ceder_group/material_dircovery/chggen_table/data/perov_5/test_zpc.csv',
    name = 'A_good_name',
    prop_list = ['heat_all'],
)
lattice_scaler = get_scaler(dataset= dataset)

# Load model.
device = torch.device('cuda')
model_hparams = {'latent_dim': 64, 'hidden_dim': 128, 
                'predict_property': True, 'property_dim': 1, # predict the multiple property 
                'load_pretrain': True, 'fc_num_layers': 1, 
                'sigma_begin': 0.5, 'sigma_end': 0.005,
                'type_sigma_begin': 5.0, 'type_sigma_end': 0.01,
                'max_atoms': 10,        # should be larger than the training set.
                'num_noise_level': 50, 
                'lattice_scale_method': 'scale_length', 
                'cost_natom': 1.0, 'cost_latt': 10.0, 'cost_coord': 10.0, 'cost_type': 1.0, 'cost_lattice': 10.0, 'cost_composition': 1.0, 'cost_edge': 10.0, 'cost_property': 1.0,
                'beta': 0.01,
                'teacher_forcing_lattice': True,
                'teacher_forcing_max_epoch': 1000,
                'decoder': 'nequip_table'}
chggen = CHGGen(
    lattice_scaler = lattice_scaler, hparams_dict = model_hparams
)

checkpoint_path = "./test_models/perov_table/trainer_perov.ckpt"
chggen = chggen.load_from_checkpoint(checkpoint_path = checkpoint_path)
chggen.lattice_scaler = lattice_scaler
chggen.to(device = device)
print('Model loaded')

# Sampling.
ld_kwargs = SimpleNamespace(
    n_step_each = 50,
    step_lr = 1e-4,
    min_sigma = 0,
    save_traj = False,
    disable_bar = False,
    compute_force = True,
    beta_c = 0,         # property update rate
    beta_f = 0,         # atomic force update rate                          
)


cur_frac_coords = torch.rand(50, 3, requires_grad = False, device = device)
cur_atom_types = torch.tensor([7, 7, 8, 22, 20, 7, 8, 8, 22, 20,7, 7, 8, 40, 20,7, 7, 8, 72, 20,7, 7, 8, 22, 20
                               ,7, 7, 8, 22, 20,7, 7, 8, 28, 30,7, 7, 8, 40, 56,7, 7, 8, 22, 20,8, 8, 8, 40, 20], device = device)
num_atoms = torch.tensor([5,5,5,5,5,5,5,5,5,5], device = device)
lengths = 3*torch.tensor([[3.905, 3.905, 3.905],
                        [3.905, 3.905, 3.905],
                        [3.905, 3.905, 3.905],
                        [3.905, 3.905, 3.905],
                        [3.905, 3.905, 3.905],
                        [3.905, 3.905, 3.905],
                        [3.905, 3.905, 3.905],
                        [3.905, 3.905, 3.905],
                        [3.905, 3.905, 3.905],
                        [3.905, 3.905, 3.905],], device = device)
angles = torch.tensor([[90, 90, 90],
                       [90, 90, 90],
                       [90, 90, 90],
                       [90, 90, 90],
                       [90, 90, 90],
                       [90, 90, 90],
                       [90, 90, 90],
                       [90, 90, 90],
                       [90, 90, 90],
                       [90, 90, 90],], device = device)

from chggen.common.data_utils import (
    EPSILON, cart_to_frac_coords, mard, lengths_angles_to_volume,
    frac_to_cart_coords, min_distance_sqr_pbc)
from tqdm import tqdm

# annealed langevin dynamics.
for sigma in tqdm(chggen.sigmas, total=chggen.sigmas.size(0), disable=ld_kwargs.disable_bar):
    if sigma < ld_kwargs.min_sigma:
        break
    step_size = ld_kwargs.step_lr * (sigma / chggen.sigmas[-1]) ** 2

    for step in range(ld_kwargs.n_step_each):
        noise_cart = torch.randn_like(
            cur_frac_coords) * torch.sqrt(step_size * 2)
        with torch.no_grad():
            pred_cart_coord_diff = chggen.decoder(
                cur_frac_coords, cur_atom_types, num_atoms, lengths, angles)

        cur_cart_coords = frac_to_cart_coords(
            cur_frac_coords, lengths, angles, num_atoms)
        
        pred_cart_coord_diff = pred_cart_coord_diff / sigma
        cur_cart_coords = cur_cart_coords + step_size * pred_cart_coord_diff + noise_cart

        
        cur_frac_coords = cart_to_frac_coords(
            cur_cart_coords, lengths, angles, num_atoms)

results = {'num_atoms': num_atoms, 'lengths': lengths, 'angles': angles,
                'frac_coords': cur_frac_coords, 'atom_types': cur_atom_types,
                'is_traj': False}

# # training data.
# z = chggen.encoder([dataset[0].crys_graph.to(device = device)])
# z = z['crystal_fea']

# # Initialize letent variable z.
# z = torch.rand(1, 64, requires_grad = False, device = device)

# results = chggen.langevin_dynamics_guidance(
#     z = z, 
#     prop_guidance = torch.tensor(-0.05, device = device), 
#     ld_kwargs = ld_kwargs
# )

lengths = results['lengths']
angles = results['angles']
num_atoms = results['num_atoms']
frac_coords = results['frac_coords']
atom_types = results['atom_types']

print(lengths)
print(angles)
print(num_atoms)
print(frac_coords)
print(atom_types)

# # Save final structures.
# for structure_id in range(len(num_atoms)):
#     s = chggen.get_pymatgen_structure(
#         lengths = lengths[structure_id].reshape(1,3),         
#         angles = angles[structure_id].reshape(1,3),
#         num_atoms = num_atoms[structure_id].reshape(1),
#         frac_coords = frac_coords[structure_id:structure_id+5],
#         atom_types = atom_types[structure_id:structure_id+5],
#     )
#     s.to(filename='./test_models/table_structures/structure_' + str(structure_id) + '.cif')

s_list = chggen.get_pymatgen_structure(
    lengths = lengths,         
    angles = angles,
    num_atoms = num_atoms,
    frac_coords = frac_coords,
    atom_types = atom_types,
)

for structure_id in range(len(num_atoms)):
    s_list[structure_id].to(filename='./test_models/structures/structure_' + str(structure_id) + '.cif')

print("Done")