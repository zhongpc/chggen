from pathlib import Path
from datetime import datetime
from typing import List
from types import SimpleNamespace
import pickle
import numpy as np
import torch
from pymatgen.core import Structure, Lattice, Species, Element


from chggen.pl_data.dataset import CHGNetDataset
from chggen.pl_modules.model import CHGGen
from chggen.common.data_utils import get_scaler_from_data_list
from chggen.pl_modules.decoder import NequipDecoder
# from nequip.radial_embedding import InitialEmbedding
# from nequip.e3nn_nequip import NequIP

from torch_geometric.data import Data
from nequip.mlp import MLP_c, MLP_l, MLP_n
from torch import nn
import torch.nn.functional as F



def get_scaler(dataset, use_prop_scaler = False, 
               scaler_path = None):
    # Load once to compute property scaler
    if scaler_path is None:
        lattice_scaler = get_scaler_from_data_list(
            dataset.cached_data,
            key='scaled_lattice')
        if use_prop_scaler:
            NotImplementedError("Not implemented the multi prop scaler yet.")
    else:
        lattice_scaler = torch.load(
            Path(scaler_path) / 'lattice_scaler.pt')
    return lattice_scaler



# setting
device = torch.device("cpu")

# dataset
dataset = CHGNetDataset(path= '/home/zhongpc/chggen/data/perov_5/test_zpc.csv',
                        name = 'zpc_test',
                        prop_list = ['heat_ref', 'heat_all'],
                        )
lattice_scaler = get_scaler(dataset= dataset)
item = dataset[0]
x_ = item['crys_graph']
y_ = item['properties']
x_ = x_.to(device = device)
y_ = y_.to(device = device)

# model
model_hparams ={'latent_dim': 64, 'hidden_dim': 128, 
                'predict_property': True, 'property_dim': 1, # predict the multiple property 
                'load_pretrain': True, 'fc_num_layers': 1, 
                'sigma_begin': 10.0, 'sigma_end': 0.01, 'type_sigma_begin': 5.0, 'type_sigma_end': 0.01,
                'max_atoms': 20, # should be larger than the training set.
                'num_noise_level': 50, 
                'lattice_scale_method': 'scale_length', 
                'cost_natom': 1.0, 'cost_coord': 10.0, 'cost_type': 1.0, 'cost_lattice': 10.0, 'cost_composition': 1.0, 'cost_edge': 10.0, 'cost_property': 1.0,
                'beta': 0.01,
                'teacher_forcing_lattice': True,
                'teacher_forcing_max_epoch': 1000,
                'decoder': 'nequip'}

chggen = CHGGen(lattice_scaler= lattice_scaler, 
                hparams_dict= model_hparams)

chggen.to(device = device)

# langevin dynamics
ld_kwargs = SimpleNamespace(n_step_each = 10,
                            step_lr = 1e-4,
                            min_sigma = 0,
                            save_traj = False,
                            disable_bar = False)

z = torch.rand(3, 64, requires_grad=True)
# results = chggen.langevin_dynamics(z = z, ld_kwargs= ld_kwargs)
results = chggen.langevin_dynamics_guidance(z = z, ld_kwargs= ld_kwargs)

# save the results from langevin dynamics
lengths = results['lengths']
angles= results['angles']
num_atoms = results['num_atoms']
frac_coords = results['frac_coords']
atom_types = results['atom_types']

batch = torch.arange(len(num_atoms))
batch = batch.repeat_interleave(num_atoms)

for ii in range(len(num_atoms)):
    indices = torch.where(batch == ii)[0]
    print(ii, indices)
    if len(indices) == 0:
        continue
    
    
    Latt = Lattice.from_parameters(a = lengths[ii,0], b = lengths[ii,1], c = lengths[ii,2],
                                   alpha= angles[ii, 0], beta= angles[ii,1], gamma=angles[ii, 2])
                                   
    frac_ = frac_coords[indices]
    type_ = atom_types[indices]
    species_ = [Element.from_Z(ele_Z) for ele_Z in type_]
    
    s_gen = Structure(lattice= Latt , species= species_, coords= frac_,
                      to_unit_cell=False,coords_are_cartesian=False);
    print(s_gen.composition)
    s_gen.to(filename= './test_models/structures/prop_guidance_' + str(ii) + '.cif')
    


print("Done")
