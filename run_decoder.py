from pathlib import Path
from datetime import datetime
from typing import List
from types import SimpleNamespace
import pickle
import numpy as np
import torch

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
chggen = CHGGen(lattice_scaler= lattice_scaler) # good
chggen.to(device = device)




# encode
mu, log_var, z = chggen.encode([x_, x_])

# chggen([x_, x_])

# generate stats
num_atoms, _, lengths, angles, composition_per_atom = chggen.decode_stats(z)
# obtain atom types.
composition_per_atom = F.softmax(composition_per_atom, dim=-1)
cur_atom_types = chggen.sample_composition(
                composition_per_atom, num_atoms)
            
# init coords.
cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=z.device)


# decode
decoder = NequipDecoder()

new_cart_coord_diff, new_atom_types = decoder(z, cur_frac_coords, cur_atom_types, num_atoms, lengths, angles)


print("Done")

# # # predict.
# # composition = mlp_composition(z).detach()          # 1D tensor
# # pred_lengths_and_angles = mlp_lattice(z).detach()              # 3D tensor
# # num_atoms = mlp_num_atoms(z).detach()       # 1D tensor with max_atom + 1


# # # predict.
# # num_atoms_prob = chggen.predict_num_atoms(z).detach() 
# # num_atoms = chggen.predict_num_atoms(z).argmax(dim=-1) # 1D tensor with max_atom + 1
# # composition_per_atom = chggen.predict_composition(z, num_atoms).detach()          # 2D tensor [atom_batch, max_Z]
# # lengths_and_angles, lengths, angles = chggen.predict_lattice(z, num_atoms)              # 3D tensor

# # # pred_composition = chggen.sample_composition(composition_prob= composition_prob_per_atom, num_atoms= num_atoms)