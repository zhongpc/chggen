from pathlib import Path
from datetime import datetime
from typing import List
from types import SimpleNamespace
import pickle
import numpy as np
import torch

from chggen.pl_data.dataset import CHGNetDataset
# process_csv
# from chggen.pl_modules.encoder import CHGNet_encoder
# from chggen.pl_modules.decoder import GemNetTDecoder
from chggen.pl_modules.model import CHGGen
from chggen.common.data_utils import get_scaler_from_data_list

from nequip.radial_embedding import InitialEmbedding
from nequip.e3nn_nequip import NequIP

from torch_geometric.data import Data
from nequip.mlp import MLP_c, MLP_l, MLP_n

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
device = torch.device("cuda")

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

from chggen.pl_modules.model import build_mlp

mlp_composition, mlp_lattice, mlp_num_atoms = MLP_c(), MLP_l(), MLP_n()

NUM_SPECIES = 2
CUTOFF = 5.0

nequip = NequIP(
    init_embed     = InitialEmbedding(num_species=NUM_SPECIES, cutoff=CUTOFF),
    irreps_node_x  = '8x0e',
    irreps_node_z  = '8x0e',
    irreps_hidden  = '8x0e + 8x1e + 4x2e',
    irreps_edge    = '1x0e + 1x1e + 1x2e',
    irreps_out     = '1x1e',
    num_convs      = 3,
    radial_neurons = [16, 64],
    num_neighbors  = 12,
)

# encode
mu, log_var, z = chggen.encode([x_, x_])

# predict.
composition = mlp1(z).detach()          # 1D tensor
lattice = mlp2(z).detach()              # 3D tensor
num_atoms = mlp3(z).detach().item()     # scaler

# random initialize.
nums_atoms =  torch.round(composition*num_atoms)
atom_type = 0
x = []
for num in nums_atoms:
    x_temp = [atom_type] * num 
x += x_temp                             # node feature, atomic type here
pos = torch.rand(size=(round(num_atoms), 3))   # atoms position
cell = lattice                          # cell size 
pbc = True                              # periodic boundary condition

data = Data(
    x = torch.tensor(x).long(),
    pos = torch.tensor(pos).float(),
    cell = torch.tensor(cell).float(),
    pbc = torch.tensor(pbc).bool(),
)

# denoise
nequip(data)

ld_kwargs = SimpleNamespace(n_step_each = 100,
                                step_lr = 1e-4,
                                min_sigma = 0,
                                save_traj = False,
                                disable_bar = False)

output = chggen.langevin_dynamics(z= z,
                                  ld_kwargs= ld_kwargs)

with open('./output_from_ld', 'wb') as fp:
    pickle.dump(output, fp)

print("Done")



## 
# data_params = {
#     "batch_size": 16,
#     "pin_memory": True,
#     "shuffle": True,
#     "collate_fn": collate_batch_v1,
# }

# data_loader = DataLoader(dataset, **data_params