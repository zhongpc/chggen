from pathlib import Path
from datetime import datetime
from typing import List
from types import SimpleNamespace
import pickle
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from chggen.pl_data.dataset import CHGNetDataset, process_csv


from chggen.pl_modules.encoder import CHGNet_encoder
from chggen.pl_modules.decoder import GemNetTDecoder
from chggen.pl_modules.model import CHGGen

from chggen.common.data_utils import get_scaler_from_data_list

device = torch.device("cuda:7")

# process_csv(input_file='/home/zhongpc/cdvae/data/perov_5/test_zpc.csv',
#             num_workers = 8, 
#             niggli  = True, 
#             primitive = False, 
#             prop_list = ['heat_ref'])


dataset = CHGNetDataset(path= '/home/zhongpc/chggen/data/perov_5/test_zpc.csv',
                        name = 'zpc_test',
                        prop_list = ['heat_ref', 'heat_all'],
                        )


### test the lattice scaler ##

def get_scaler(dataset, use_prop_scaler = False, 
               scaler_path = None):
    # Load once to compute property scaler
    if scaler_path is None:
        lattice_scaler = get_scaler_from_data_list(
            dataset.cached_data,
            key='scaled_lattice')
        if use_prop_scaler:
            NotImplementedError("Not implemented the multi prop scaler yet.")
            # prop_scaler = get_scaler_from_data_list(
            #     dataset.cached_data,
            # key= dataset.prop_list)
    else:
        lattice_scaler = torch.load(
            Path(scaler_path) / 'lattice_scaler.pt')
        # prop_scaler = torch.load(Path(scaler_path) / 'prop_scaler.pt')
    return lattice_scaler


lattice_scaler = get_scaler(dataset= dataset)
item = dataset[0]

x_ = item['crys_graph']
y_ = item['y']

x_ = x_.to(device = device)
y_ = y_.to(device = device)

# encoder = CHGNet_encoder().load()

# decoder = GemNetTDecoder

chggen = CHGGen(lattice_scaler= lattice_scaler) # good
chggen.to(device = device)


mu, log_var, z = chggen.encode([x_, x_])

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