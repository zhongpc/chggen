from pathlib import Path
import pickle
import torch
from chggen.pl_data.dataset import CHGNetDataset
from chggen.pl_modules.model import CHGGen
from chggen.common.data_utils import get_scaler_from_data_list

device = torch.device("cuda:7")

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
    else:
        lattice_scaler = torch.load(
            Path(scaler_path) / 'lattice_scaler.pt')
    return lattice_scaler


lattice_scaler = get_scaler(dataset= dataset)
item = dataset[0]

x_ = item['crys_graph']
y_ = item['y']

x_ = x_.to(device = device)
y_ = y_.to(device = device)

chggen = CHGGen(lattice_scaler= lattice_scaler) # good
chggen.to(device = device)