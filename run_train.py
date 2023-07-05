from pathlib import Path
import pickle
import torch
from chggen.pl_data.dataset import CHGNetDataset
from chggen.pl_data.datamodule import CrystDataModule
from chggen.pl_modules.model import CHGGen
from chggen.common.data_utils import get_scaler_from_data_list
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
# from chggen.pl_modules.trainer import Trainer
from torch_geometric.data import Batch

import pytorch_lightning as pl


# device = torch.device("cuda:7")

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


def collate_graphs(batch_data: list):
    """Collate of list of (graph, targets) into batch data.
    """
    all_data = []
    all_graphs = []
    all_targets = []
    for item in batch_data:
        graph = item[0]
        data =  item[1]
        targets = data['properties']
        all_graphs.append(graph)
        all_data.append(data)
        all_targets.append(targets)
    return all_graphs, all_data, all_targets


def get_loader(dataset, batch_size=64, num_workers=0, pin_memory=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_graphs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

def get_loader_batch(dataset, batch_size=64, num_workers=0, pin_memory=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


lattice_scaler = get_scaler(dataset= dataset)



data_list = [dataset[i] for i in range(len(dataset))]


# data_batch = Batch.from_data_list(data_list= data_list)

# dataloader = get_loader_batch(dataset=data_batch, batch_size=7)

datamodule = CrystDataModule(train_dataset= dataset,
                             val_dataset= dataset,
                             num_workers=8,
                             batch_size= 7,
                                )

chggen = CHGGen(lattice_scaler= lattice_scaler)

# chggen.to(device=device)
# trainer = pl.Trainer(devices=2, accelerator="gpu")

for name, param in chggen.named_parameters():
    print(name)
    if "encoder" in name:
        param.requires_grad = False

trainer = pl.Trainer(accelerator = "gpu", 
                     devices = [6, 7],
                     max_epochs= 5,
                     strategy = 'ddp_find_unused_parameters_true')

trainer.fit(model= chggen, datamodule= datamodule)


print("Done")