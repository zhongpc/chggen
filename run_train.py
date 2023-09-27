from pathlib import Path
import pickle
import torch
import os
from chggen.common.data_utils import get_scaler_from_data_list
from chggen.pl_data.dataset import CHGNetDataset
from chggen.pl_data.datamodule import CrystDataModule
from chggen.pl_modules.model import CHGGen
# from chggen.pl_modules.model import CHGGen_gemnet as CHGGen

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.data import Batch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


import warnings
warnings.simplefilter(action='ignore', category=Warning)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def mkdir(path: str):
    """Make directory.

    Args:
        path (str): directory name

    Returns:
        path
    """
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("Folder exists")
    return path

# dataset = CHGNetDataset(path= '/home/zhongpc/chggen/data/mptrj/MPtrj_debug.csv',
#                         name = 'mptrj_debug',
#                         prop_list = ['e_hull'],
#                         )

# dataset = CHGNetDataset(path= '/home/zhongpc/chggen/data/perov_5/train.csv',
# name = 'train_perov',
# prop_list = ['heat_all'],
# )

dataset = CHGNetDataset(path= '/home/zhongpc/chggen/data/perov_5/test_zpc.csv',
name = 'zpc_debug',
prop_list = ['heat_all'],
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

with open('./test_models/lattice_scaler', 'wb') as fp:
    pickle.dump(lattice_scaler, fp)



data_list = [dataset[i] for i in range(len(dataset))]


# data_batch = Batch.from_data_list(data_list= data_list)

# dataloader = get_loader_batch(dataset=data_batch, batch_size=7)

datamodule = CrystDataModule(train_dataset= dataset,
                             val_dataset= dataset,
                             num_workers=8,
                             batch_size= 32,
                                )


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

# chggen.to(device=device)
# trainer = pl.Trainer(devices=2, accelerator="gpu")

for name, param in chggen.named_parameters():
    print(name)
    if "encoder" in name:
        param.requires_grad = False



# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filename='{epoch}',        # Save the checkpoint after every epoch
    save_top_k=-1,            # Set to -1 to save all checkpoints
    save_last=True,           # Save the last model too, useful for resuming
    every_n_train_steps=1,     # Save every epoch (assuming you're validating every epoch)
    verbose=True              # Print save messages for debugging
)

trainer = pl.Trainer(accelerator = "gpu", 
                     devices = [7],
                     max_epochs= 10,
                     callbacks=[checkpoint_callback],
                    #  strategy = 'ddp_find_unused_parameters_true',
                     )

trainer.fit(model= chggen, datamodule= datamodule)

mkdir("./test_models/perov/")
trainer.save_checkpoint("./test_models/perov/trainer_perov.ckpt")

print("Done")