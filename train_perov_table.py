from pathlib import Path
import pickle
import torch
import os
from chggen.common.data_utils import get_scaler_from_data_list, get_scaler
from chggen.pl_data.dataset import CHGNetDataset
from chggen.pl_data.datamodule import CrystDataModule
from chggen.pl_modules.model_table import CHGGen
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar



import warnings
warnings.simplefilter(action='ignore', category=Warning)
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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



mkdir("./test_models/perov_table/")


train_dataset = CHGNetDataset(path= './data/perov_5/test_zpc.csv', # train.csv
name = 'train_perov',
prop_list = ['heat_all'],
)

val_dataset = CHGNetDataset(path= './data/perov_5/test_zpc.csv', # val.csv
name = 'val_perov',
prop_list = ['heat_all'],
)

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

lattice_scaler = get_scaler(dataset= train_dataset)

with open('./test_models/lattice_scaler_perov', 'wb') as fp:
    pickle.dump(lattice_scaler, fp)

datamodule = CrystDataModule(train_dataset= train_dataset,
                             val_dataset= val_dataset,
                             num_workers=8,
                             batch_size= 16,
                                )

model_hparams ={'latent_dim': 64, 'hidden_dim': 64, 
                'predict_property': True, 'property_dim': 1, # predict the multiple property 
                'load_pretrain': True, 'fc_num_layers': 2, 
                'sigma_begin': 10.0, 'sigma_end': 0.01, 'type_sigma_begin': 5.0, 'type_sigma_end': 0.01,
                'max_atoms': 20, # should be larger than the training set.
                'num_noise_level': 50, 
                'lattice_scale_method': 'scale_length', 
                'cost_natom': 1.0, 'cost_coord': 10.0, 'cost_type': 1.0, 'cost_lattice': 10.0, 'cost_composition': 1.0, 'cost_edge': 10.0, 'cost_property': 0,
                'beta': 0.01, # cost ratio of the KLD for VAE
                'teacher_forcing_lattice': True,
                'teacher_forcing_max_epoch': 1000,
                'decoder': 'nequip_table'}

chggen = CHGGen(lattice_scaler= lattice_scaler, 
                hparams_dict= model_hparams)


# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath= './test_models/perov_table/',
    filename='{epoch}',        # Save the checkpoint after every epoch
    save_top_k=-1,            # Set to -1 to save all checkpoints
    save_last=True,           # Save the last model too, useful for resuming
    every_n_train_steps=100,     # Save every epoch (assuming you're validating every epoch)
    verbose=True              # Print save messages for debugging
)


trainer = pl.Trainer(accelerator = "cpu", 
                    #  devices = [0],
                     max_epochs= 10,
                     callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate = 1)],
                    #  strategy = 'ddp_find_unused_parameters_true',  # multi-GPU training
                     )

trainer.fit(model= chggen, datamodule= datamodule)

trainer.save_checkpoint("./test_models/perov_table/trainer_perov.ckpt")
print("Done")