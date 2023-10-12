"""Main file for training DiffCSP."""
import pickle
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from chggen.common.data_utils import get_scaler
from chggen.pl_data.dataset import CHGNetDataset
from chggen.pl_data.datamodule import CrystDataModule
from chggen.pl_modules.model_egnn import CHGGen



# import warnings
# warnings.simplefilter(action='ignore', category=Warning)
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Functions.
def mkdir(path: str) -> None:
    """Make directory if folder not exist.

    Args:
        path (str): directory name
    """
    if os.path.exists(path):
        print("Folder exists")
    else:
        os.makedirs(path)
    return None

# Dataset.
train_dataset = CHGNetDataset(
    path= '/home/xzdai/ceder_group/material_dircovery/chggen_old/data/perov_5/train.csv', # train.csv
    name = 'train_perov',
    prop_list = ['heat_all'],
)

val_dataset = CHGNetDataset(
    path= '/home/xzdai/ceder_group/material_dircovery/chggen_old/data/perov_5/val.csv', # val.csv
    name = 'val_perov',
    prop_list = ['heat_all'],
)

# Compute lattice scaler and save.
lattice_scaler = get_scaler(dataset= train_dataset)
with open('./test_models/lattice_scaler_perov', 'wb') as fp:
    pickle.dump(lattice_scaler, fp)

datamodule = CrystDataModule(
    train_dataset = train_dataset,
    val_dataset = val_dataset,
    num_workers = 8,
    batch_size = 16,
)

model_hparams = {'latent_dim': 64, 'hidden_dim': 128, 
                'predict_property': True, 'property_dim': 1, # predict the multiple property 
                'load_pretrain': True, 'fc_num_layers': 1, 
                'sigma_F_begin': 0.5, 'sigma_F_end': 0.005, 
                'sigma_L_begin': 1.5, 'sigma_L_end': 0.015, 
                'type_sigma_begin': 5.0, 'type_sigma_end': 0.01,
                'max_atoms': 10, # should be larger than the training set.
                'num_noise_level': 50, 
                'lattice_scale_method': 'scale_length', 
                'cost_natom': 1.0, 'cost_latt': 10.0, 'cost_coord': 10.0, 'cost_type': 1.0, 'cost_lattice': 10.0, 'cost_composition': 1.0, 'cost_edge': 10.0, 'cost_property': 1.0,
                'beta': 0.01,
                'teacher_forcing_lattice': True,
                'teacher_forcing_max_epoch': 1000,
                'decoder': 'egnn'}

chggen = CHGGen(
    lattice_scaler = lattice_scaler, hparams_dict = model_hparams
)

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath= './test_models/perov/',
    filename='{epoch}',       # Save the checkpoint after every epoch
    save_top_k=-1,            # Set to -1 to save all checkpoints
    save_last=True,           # Save the last model too, useful for resuming
    every_n_train_steps=100,  # Save every epoch (assuming you're validating every epoch)
    verbose=True              # Print save messages for debugging
)

trainer = pl.Trainer(
    accelerator = "gpu", 
    devices = [0],
    max_epochs = 30,
    callbacks = [checkpoint_callback, TQDMProgressBar(refresh_rate = -1)],
    #  strategy = 'ddp_find_unused_parameters_true',  # multi-GPU training                    
)

trainer.fit(model = chggen, datamodule = datamodule)

mkdir("./test_models/perov/")
trainer.save_checkpoint("./test_models/perov/trainer_perov.ckpt")

with open('./test_models/chggen_perov.pk', 'wb') as fp:
    pickle.dump(chggen, fp)

print("Done")