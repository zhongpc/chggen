import pickle

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from chggen.common.data_utils import get_scaler, mkdir
from chggen.pl_data.dataset import CHGNetDataset
from chggen.pl_data.datamodule import CrystDataModule
from chggen.pl_modules.model import CHGGen



mkdir("./test_models/ICSD19_Na/")

# Initialize dataset.
train_dataset = CHGNetDataset(
    path= '/home/xzdai/ceder_group/material_dircovery/01_data_preparation/ICSD19/data/ICSD19_Na.csv', # train.csv
    name = 'train_MP',
    prop_list = [],
)

val_dataset = CHGNetDataset(
    path= '/home/xzdai/ceder_group/material_dircovery/01_data_preparation/ICSD19/data/ICSD19_Na.csv', # val.csv
    name = 'val_MP',
    prop_list = [],
)

datamodule = CrystDataModule(train_dataset= train_dataset,
                             val_dataset= val_dataset,
                             num_workers=20,
                             batch_size= 4,
                            )

# Initialize model.
model_hparams ={'latent_dim': 64, 'hidden_dim': 64, 
                'predict_property': True, 'property_dim': 1, # predict the multiple property 
                'load_pretrain': True, 'fc_num_layers': 2, 
                'sigma_begin': 10.0, 'sigma_end': 0.001, 
                'num_noise_level': 200, 
                'lattice_scale_method': 'scale_length', 
                'cost_coord': 10.0, 'cost_property': 0,
                'teacher_forcing_lattice': True,
                'teacher_forcing_max_epoch': 1000,
                'decoder': 'nequip_v2',
                'lr': 1e-3,
                }

chggen = CHGGen(hparams_dict= model_hparams)


# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath= './test_models/ICSD19_Na/',
    filename='{epoch}',         # Save the checkpoint after every epoch
    save_top_k=-1,              # Set to -1 to save all checkpoints
    save_last=True,             # Save the last model too, useful for resuming
    every_n_train_steps = 5000, # Save every epoch (assuming you're validating every epoch)
    verbose=False               # Print save messages for debugging
)

trainer = pl.Trainer(accelerator = "gpu", 
                     devices = [1],
                     max_epochs= 10,
                     callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate = 1)],
                     log_every_n_steps=100,
                     gradient_clip_val=1.0,
                    #  strategy = 'ddp_find_unused_parameters_true',  # multi-GPU training
                     )

trainer.fit(model= chggen, datamodule= datamodule)

trainer.save_checkpoint("./test_models/ICSD19_Na/trainer_ICSD19_Na.ckpt")
print("Done")