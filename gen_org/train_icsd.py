import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from chggen.common.data_utils import mkdir
from chggen.pl_data.dataset import CHGNetDataset
from chggen.pl_data.datamodule import CrystDataModule
from chggen.pl_modules.model import CHGGen



mkdir("./data/test_models/ICSD19_Zn/")

# Initialize dataset.
train_dataset = CHGNetDataset(
    path= './data/dataset/icsd/icsd_Zn.csv', # train.csv
    name = 'train_MP',
    prop_list = [],
)

val_dataset = CHGNetDataset(
    path= './data/dataset/mp_20/val.csv', # val.csv
    name = 'val_MP',
    prop_list = [],
)

datamodule = CrystDataModule(
    train_dataset= train_dataset,
    val_dataset= val_dataset,
    num_workers=20,
    batch_size= 1,
)

# Initialize model.
model_hparams ={'latent_dim': 64,           # Model dimension.
                'hidden_dim': 64, 
                'predict_property': False,  # Property guidance. 
                'property_dim': 1, 
                'fc_num_layers': 2, 
                'sigma_begin': 10.0,        # Noise level.
                'sigma_end': 0.001, 
                'num_noise_level': 200,     
                'cost_coord': 1.0,         # Loss weight.
                'cost_property': 0,
                'teacher_forcing_lattice': True,
                'teacher_forcing_max_epoch': 1000,
                'decoder': 'nequip_v2',
                'lr': 1e-2,
                }

chggen = CHGGen(hparams_dict= model_hparams)

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath= './data/test_models/ICSD19_Zn/',
    filename='{epoch}',         # Save the checkpoint after every epoch
    save_top_k=-1,              # Set to -1 to save all checkpoints
    save_last=True,             # Save the last model too, useful for resuming
    every_n_train_steps = 5000, # Save every epoch (assuming you're validating every epoch)
    verbose=False               # Print save messages for debugging
)

trainer = pl.Trainer(
    accelerator = "gpu", 
    devices = [1],
    max_epochs= 10,
    callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate = 1)],
    log_every_n_steps=100,
    gradient_clip_val=1.0,
    default_root_dir='./data/test_models/ICSD19_Zn/',
    # strategy = 'ddp_find_unused_parameters_true',  # multi-GPU training
)

trainer.fit(model= chggen, datamodule= datamodule)

trainer.save_checkpoint("./data/test_models/ICSD19_Zn/trainer_ICSD19_Zn.ckpt")
print("Done")