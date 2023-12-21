import pickle

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from chggen.common.data_utils import get_scaler, mkdir
from chggen.pl_data.dataset import CHGNetDataset
from chggen.pl_data.datamodule import CrystDataModule
from chggen.pl_modules.model import CHGGen

import warnings

# Suppress only UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="^Issues encountered while parsing CIF")



# Initialize dataset.
train_dataset = CHGNetDataset(path= '/home/zhongpc/chggen/data/mp_20/SC-15_train.csv', # train.csv
name = 'train_MP',
prop_list = ['band_gap'],
)
val_dataset = CHGNetDataset(path= '/home/zhongpc/chggen/data/mp_20/val.csv', # val.csv
name = 'val_MP',
prop_list = ['band_gap'],
)


mkdir("./test_models/MP_20/")

### need to remove in the future ###
lattice_scaler = get_scaler(dataset= val_dataset)
with open('./test_models/lattice_scaler_MP20', 'wb') as fp:
    pickle.dump(lattice_scaler, fp)

datamodule = CrystDataModule(train_dataset= train_dataset,
                             val_dataset= val_dataset,
                             num_workers=0,
                             batch_size= 4,
                            )

# Initialize model.
model_hparams ={'latent_dim': 64, 'hidden_dim': 64, 
                'predict_property': True, 'property_dim': 1, # predict the multiple property 
                'load_pretrain': True, 'fc_num_layers': 2, 
                'sigma_begin': 10.0, 'sigma_end': 0.001, 'type_sigma_begin': 5.0, 'type_sigma_end': 0.01,
                'max_atoms': 10001, # should be larger than the training set.
                'num_noise_level': 200, 
                'lattice_scale_method': 'scale_length', 
                'cost_natom': 1.0, 'cost_coord': 10.0, 'cost_type': 1.0, 'cost_lattice': 10.0, 'cost_composition': 1.0, 'cost_edge': 10.0, 'cost_property': 0,
                'beta': 0.01, # cost ratio of the KLD for VAE
                'teacher_forcing_lattice': True,
                'teacher_forcing_max_epoch': 1000,
                'decoder': 'nequip_v2'}

chggen = CHGGen(lattice_scaler= lattice_scaler, 
                hparams_dict= model_hparams)


# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath= './test_models/MP_20/',
    filename='{epoch}',         # Save the checkpoint after every epoch
    save_top_k=-1,              # Set to -1 to save all checkpoints
    save_last=True,             # Save the last model too, useful for resuming
    every_n_train_steps = 5000,    # Save every epoch (assuming you're validating every epoch)
    verbose=False                # Print save messages for debugging
)

trainer = pl.Trainer(accelerator = "gpu", 
                     devices = [1],
                     max_epochs= 10,
                     callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate = 1)],
                    #  strategy = 'ddp_find_unused_parameters_true',  # multi-GPU training
                     )

trainer.fit(model= chggen, datamodule= datamodule)

trainer.save_checkpoint("./test_models/MP_20/trainer_MP20.ckpt")
print("Done")
