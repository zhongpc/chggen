import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from chggen.common.data_utils import mkdir
from chggen.pl_data.dataset import CHGNetDataset
from chggen.pl_data.datamodule import CrystDataModule
from chggen.pl_modules.model import CHGGen


ROOT = "./data/trained_models/mp_pretrain_uncond/"
mkdir(ROOT)

# Initialize dataset.
train_dataset = CHGNetDataset(
    path=  "./data/debug.csv", # for debug purpose  # './data/mp/mp_pretrain_train.csv', 
    name = 'train_MP',
    prop_list = [],
)

val_dataset = CHGNetDataset(
    path= "./data/debug.csv", # for debug purpose # './data/mp/mp_pretrain_val.csv',
    name = 'val_MP',
    prop_list = [],
)


datamodule = CrystDataModule(
    train_dataset= train_dataset,
    val_dataset= val_dataset,
    num_workers=32,
    batch_size= 4,
)

# Initialize model.
model_hparams = {'latent_dim': 64,           # Model dimension.
                'hidden_dim': 64,
                'predict_property': False,  # Property guidance.
                'property_dim': 1,
                'fc_num_layers': 2,
                'sigma_begin': 10.0,        # Noise level.
                'sigma_end': 0.001,
                'num_noise_level': 200,
                'cost_coord': 1.0,          # Loss weight.
                'cost_property': 0,
                'decoder': 'nequip_cond',     # Decoder type.
                'max_neighbors': 60,
                'cutoff': 7.,
                'irreps_node_x': '32x0e',
                'irreps_node_z': '32x0e', # must be the same as element embedding to make sure the cond_addition works.
                'irreps_hidden': '64x0e + 32x1e + 16x2e',
                'irreps_edge': '1x0e + 2x1e + 4x2e',
                'num_convs': 3,             # Number of convolutional layers.
                'num_element_emb': 32, # number of element embedding.
                'num_radical_emb': 32, # number of radical embedding.
                'radial_neurons': [32,64], # used for the radical MLP to form the radial embedding.
                'lr': 1e-3,
                'lr_scheduler': 'exp_decay',# Learning rate scheduler.
                'lr_shrink': 0.01,          # Learning rate shrink.
                'gamma': 2, # the ration control unconditional and conditional generation
                'if_linear': True,
                'num_batch': len(datamodule.train_dataloader()),    # For warmup scheduler.
                }


chggen = CHGGen(hparams_dict=model_hparams)

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath= ROOT,
    filename='{epoch}-{val_loss:.2f}',              # Save the checkpoint after every epoch
    monitor='val_loss',                             # Monitor the validation loss
    mode='min',                                     # Save the model with the minimum validation loss
    save_top_k=1,                                   # Set to -1 to save all checkpoints
    save_last=True,                                 # Save the last model too, useful for resuming
)

trainer = pl.Trainer(
    accelerator = "gpu", 
    devices = [0],
    max_epochs= 50,
    callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate = 100)],
    log_every_n_steps=100,
    gradient_clip_val=0.1,
    default_root_dir= ROOT,
    # strategy = 'ddp_find_unused_parameters_true',  # multi-GPU training
)

trainer.fit(model= chggen, datamodule= datamodule)

print("Done")
