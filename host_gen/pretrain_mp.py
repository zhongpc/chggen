import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from chggen.common.data_utils import mkdir
from chggen.pl_data.dataset import CHGNetDataset
from chggen.pl_data.datamodule import CrystDataModule
from chggen.pl_modules.model import CHGGen



mkdir("./data/test_models/mp_pretrain/")

# Initialize dataset.
train_dataset = CHGNetDataset(
    path= './data/dataset/mp/mp_pretrain_train.csv', # mp_train.csv for fine-tuning
    name = 'train_MP',
    prop_list = [],
)

val_dataset = CHGNetDataset(
    path= './data/dataset/mp/mp_pretrain_val.csv', # mp_val.csv for fine-tuning
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
model_hparams ={'latent_dim': 64,           # Model dimension.
                'hidden_dim': 64, 
                'predict_property': False,  # Property guidance. 
                'property_dim': 1, 
                'fc_num_layers': 2, 
                'sigma_begin': 10.0,        # Noise level.
                'sigma_end': 0.001,    
                'cost_coord': 1.0,          # Loss weight.
                'cost_property': 0,
                'decoder': 'nequip_ee',    # Decoder type.: 'nequip_ee': element embedding or 'nequip_pte' periodic table embedding
                'lr': 1e-3,                 
                'num_batch': len(datamodule.train_dataloader()),    # For warmup scheduler.
                }

chggen = CHGGen(hparams_dict=model_hparams)

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath= './data/test_models/mp_pretrain/',
    filename='{epoch}',         # Save the checkpoint after every epoch
    save_top_k=-1,              # Set to -1 to save all checkpoints
    save_last=True,             # Save the last model too, useful for resuming
    every_n_train_steps = 5000, # Save every epoch (assuming you're validating every epoch)
    verbose=False               # Print save messages for debugging
)

trainer = pl.Trainer(
    accelerator = "gpu", 
    devices = [6],
    max_epochs= 50,
    callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate = 1)],
    log_every_n_steps=100,
    gradient_clip_val=0.5,
    default_root_dir='./data/test_models/mp_pretrain/',
    # strategy = 'ddp_find_unused_parameters_true',  # multi-GPU training
)

trainer.fit(model= chggen, datamodule= datamodule)

trainer.save_checkpoint("./data/test_models/mp_pretrain/trainer_mp_pretrain.ckpt")
print("Done")