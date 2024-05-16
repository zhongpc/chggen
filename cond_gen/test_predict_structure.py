import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

from chggen.common.data_utils import mkdir
from chggen.pl_data.dataset import CHGNetDataset
from chggen.pl_data.datamodule import CrystDataModule
from chggen.pl_modules.model import CHGGen

from chggen.common.data_utils import mkdir, get_batch_tensor_from_structures, get_tensor_from_structure

from pymatgen.core import Structure

# Initialize model.
model_hparams ={'latent_dim': 64,           # Model dimension.
                'hidden_dim': 64, 
                'predict_property': False,  # Property guidance. 
                'property_dim': 1, 
                'fc_num_layers': 2, 
                'sigma_begin': 10.0,        # Noise level.
                'sigma_end': 0.001,
                'num_noise_level': 200,
                'cost_coord': 1.0,          # Loss weight.
                'cost_property': 0,
                'decoder': 'nequip_ee',     # Decoder type.
                'max_neighbors': 60,
                'cutoff': 7.,
                'irreps_hidden': '32x0e + 32x1e',
                'irreps_edge': '32x0e + 32x1e + 16x2e',
                'num_convs': 4,             # Number of convolutional layers.
                'lr': 1e-3,                 
                'lr_scheduler': 'exp_decay',# Learning rate scheduler.
                'lr_shrink': 1,          # Learning rate shrink.
                # 'num_batch': len(datamodule.train_dataloader()),    # For warmup scheduler.
                }

chggen = CHGGen(hparams_dict=model_hparams)


### test the single structure version ###

structure = Structure.from_file('./MnZnO2.cif')
(cur_atom_types, cur_frac_coords, num_atoms, angles, lengths) = get_tensor_from_structure(structure)

pred_cart_coords = chggen.predict_structures(cur_atom_types = cur_atom_types, 
                                             cur_frac_coords = cur_frac_coords, 
                                             num_atoms = num_atoms, 
                                             lengths = lengths, 
                                             angles = angles)



### test the batch version ###
structure_list = [structure] * 10
batch_tensor = get_batch_tensor_from_structures(structure_list)

pred_cart_coords_batch = chggen.predict_structures(cur_atom_types = batch_tensor[0], 
                                              cur_frac_coords = batch_tensor[1], 
                                             num_atoms = batch_tensor[2], 
                                             lengths = batch_tensor[3], 
                                             angles = batch_tensor[4])


print(pred_cart_coords_batch.shape)
print("Done")
                                             
                                
