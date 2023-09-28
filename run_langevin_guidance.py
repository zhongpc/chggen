from pathlib import Path
from datetime import datetime
from typing import List
from types import SimpleNamespace
import pickle
import numpy as np
import torch
from pymatgen.core import Structure, Lattice, Species, Element


from chggen.pl_data.dataset import CHGNetDataset
from chggen.pl_modules.model import CHGGen
from chggen.common.data_utils import get_scaler_from_data_list
from chggen.pl_modules.decoder import NequipDecoder

from torch_geometric.data import Data
from torch import nn
import torch.nn.functional as F

with open('./test_models/chggen.pk', 'rb') as fp:
    chggen = pickle.load(fp)

device = torch.device('cpu')
chggen.to(device = device)

# langevin dynamics
ld_kwargs = SimpleNamespace(n_step_each = 10,
                            step_lr = 1e-4,
                            min_sigma = 0,
                            save_traj = False,
                            disable_bar = False,
                            compute_force = False,
                            beta_c = 0, # property update rate
                            beta_f = 0, # atomic force update rate
                            )

z = torch.rand(1, 64, requires_grad= True, device = device)
results = chggen.langevin_dynamics_guidance(z = z, 
                                            prop_guidance = torch.tensor(-0.05, device= device), 
                                            ld_kwargs= ld_kwargs)

# save the results from langevin dynamics
lengths = results['lengths']
angles= results['angles']
num_atoms = results['num_atoms']
frac_coords = results['frac_coords']
atom_types = results['atom_types']

batch = torch.arange(len(num_atoms))
batch = batch.repeat_interleave(num_atoms)

for ii in range(len(num_atoms)):
    indices = torch.where(batch == ii)[0]
    print(ii, indices)
    if len(indices) == 0:
        continue
    
    
    Latt = Lattice.from_parameters(a = lengths[ii,0], b = lengths[ii,1], c = lengths[ii,2],
                                   alpha= angles[ii, 0], beta= angles[ii,1], gamma=angles[ii, 2])
                                   
    frac_ = frac_coords[indices]
    type_ = atom_types[indices]
    species_ = [Element.from_Z(ele_Z) for ele_Z in type_]
    
    s_gen = Structure(lattice= Latt , species= species_, coords= frac_.detach().numpy(),
                      to_unit_cell=False,coords_are_cartesian=False);
    print(s_gen.composition)
    s_gen.to(filename= './test_models/structures/prop_guidance_' + str(ii) + '.cif')
    


print("Done")
