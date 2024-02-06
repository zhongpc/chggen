from __future__ import annotations

import sys
sys.path.append('../../')

import time
from types import SimpleNamespace
from tqdm import tqdm

import pandas as pd
import numpy as np

import torch

from chggen.pl_modules.model import CHGGen
from chggen.common.data_utils import get_pymatgen_structure
from chggen.common.data_utils import ( 
    cart_to_frac_coords, 
    frac_to_cart_coords, 
)

from pymatgen.core import Structure, Element
from pymatgen.io.cif import CifWriter



def get_data(df, num, device):
    s0 = Structure.from_str(df['cif'][num], fmt='cif', frac_tolerance=0, site_tolerance=0)
    # Obtain number of Li atoms in the structure.
    num_insert_ion = df['num_insert_ion'][num]

    insert_ion = Element('Li')

    cur_atom_types = []
    atom_masks = []

    # Add atom types and atom masks for host atoms.
    for site in s0.sites:
        cur_atom_types.append(site.specie.Z)
        atom_masks.append(0)

    # Add atom types and atom masks for inserted ions.
    cur_atom_types += [insert_ion.Z] * num_insert_ion
    cur_atom_types = torch.tensor(cur_atom_types, device = device, dtype = torch.float32)
    atom_masks += [1] * num_insert_ion
    atom_masks = torch.tensor(atom_masks, device = device).view(-1, 1) #  torch.float32)
    
    # Add random coordinates for host and inserted ions.
    insert_frac_coords = torch.rand(num_insert_ion, 3, requires_grad = False, device = device, dtype = torch.float32)
    cur_frac_coords = torch.tensor(s0.frac_coords, device = device)
    cur_frac_coords = torch.concat((cur_frac_coords, insert_frac_coords), axis = 0)
    cur_frac_coords = torch.tensor(cur_frac_coords, dtype = torch.float32)

    # Cell information.
    num_atoms = torch.tensor([len(cur_atom_types)], device = device)
    angles = torch.tensor([s0.lattice.angles], device = device)
    lengths = torch.tensor([s0.lattice.lengths], device = device)
    
    return cur_atom_types, atom_masks, cur_frac_coords, num_atoms, angles, lengths

def get_data(df, num, device):
    s0 = Structure.from_str(df['cif'][num], fmt='cif', frac_tolerance=0, site_tolerance=0)

    cur_atom_types = []
    for site in s0.sites:
        cur_atom_types.append(site.specie.Z)
    cur_atom_types = torch.tensor(cur_atom_types, device = device, dtype = torch.float32)
    
    # Add random coordinates for host and inserted ions.
    cur_frac_coords = torch.rand(len(cur_atom_types), 3, requires_grad = False, device = device, dtype = torch.float32)
    cur_frac_coords = torch.tensor(cur_frac_coords, dtype = torch.float32)

    # Cell information.
    num_atoms = torch.tensor([len(cur_atom_types)], device = device)
    angles = torch.tensor([s0.lattice.angles], device = device)
    lengths = torch.tensor([s0.lattice.lengths], device = device)
    
    return cur_atom_types, cur_frac_coords, num_atoms, angles, lengths

# Construct dataloader
def get_batch_data(df, id_start, id_end, device):
    cur_atom_types, cur_frac_coords, num_atoms, angles, lengths = \
        zip(*[get_data(df, i, device) for i in range(id_start, id_end)])

    # Construct batch data
    cur_atom_types = torch.cat(cur_atom_types, axis = 0)
    cur_frac_coords = torch.cat(cur_frac_coords, axis = 0)
    num_atoms = torch.cat(num_atoms, axis = 0)
    angles = torch.cat(angles, axis = 0)
    lengths = torch.cat(lengths, axis = 0)
    
    return cur_atom_types, cur_frac_coords, num_atoms, angles, lengths
