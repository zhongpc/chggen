from __future__ import annotations
from types import SimpleNamespace

import torch
import numpy as np
import os

from chggen.pl_modules.model import CHGGen
from chggen.common.data_utils import get_scaler, mkdir, get_pymatgen_structure

from pymatgen.core import Structure, Composition, Element



def run_langevin_from_structures(model, 
                                 s_init_list,
                                 ld_kwargs):
    
    all_lengths = []
    all_angles = []
    all_num_atoms = []
    all_atom_types = []

    device = model.device

    for s_init in s_init_list:    
        lengths = s_init.lattice.lengths
        angles = s_init.lattice.angles
    
        composition = s_init.composition
        atom_types = []
        for key in composition:
            atom_types +=  [key.Z] * int(composition[key]) 
    
        all_atom_types += atom_types
        all_num_atoms.append(s_init.num_sites)
        all_lengths.append(lengths)
        all_angles.append(angles)

    all_atom_types = torch.tensor(all_atom_types, device = device)
    all_angles = torch.tensor(all_angles, device = device)
    all_lengths =  torch.tensor(all_lengths, device = device)
    all_num_atoms = torch.tensor(all_num_atoms, device = device)

    ###  return the results ###
    results = model.langevin_dynamics(
        lengths= all_lengths,
        angles= all_angles,
        composition= all_atom_types,
        num_atoms= all_num_atoms,
        ld_kwargs = ld_kwargs,
    )
    
    ### convert to pymatgen structure ###
    lengths = results['lengths']
    angles = results['angles']
    num_atoms = results['num_atoms']
    frac_coords = results['frac_coords']
    atom_types = results['atom_types']
    
    s_list = get_pymatgen_structure(
        lengths = lengths,         
        angles = angles,
        num_atoms = num_atoms,
        frac_coords = frac_coords,
        atom_types = atom_types,
    )

    return s_list



if __name__ == "__main__":

    device = torch.device('cuda')
    checkpoint_path = "./test_models/MP_20/last.ckpt"
    chggen = CHGGen.load_from_checkpoint(checkpoint_path = checkpoint_path, map_location= device)

    root_dir = './auto_generation/LZC_x1/'
    file_list = os.listdir(root_dir)
    s_init_list = []

    for file in file_list:
        if 'cif' in file:
            pass
        else:
            continue
        s_init = Structure.from_file(root_dir + file)
        s_init_list.append(s_init)


    ld_kwargs = SimpleNamespace(
        n_step_each = 1,
        step_lr = 1e-4,
        min_sigma = 0,
        save_traj = False,
        disable_bar = False, )

    s_list = run_langevin_from_structures(model= chggen, s_init_list= s_init_list, ld_kwargs= ld_kwargs)

    mkdir('./auto_generation/LZC_x1/second_diff/')

    count = 0
    for s_init, s_diff in zip(s_init_list, s_list):
        s_init.to(filename= './auto_generation/LZC_x1/second_diff/' + str(count) +'_first' + '.cif')
        s_diff.to(filename= './auto_generation/LZC_x1/second_diff/' + str(count) +'_second' + '.cif')
        count += 1
        



