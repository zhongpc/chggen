from __future__ import annotations
from types import SimpleNamespace
import argparse
import sys

import torch
import numpy as np
import os
import multiprocessing as mp

from chggen.pl_modules.model import CHGGen
from chggen.common.data_utils import get_scaler, mkdir, get_pymatgen_structure

from pymatgen.core import Structure, Composition, Element, Lattice
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import StructOptimizer

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pandas as pd

# from e_hull_calculator import EHullCalculator

import time
from datetime import datetime

import matplotlib.pyplot as plt




def convert_string_to_list(string):
    try:
        # Safely evaluate the string as a Python literal (list)
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        # Handle the case where the string is not a valid list
        return []
    




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
		    lengths=all_lengths,
		    angles=all_angles,
		    composition=all_atom_types,
		    num_atoms=all_num_atoms,
		    ld_kwargs = ld_kwargs,
		)

    s_list = get_pymatgen_structure(
            lengths = results['lengths'],
            angles = results['angles'],
            num_atoms = results['num_atoms'],
            frac_coords = results['frac_coords'],
            atom_types = results['atom_types'],
        )


    return s_list, results



def run_langevin_fromLattice(model, 
                             comp_str, # the string of composition
                             atom_volume, # avg atom volume
                             gen_kwargs,
                             ld_kwargs):
    

    num_atoms = []
    all_atom_types = []
    
    DEVICE = model.device

    NUM_GEN = gen_kwargs.num_gen
    VOL_ATOM = atom_volume

    for _ in range(NUM_GEN):
        comp = Composition(comp_str)
        num_atoms_formula = int(comp.num_atoms)


        num_atoms.append( int(num_atoms_formula * gen_kwargs.num_cell) )

        atom_types = []
        comp_dict = comp.as_dict()
        for key in comp_dict:
            amount = comp_dict[key]
            element = Element(key)
            atom_types += [element.Z] * int(amount)
        atom_types = atom_types * gen_kwargs.num_cell
        all_atom_types += atom_types
        
    volumes = np.array(num_atoms) * VOL_ATOM 
    lengths = volumes**(1/3)
    
        
    num_atoms = torch.tensor(num_atoms, device = DEVICE)

    cur_atom_types = torch.tensor(all_atom_types, device = DEVICE)
    angles = torch.ones((len(num_atoms), 3), device = DEVICE) * 90
    lengths =  torch.tensor(lengths, device = DEVICE, dtype = torch.float32).view(-1, 1)
    lengths = lengths.expand(-1, 3)
    
    print(lengths)


    rand_frac_coords = torch.rand((num_atoms.sum(), 3), device= DEVICE, requires_grad = False)
    

    ###  return the results ###
    results = model.langevin_dynamics(
        lengths=lengths,
        angles=angles,
        composition=cur_atom_types,
        num_atoms=num_atoms,
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

    return s_list, results



def generate_crystal(fv_dict):

    comp_str = fv_dict['formula']
    atom_volume = fv_dict['volume']
    
    device = torch.device(CUDA_DEVICE)
    chggen = CHGGen.load_from_checkpoint(checkpoint_path = CHGGEN_PATH, 
                                         map_location= device, strict= False)

    print(chggen.hparams)
    
    model = CHGNet.load()
    chg_relaxer = StructOptimizer(
        model=model,
        stress_weight = gen_kwargs.stress_weight,
        use_device= device)

    ### start score dynamics ###
    s_init_list, results = run_langevin_fromLattice(model= chggen, 
                                           comp_str= comp_str,
                                           atom_volume = atom_volume,
                                           gen_kwargs = gen_kwargs,
                                           ld_kwargs= ld_kwargs)
    
    
    for s_init in s_init_list:
        mkdir('./auto_generation/supercell/')
        s_init.to(filename = './auto_generation/supercell/super_' + s_init.composition.reduced_formula + '.cif')
        continue 

        s_init.to(filename = './auto_generation/supercell/POSCAR')

        structure = s_init

        prediction = model.predict_structure(structure)
        E0_tot = prediction['e'] * structure.num_sites
        Fmax = np.max(np.abs(prediction['f']))
        print(E0_tot, Fmax)

        atoms = AseAtomsAdaptor().get_atoms(structure)
        
        result = chg_relaxer.relax(atoms= atoms,
                                    fmax = 0.1,
                                    steps = 50000,
                                    relax_cell = True,
                                    verbose = True,
                                    # trajectory_path = None,
                )

        toten = result['trajectory'].energies[-1]
        print(E0_tot, toten, Fmax)

        s_relax = result["final_structure"]
        s_relax.to(filename = './auto_generation/supercell/superRelax_' + s_init.composition.reduced_formula + '.cif')

    print("Done")


    
    
PPD_PATH = "/home/zhongpc/chggen/host_gen/file_trans/2023-02-07-ppd-mp.pkl.gz"
CUDA_DEVICE = "cuda"
CHGGEN_PATH = "/home/zhongpc/hui_amorphous_pretrain_cond/epoch=4-val_loss=1.11.ckpt"

# e_hull_calculator = EHullCalculator(PPD_PATH)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default= "file.csv", help="a csv file containing formula and atom_volume")
    parser.add_argument("-n", "--numcell", type=int, default= 1, help="number of cell for the formula")    
    parser.add_argument("-d", "--device", type=str, default= "cuda", help="gpu device")
    args = parser.parse_args()

    # CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")
    ld_kwargs = SimpleNamespace(
        n_step_each = 5,            # Corrector
        min_sigma = 0.01,
        num_noise_level = 200,
        signal_to_noise_ratio = 0.4,
        save_traj = False,
        disable_bar = False,
        # step_lr = 1e-4,
    )


    gen_kwargs = SimpleNamespace(
        num_gen = 1, # number of structures generated from the cubic lattice
        num_mutation = 3, # number of mutations during the relax-generation iteration
        num_cell = 30, # args.numcell, # 2
        stress_weight = 1 / 160.21766208, # 0.2,
        ehull_cutoff = 0.06,
        )

    
    df_Na = pd.read_csv(args.input)
    
    valid_fv_list = []

    formula_list = df_Na.formula.values.tolist()
    volume_list = df_Na.avg_volume.values.tolist()
    
    for formula,volume in zip(formula_list, volume_list): 
        valid_fv_list.append({'formula': formula, 'volume': volume})

        
    # for debug
    # generate_crystal(valid_fv_list[1])
    print(torch.cuda.is_available())

    for ii in range(len(valid_fv_list)):
        generate_crystal(valid_fv_list[ii])

