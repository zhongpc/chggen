from __future__ import annotations
from types import SimpleNamespace
import argparse

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

from chggen.common.e_hull_calculator import EHullCalculator

import time
from datetime import datetime

import matplotlib.pyplot as plt



import numpy as np

def generate_lattice_cell(lattice_type, volume):
    if lattice_type == 'cubic':
        a = np.cbrt(volume)
        lattice_params = (a, a, a, 90, 90, 90)
    elif lattice_type == 'tetragonal':
        a = np.sqrt(volume / 1.5)
        c = 1.5 * a
        lattice_params = (a, a, c, 90, 90, 90)
    elif lattice_type == 'orthorhombic':
        a = np.cbrt(volume / 1.5)
        b = 1.5 * a
        c = a
        lattice_params = (a, b, c, 90, 90, 90)
    elif lattice_type == 'monoclinic':
        a = np.cbrt(volume / 1.5 / 0.984)
        b = 1.5 * a
        c = a
        beta = 100
        lattice_params = (a, b, c, 90, beta, 90)

    elif lattice_type == 'aP':
        a = np.cbrt(volume / 0.963) # scale = 0.963
        b = a
        c = a
        alpha = 80
        beta = 85
        gamma = 100
        lattice_params = (a, b, c, alpha, beta, gamma)
    elif lattice_type == 'hP':
        a = np.sqrt(volume / (np.sqrt(3) / 2) / 1.5)
        c = a * 1.5
        lattice_params = (a, a, c, 90, 90, 120)
    elif lattice_type == 'hR':
        a = np.cbrt(volume / 0.707)
        alpha = 60
        lattice_params = (a, a, a, alpha, alpha, alpha)
    else:
        raise ValueError("Invalid lattice type")
    
    return lattice_params




def convert_string_to_list(string):
    try:
        # Safely evaluate the string as a Python literal (list)
        return ast.literal_eval(string)
    except (ValueError, SyntaxError):
        # Handle the case where the string is not a valid list
        return []
    




def run_SDE_from_structures(model, 
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
    results = model.reverse_SDE(
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
    results = model.reverse_SDE(
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



def run_SDE_fromBravis(model, 
                             comp_str, # the string of composition
                             atom_volume, # avg atom volume
                             gen_kwargs,
                             ld_kwargs):
    

    num_atoms = []
    all_atom_types = []
    
    DEVICE = model.device
    VOL_ATOM = atom_volume

    lengths = []
    angles = []

    # generate the lattice cell

    comp = Composition(comp_str)
    num_atoms_formula = int(comp.num_atoms)
    num_atoms = [num_atoms_formula] * 7 # for 7 different lattice types

    for lattice_type in ['cubic', 'tetragonal', 'orthorhombic', 'monoclinic', 'aP', 'hP', 'hR']:
        lattice_params = generate_lattice_cell(lattice_type, num_atoms_formula * VOL_ATOM)
        lengths.append(lattice_params[:3])
        angles.append(lattice_params[3:])

        atom_types = []
        comp_dict = comp.as_dict()
        for key in comp_dict:
            amount = comp_dict[key]
            element = Element(key)
            atom_types += [element.Z] * int(amount)
        atom_types = atom_types * gen_kwargs.num_cell
        all_atom_types += atom_types
        

        
    num_atoms = torch.tensor(num_atoms, device = DEVICE)
    print("num_atoms", num_atoms)

    cur_atom_types = torch.tensor(all_atom_types, device = DEVICE)
    print("cur_atom_types", cur_atom_types)

    lengths = torch.tensor(lengths, device = DEVICE, dtype = torch.float32).view(-1, 3)
    # lengths = lengths.expand(-1, 3)

    angles = torch.tensor(angles, device = DEVICE, dtype = torch.float32).view(-1, 3)
    # angles = angles.expand(-1, 3)
    
    print(lengths)
    print(angles)


    rand_frac_coords = torch.rand((num_atoms.sum(), 3), device= DEVICE, requires_grad = False)
    

    ###  return the results ###
    results = model.reverse_SDE(
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
    
    model = CHGNet.load()
    chg_relaxer = StructOptimizer(
        model=model,
        stress_weight = gen_kwargs.stress_weight,
        use_device= device)

    ### start score dynamics ###
    s_init_list, results = run_SDE_fromBravis(model= chggen, 
                                           comp_str= comp_str,
                                           atom_volume = atom_volume,
                                           gen_kwargs = gen_kwargs,
                                           ld_kwargs= ld_kwargs)
    
    
    se_list = []
    init_type_list = ['cubic', 'tetragonal', 'orthorhombic', 'monoclinic', 'aP', 'hP', 'hR']
    

    for s_init, type_init in zip(s_init_list, init_type_list): # loop over different cubic lattice generated structures
        # init the structure energy list
        df = pd.DataFrame(columns=['material_id', 'formula', 'lattice_type', 'mutation', 
                                'spacegroup_init', 's_refine_init_cif',  # as generated structure information 
                                'spacegroup_refine', 's_refine_relax_cif', # relaxed by pretrained chgnet structure information
                                's_init_cif', 's_relax_cif', 'E0_chgnet', 'E_chgnet']) # energy information from chgnet



        for mutation in range(gen_kwargs.num_mutation):
            if mutation ==0:
                structure = s_init
            else:
                s_prim = s_CG.get_primitive_structure()
                s_list, results = run_SDE_from_structures(model= chggen, s_init_list= [s_prim], ld_kwargs= ld_kwargs)
                structure = s_list[0]


            try:
                prediction = model.predict_structure(structure)
                E0_tot = prediction['e'] * structure.num_sites
                Fmax = np.max(np.abs(prediction['f']))

                atoms = AseAtomsAdaptor().get_atoms(structure)

                result = chg_relaxer.relax(atoms= atoms,
                                    fmax = 0.1,
                                    steps = 2000,
                                    relax_cell = True,
                                    verbose = True,
                                    # trajectory_path = None,
                )
            except:
                print("CHGNet prediction failed")
                E0_tot = np.nan
                Fmax = np.nan
                continue



            s_relax = result["final_structure"]
            toten = result['trajectory'].energies[-1]


            ### analyze space group and refine ###
            analyzer_init = SpacegroupAnalyzer(structure= structure, symprec= 0.15, angle_tolerance= 15)
            analyzer = SpacegroupAnalyzer(structure= s_relax, symprec= 0.15, angle_tolerance= 15)

            sym_list = [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0]
            angle_list = [10, 15, 15, 15, 20, 30, 30]
            s_CG_list = []
            for symprec, angle_tolerance in zip(sym_list, angle_list):
                analyzer_CG = SpacegroupAnalyzer(structure= s_relax, symprec= symprec, angle_tolerance= angle_tolerance)
                try:
                    s_CG = analyzer_CG.get_conventional_standard_structure()
                    print("symmetry: ", symprec, "angle_tolerance: ", angle_tolerance)
                    print("CG spacegroup: ", analyzer_CG.get_space_group_symbol()) 
                    if analyzer_CG.get_space_group_symbol() == 'P1' or analyzer_CG.get_space_group_symbol() == 'P-1':
                        continue
                    else:
                        break
                except:
                    print("failed to analyze CG space group")
                    continue

            try:
                print()
                print("init spacegroup: ", analyzer_init.get_space_group_symbol())
                print("spacegroup: ",  analyzer.get_space_group_symbol())

                s_refine_init = analyzer_init.get_conventional_standard_structure()
                symbol_init = analyzer_init.get_space_group_symbol()

                s_refine = analyzer.get_conventional_standard_structure()
                symbol_refine = analyzer.get_space_group_symbol()
            except:
                s_refine = s_relax.copy()
                symbol = 'P1'
                print("failed to analyze space group")

            # if symbol == 'P1':
            #     continue
            
            save_dict = {'material_id': hex(int(time.time()*1e8)),
                         'formula': structure.composition.reduced_formula,
                         's_refine_init_cif': str(CifWriter(s_refine_init)),
                         'spacegroup_init': symbol_init,
                         's_refine_relax_cif': str(CifWriter(s_refine)),
                         'spacegroup_refine': symbol_refine,
                         's_init_cif': str(CifWriter(structure)),
                         's_relax_cif': str(CifWriter(s_relax)),
                         'Fmax_chgnet': Fmax, 
                         'E0_chgnet': E0_tot,
                         'E_chgnet': toten,
                         'lattice_type': type_init,
                         'mutation': mutation
                         }
            row_df = pd.DataFrame([save_dict])
            df = pd.concat([df, row_df], ignore_index=True)
 
            
        CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")
        ROOT_DIR = './auto_generation/' + args.output + '/'+ CURRENT_DATE

        if os.path.exists(ROOT_DIR + '/mutation_gen_summary.csv'):
            df.to_csv(ROOT_DIR + '/mutation_gen_summary.csv', mode='a', index=False, header=False)
        else:
            mkdir(ROOT_DIR)
            df.to_csv(ROOT_DIR + '/mutation_gen_summary.csv', index=False, header=True)

    print("Done")


    
    
PPD_PATH = "/home/zhongpc/chggen/host_gen/file_trans/2023-02-07-ppd-mp.pkl.gz"
CUDA_DEVICE = "cuda:6"
CHGGEN_PATH = "./files/cut_7_conv_3_epoch=27-val_loss=0.87.ckpt"

# e_hull_calculator = EHullCalculator(PPD_PATH)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default= "file.csv", help="a csv file containing formula and atom_volume")
    parser.add_argument("-o", "--output", type=str, default= "genStable_structures", help="the name of output directory")
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
        step_lr = 1e-4,
    )


    gen_kwargs = SimpleNamespace(
        num_gen = 1, # number of structures generated from the cubic lattice
        num_mutation = 2, # number of mutations during the relax-generation iteration
        num_cell = args.numcell, # 2
        stress_weight = 1 / 160.21766208, # 0.2,
        ehull_cutoff = 0.06,
        )

    
    df_read = pd.read_csv(args.input)
    
    valid_fv_list = []

    formula_list = df_read.formula.values.tolist()
    volume_list = df_read.avg_volume.values.tolist()
    
    for formula,volume in zip(formula_list, volume_list): 
        valid_fv_list.append({'formula': formula, 'volume': volume})

        
    # for debug
    # generate_crystal(valid_fv_list[1])
    print(torch.cuda.is_available())

    for ii in range(len(valid_fv_list)):
        generate_crystal(valid_fv_list[ii])

