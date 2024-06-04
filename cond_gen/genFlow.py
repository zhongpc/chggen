from chggen.common.sample_utils import CSP_Generator
from chggen.common.data_utils import mkdir
from chggen.common.sample_utils import get_inpaint_data_fromHost
from chggen.common.sample_utils import get_batch_inpaint_data_fromHost
from chggen.common.sample_utils import get_coarse_grain_framework, filter_nan_structure, compute_ewald_energy_single_structure
from chggen.common.e_hull_calculator import EHullCalculator

from types import SimpleNamespace
import numpy as np

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure, Composition, Element, Lattice
from pymatgen.io.cif import CifWriter

import argparse
import pandas as pd
import os
import time
from datetime import datetime

def relax_structures(csp, s_list):
    s_list_relax = []
    for s in s_list:
        atoms = AseAtomsAdaptor().get_atoms(s)

        result = csp.relaxer.relax(atoms= atoms,
                            fmax = 0.1,
                            steps = 2000,
                            relax_cell = True,
                            verbose = True,
                            # trajectory_path = None,
        )
        s_relax = result["final_structure"]
        s_list_relax.append(s_relax)

    return s_list_relax

def coarse_grain_framework(s_list, species_to_remove):
    
    host_structure_list = []
    num_intercalat_list = []

    for s in s_list:
        analyzer_asGen = SpacegroupAnalyzer(structure= s, symprec= 0.15, angle_tolerance= 15)
        symbol_asGen = analyzer_asGen.get_space_group_symbol()
        print("As generated spacegroup: ", symbol_asGen)

        s_frame, symbol_frame, num_species = get_coarse_grain_framework(s, species_to_remove= species_to_remove)
        s_frame = s_frame.get_primitive_structure()

        if symbol_frame in ['P1', 'P-1', 'Pm']: 
            continue

        host_structure_list.append(s_frame)
        num_intercalat_list.append(int(num_species))

    return host_structure_list, num_intercalat_list



def refine_structure(s_list, symprec= 0.2, angle_tolerance= 15):
    s_list_refine = []

    for s in s_list:
        analyzer_inpaint = SpacegroupAnalyzer(structure= s, symprec= symprec, angle_tolerance= angle_tolerance)
        symbol_inpaint = analyzer_inpaint.get_space_group_symbol()

        if symbol_inpaint== 'P1' or symbol_inpaint== 'P-1':
            continue
        else:
            print(symbol_inpaint)
            s_conventional_unit = analyzer_inpaint.get_conventional_standard_structure()
            s_list_refine.append(s_conventional_unit)

    return s_list_refine


# %%
def get_save_dict_list(csp, s_list):
    save_dict_list = []
    for ii, s in enumerate(s_list):
        atoms = AseAtomsAdaptor().get_atoms(s)

        analyzer_inpaint = SpacegroupAnalyzer(structure= s, symprec= 0.01, angle_tolerance= 5)
        symbol_inpaint = analyzer_inpaint.get_space_group_symbol()
        prediction = csp.chgnet.predict_structure(s)
        E0_atom = prediction['e'] 
        F_max = np.max(np.abs(prediction['f']))


        result = csp.relaxer.relax(atoms= atoms,
                            fmax = 0.1,
                            steps = 2000,
                            relax_cell = True,
                            verbose = True,
                            # trajectory_path = None,
        )
        s_relax = result["final_structure"]
        toten = result['trajectory'].energies[-1]

        analyzer_relax = SpacegroupAnalyzer(structure= s_relax, symprec= 0.15, angle_tolerance= 15)
        symbol_relax = analyzer_relax.get_space_group_symbol()
        s_relax_refine = analyzer_relax.get_refined_structure()

        save_dict ={'material_id': hex(int(time.time()*1e8)),
                    'formula': s_relax.composition.reduced_formula,
                    's_inpaint_asGen_cif': str(CifWriter(s)),
                    'spacegroup_asGen': symbol_inpaint,
                    's_relax_refine_cif': str(CifWriter(s_relax_refine)),
                    'spacegroup_refine': symbol_relax,
                    's_relax_cif': str(CifWriter(s_relax)),
                    'Fmax_chgnet': F_max, 
                    'E0_chgnet_atom': E0_atom,
                    'E_chgnet_atom': toten / s_relax.num_sites,
                    'energy': toten,
                    'structure': s_relax,
                    # 'lattice_type': type_init,
                    # 'mutation': mutation
                    }
        save_dict_list.append(save_dict)
    return save_dict_list

# %%


# %%
def main(csp, 
         chemical_formula, atomic_volume, species,
         gen_kwargs, ld_kwargs):

    #####  Generate seven different bravis lattices via diffusion  #####
    print("--"*5 + "Generate seven different bravis lattices via diffusion" + "--"*5)
    s_list_Bravis = csp.generate_structures_from_Bravis(comp_str= chemical_formula, atom_volume= atomic_volume,
                                                        gen_kwargs=gen_kwargs, ld_kwargs=ld_kwargs, )
    s_list_Bravis = filter_nan_structure(s_list_Bravis)

    #####  Relax the generated structures from Bravis lattices #####
    print("--"*5 + "Relax the generated structures from Bravis lattices" + "--"*5)
    s_list_relax = relax_structures(csp, s_list_Bravis)

    #####  Coarse grain the relaxed structures and get the host structures#####
    print("--"*5 + "Coarse grain the relaxed structures" + "--"*5)
    host_structure_list, num_intercalat_list = coarse_grain_framework(s_list_relax, species_to_remove= species)

    #####  Run inpaiting based on the host structures  #####
    print("--"*5 + "Run inpainting based on the host structures" + "--"*5)

    if species in chemical_formula:
        s_list_inpaint = csp.generate_from_host_structure(host_structure_list= host_structure_list,
                                 num_intercalant_list= num_intercalat_list,
                                 ld_kwargs=ld_kwargs, 
                                 species= species)
    else:
        print("The species is not in the chemical formula, will not run inpainting with framework. As-generated structures will be used.")
        s_list_inpaint = host_structure_list
    
    #####  Refine the inpainted structures  #####
    print("--"*5 + "Run inpaiting based on the host structures" + "--"*5)
    s_list_inpaint_conventional_unit = refine_structure(s_list_inpaint, symprec= 0.2, angle_tolerance= 15)

    save_dict_list = get_save_dict_list(csp, s_list_inpaint_conventional_unit)
    save_dict_list = csp.e_hull_calculator.get_e_hull(save_dict_list)
    
    #####  Post process & compute phase stability & save structural information  #####
    print("--"*5 + "Post process and save structural information" + "--"*5)
    df = pd.DataFrame(columns=['material_id', 'formula', 'e_hull',
                              's_inpaint_asGen_cif', 'spacegroup_asGen', 
                              's_relax_refine_cif', 'spacegroup_refine', 's_relax_cif', 
                              'Fmax_chgnet', 'E0_chgnet_atom', 'E_chgnet_atom', 'energy', 'structure'])

    for save_dict in save_dict_list:
        row_df = pd.DataFrame([save_dict])
        df = pd.concat([df, row_df], ignore_index=True)
    
    df = df.sort_values('e_hull')
    df['e_hull_diff'] = df.groupby('spacegroup_asGen')['e_hull'].diff()
    df = df[(df['e_hull_diff'].isna()) | (df['e_hull_diff'].abs() >= 0.002)]
    
    df = df.drop('e_hull_diff', axis=1)
    df = df.drop('energy', axis=1)
    df = df.drop('structure', axis=1)

    CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")
    ROOT_DIR = './files/' + chemical_formula 

    if os.path.exists(ROOT_DIR + '/generation_summary'+ '_' + CURRENT_DATE +'.csv'):
        df.to_csv(ROOT_DIR + '/generation_summary' + '_' + CURRENT_DATE +'.csv', mode='a', index=False, header=False)
    else:
        mkdir(ROOT_DIR)
        df.to_csv(ROOT_DIR + '/generation_summary' + '_' + CURRENT_DATE +'.csv', index=False, header=True)

    print("Done")

    return df



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chemical_formula", type=str, default= "ZnSP2S5", help="chemical formula")
    parser.add_argument("-v", "--atomic_volume", type=float, default= 24, help="atomic volume")
    parser.add_argument("-d", "--device", type=str, default= "cuda", help="gpu device")
    args = parser.parse_args()
    
    
    ### Define the kwargs for the generation ###
    ld_kwargs = SimpleNamespace(
            n_step_each = 5,            # Corrector
            min_sigma = 0.01,
            num_noise_level = 200,
            signal_to_noise_ratio = 0.4,
            save_traj = False,
            disable_bar = False,
        )
    gen_kwargs = SimpleNamespace(
            num_gen = 3, # number of structures generated from the cubic lattice
            num_mutation = 2, # number of mutations during the relax-generation iteration
            num_cell = 1, # number of times to the formula
            ehull_cutoff = 0.06,
            )

    ### Define the CSP file and patched_phase diagram ###
    csp = CSP_Generator(chggen_path = "./files/cut_7_conv_3_epoch=27-val_loss=0.87.ckpt",
                        device='cuda:6')

    csp.load_e_hull_calculator(ppd_path= "/home/zhongpc/chggen/host_gen/file_trans/2023-02-07-ppd-mp.pkl.gz")
    

    for num_inpaint in range(3):
        print("--"*5 + "Start generating structures with inpainting" + "--"*5)
        df_asGen = main(csp= csp, chemical_formula= args.chemical_formula, atomic_volume = args.atomic_volume, 
                        species = 'Zn', # set the same to enable inpainting, set different to use as-generated structures
                        gen_kwargs = gen_kwargs, ld_kwargs = ld_kwargs)
    
    for num_asGen in range(2):
        print("--"*5 + "Start generating structures without inpainting" + "--"*5)    
        df_inpaint = main(csp= csp, chemical_formula= args.chemical_formula, atomic_volume = args.atomic_volume, 
                        species = 'Xe', # set the same to enable inpainting, set different to use as-generated structures
                        gen_kwargs = gen_kwargs, ld_kwargs = ld_kwargs)
    
    # Example: python genFlow.py -d cuda:6 -c ZnSP2S5 -v 24

