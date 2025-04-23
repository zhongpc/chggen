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
                            steps = 500,
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
        try:
            symbol_asGen = analyzer_asGen.get_space_group_symbol()
        except:
            print("Failed to get the asGen spacegroup")
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
        try:
            symbol_inpaint = analyzer_inpaint.get_space_group_symbol()
        except:
            print("Failed to get space group symbol")
            continue

        if symbol_inpaint== 'P1' or symbol_inpaint== 'P-1':
            continue
        else:
            print(symbol_inpaint)
            s_conventional_unit = analyzer_inpaint.get_conventional_standard_structure()
            s_list_refine.append(s_conventional_unit)

    return s_list_refine

def get_save_dict_list(csp, s_list, type, 
                       symprec= 0.1, # ajdustable to a higher value for accuracy
                       angle_tolerance= 10, # ajdustable to a higher value for accuracy
                       screen_spacegroup = True,
                       ):
    save_dict_list = []
    s_relax_list = []
    for ii, s in enumerate(s_list):

        ### check the minimum distance between atoms ###
        distance_matrix = s.distance_matrix
        distance_matrix += np.diag(np.ones(len(s.sites))) * 100
        min_distance = np.min(distance_matrix)

        if min_distance < 1:
            print("The minimum distance between atoms is less than 1 Angstrom, skip this structure")
            continue

        atoms = AseAtomsAdaptor().get_atoms(s)
        analyzer = SpacegroupAnalyzer(structure= s, 
                                      symprec= symprec, angle_tolerance= angle_tolerance)

        try:
            symbol_asGen = analyzer.get_space_group_symbol()
        except:
            print("Failed to get space group symbol")
            continue
        
        prediction = csp.chgnet.predict_structure(s)
        E0_atom = prediction['e'] 
        F_max = np.max(np.abs(prediction['f']))


        result = csp.relaxer.relax(atoms= atoms,
                            fmax = 0.1,
                            steps = 500,
                            relax_cell = True,
                            verbose = True,
                            # trajectory_path = None,
        )
        s_relax = result["final_structure"]
        s_relax_list.append(s_relax)

        toten = result['trajectory'].energies[-1]

        analyzer_relax = SpacegroupAnalyzer(structure = s_relax, 
                                            symprec = symprec, 
                                            angle_tolerance = angle_tolerance)
        try:
            symbol_relax = analyzer_relax.get_space_group_symbol()
            s_relax_refine = analyzer_relax.get_refined_structure()

            print("Relaxed spacegroup: ", symbol_relax)
        except:
            print("Failed to get the space group of s_relax")
            symbol_relax = 'P1'
            s_relax_refine = s_relax


        low_sym_condition = (symbol_relax == 'P1' or symbol_relax == 'P-1')
        if screen_spacegroup and low_sym_condition:
            continue

        save_dict ={'material_id': hex(int(time.time()*1e8)),
                    'formula': s_relax.composition.reduced_formula,
                    's_asGen_cif': str(CifWriter(s)),
                    'spacegroup_asGen': symbol_asGen,
                    's_relax_refine_cif': str(CifWriter(s_relax_refine)),
                    'spacegroup_refine': symbol_relax,
                    's_relax_cif': str(CifWriter(s_relax)),
                    'Fmax_chgnet': F_max, 
                    'E0_chgnet_atom': E0_atom,
                    'E_chgnet_atom': toten / s_relax.num_sites,
                    'type': type,
                    'energy': toten,
                    'structure': s_relax,
                    # 'lattice_type': type_init,
                    # 'mutation': mutation
                    }
        save_dict_list.append(save_dict)
    return save_dict_list, s_relax_list



def main(csp, 
         chemical_formula, atomic_volume, species,
         gen_kwargs, ld_kwargs):
    
    # count the number of structures implemented in the inpainting and de novo generation
    total_denovo = 0
    total_inpaint = 0

    #####  Generate seven different bravis lattices via diffusion  #####
    print("--"*5 + "Generate seven different bravis lattices via diffusion" + "--"*5)
    s_list_Bravis = csp.generate_structures_from_Bravis(comp_str= chemical_formula, atom_volume= atomic_volume,
                                                        gen_kwargs=gen_kwargs, ld_kwargs=ld_kwargs, 
                                                        ratio_list = [0.5, 0.75, 1, 1.25, 1.5])
    s_list_Bravis = filter_nan_structure(s_list_Bravis)

    #####  Relax the generated structures from Bravis lattices #####
    print("--"*5 + "Relax the generated structures from Bravis lattices" + "--"*5)
    # s_list_relax = relax_structures(csp, s_list_Bravis)
    # total_denovo += len(s_list_relax)

    # s_list_relax_conventional_unit = refine_structure(s_list_relax, symprec= 0.15, angle_tolerance= 15)
    save_dict_list, s_list_relax = get_save_dict_list(csp, s_list_Bravis, 
                                                      type= 'denovo')
    
    total_denovo += len(s_list_relax)

    #####  Coarse grain the relaxed structures and get the host structures#####
    if species in chemical_formula:
        print("--"*5 + "Coarse grain the relaxed structures" + "--"*5)
        # s_list_relax = [save_dict['structure'] for save_dict in save_dict_list]
        host_structure_list, num_intercalat_list = coarse_grain_framework(s_list_relax, 
                                                                          species_to_remove= species)
        
        #####  Run inpaiting based on the host structures  #####
        print("--"*5 + "Run inpainting based on the host structures" + "--"*5)
        s_list_inpaint = csp.generate_from_host_structure(host_structure_list= host_structure_list * args.num_inpaintings,
                                 num_intercalant_list= num_intercalat_list * args.num_inpaintings,
                                 ld_kwargs=ld_kwargs, 
                                 species= species)

        s_list_inpaint = filter_nan_structure(s_list_inpaint)

        total_inpaint += len(s_list_inpaint)
        #####  Refine the inpainted structures  #####
        # s_list_inpaint_conventional_unit = refine_structure(s_list_inpaint, symprec= 0.15, angle_tolerance= 15)
        save_dict_list_inpaint, _ = get_save_dict_list(csp, s_list_inpaint, type= 'inpaint')

        save_dict_list += save_dict_list_inpaint

    else:
        print("The species is not in the chemical formula, will not run inpainting with framework. As-generated structures will be used.")    
    
    # save_dict_list = get_save_dict_list(csp, s_list_inpaint_conventional_unit)
    save_dict_list = csp.e_hull_calculator.get_e_hull(save_dict_list)
    
    #####  Post process & compute phase stability & save structural information  #####
    print("--"*5 + "Post process and save structural information" + "--"*5)
    df = pd.DataFrame(columns=['material_id', 'formula', 'e_hull',
                              's_asGen_cif', 'spacegroup_asGen', 
                              's_relax_refine_cif', 'spacegroup_refine', 's_relax_cif', 
                              'Fmax_chgnet', 'E0_chgnet_atom', 'E_chgnet_atom', 'type', 'energy', 'structure'])

    for save_dict in save_dict_list:
        row_df = pd.DataFrame([save_dict])
        df = pd.concat([df, row_df], ignore_index=True)
    
    # df = df.sort_values('e_hull')
    # df['e_hull_diff'] = df.groupby('spacegroup_asGen')['e_hull'].diff()
    # df = df[(df['e_hull_diff'].isna()) | (df['e_hull_diff'].abs() >= 0.002)]

    df['e_hull_chgnet'] = df['e_hull']

    df = df.drop('e_hull', axis=1)
    # df = df.drop('e_hull_diff', axis=1)
    df = df.drop('energy', axis=1)
    df = df.drop('structure', axis=1)

    return df, total_denovo, total_inpaint



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--chemical_formula", type=str, default= "LiF", help="chemical formula")
    parser.add_argument("-v", "--atomic_volume", type=float, default= 24, help="atomic volume")
    parser.add_argument("-s", "--species", type=str, default= "Li", help="species")
    parser.add_argument("-n", "--num_iterations", type=int, default= 1, help="number of iterations to run the generation function")
    parser.add_argument("-m", "--num_inpaintings", type=int, default= 3, help="number of iterations to rerun the inpainting generation with the same framework")
    parser.add_argument("-d", "--device", type=str, default= "cuda", help="gpu device")
    args = parser.parse_args()

    comp_list = ["ZnSP2S5", "Zn2S2P2S5"]
    # for n_Li in range(13,25):
    #     for n_Si in range(1,25):
    #         if (n_Li + n_Si <= 25) and (n_Li / n_Si >=1) and (n_Li / n_Si <= 5):
    #             comp_str = f"Li{n_Li}Si{n_Si}"
    #             comp_list.append(comp_str)
    #             # print(comp_str)

    print(comp_list)
    
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
            # num_gen = 3, # number of structures generated from the cubic lattice (not used)
            # num_mutation = 2, # number of mutations during the relax-generation iteration (not used)
            num_cell = 1, # number of times to the formula
            # ehull_cutoff = 0.06, # e_hull cutoff (not used)
            )

    ### Define the CSP file and patched_phase diagram ###
    csp = CSP_Generator(chggen_path = "./files/cut_7_conv_3_epoch=27-val_loss=0.87.ckpt",
                        device= args.device)
    csp.load_e_hull_calculator(ppd_path= "/home/zhongpc/backup_chggen/host_gen/file_trans/2023-02-07-ppd-mp.pkl.gz")

    CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")

    ### iterate the composition list ###
    for comp_str in comp_list:
        opt_volume = args.atomic_volume
        
        for num_run in range(args.num_iterations):
            print("**"*5 + "Start generating structures with inpainting" + "**"*5)
            df_, total_denovo, total_inpaint = main(csp= csp, chemical_formula= comp_str, atomic_volume = opt_volume, 
                            species = args.species, # set the same to enable inpainting, set different to use as-generated structures
                            gen_kwargs = gen_kwargs, ld_kwargs = ld_kwargs)
                    
            # ROOT_DIR = './files/Li-Si/' + Composition(comp_str).reduced_formula

            chemical_formula = Composition(comp_str).formula
            chemical_formula = chemical_formula.replace(" ", "")
            ROOT_DIR = './files/Zn-P-S/' + chemical_formula

            try:
                structure = Structure.from_str(df_.iloc[0]['s_relax_refine_cif'], fmt='cif')
                opt_volume = structure.volume / structure.num_sites
            except:
                opt_volume = args.atomic_volume

            if os.path.exists(ROOT_DIR + '/generate_summary'+ '_' + CURRENT_DATE +'.csv'):
                df_.to_csv(ROOT_DIR + '/generate_summary' + '_' + CURRENT_DATE +'.csv', mode='a', index=False, header=False)
            else:
                mkdir(ROOT_DIR)
                df_.to_csv(ROOT_DIR + '/generate_summary' + '_' + CURRENT_DATE +'.csv', index=False, header=True)

    
            success_denovo = len(df_[df_['type'] == 'denovo'])
            success_inpaint = len(df_[df_['type'] == 'inpaint'])
        
            
            with open(ROOT_DIR + '/successful_rate.txt', 'a') as f:
                f.write(f"Composition: {comp_str}\n")
                f.write(f"Successful Rate of denovo generation: {success_denovo}/{total_denovo}\n")
                f.write(f"Successful Rate of inpaint generation: {success_inpaint}/{total_inpaint}\n")      
        
    # Example: python gen_ZnPS.py -d cuda:3 -s Zn -v 24 -n 3 -m 3 
    # n for rerun; m for repeat in the inpainting
