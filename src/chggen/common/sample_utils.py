from __future__ import annotations

import torch
import numpy as np
import os

from chggen.pl_modules.model import CHGGen
from chggen.common.data_utils import get_scaler, mkdir, get_pymatgen_structure
from chggen.common.e_hull_calculator import EHullCalculator

from chgnet.model.model import CHGNet
from chgnet.model.dynamics import StructOptimizer

from pymatgen.core import Structure, Composition, Element, Lattice
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from pymatgen.analysis.ewald import EwaldSummation


from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pandas as pd

import time
from datetime import datetime

import numpy as np


def filter_nan_structure(s_list):
    new_s_list = []
    for structure in s_list:
        if 'nan' in str(structure):
            pass
        else:
            new_s_list.append(structure)
    return new_s_list



def compute_ewald_energy_single_structure(
        structure: Structure,
    ) -> float:
        """Compute the Ewald energy of a structure with guessed oxidation state."""
        structure.add_oxidation_state_by_guess()
        ewald_sum = EwaldSummation(structure)
        ewald_energy = ewald_sum.total_energy
        structure.remove_oxidation_states()
        return ewald_energy


def generate_lattice_cell(lattice_type, volume, ratio = None):
    if ratio is None:
        random_ratio = np.random.uniform(1, 1.5)
    else:
        random_ratio = ratio

    if lattice_type == 'cubic':
        a = np.cbrt(volume)
        lattice_params = (a, a, a, 90, 90, 90)
    elif lattice_type == 'tetragonal':
        a = np.cbrt(volume / random_ratio)
        c = random_ratio * a
        lattice_params = (a, a, c, 90, 90, 90)
    elif lattice_type == 'orthorhombic':
        a = np.cbrt(volume / random_ratio)
        b = random_ratio * a
        c = a
        lattice_params = (a, b, c, 90, 90, 90)
    elif lattice_type == 'monoclinic':
        random_beta = np.random.uniform(60, 120)
        a = np.cbrt(volume / random_ratio / np.sin(random_beta*np.pi/180) )
        b = random_ratio * a
        c = a
        beta = random_beta
        lattice_params = (a, b, c, 90, beta, 90)

    elif lattice_type == 'aP':
        
        random_alpha = np.random.uniform(75, 105)
        random_beta = np.random.uniform(75, 105)
        random_gamma = np.random.uniform(75, 105)

        scale = np.sqrt(1 - np.cos(random_alpha*np.pi/180)**2  - np.cos(random_beta*np.pi/180)**2 - np.cos(random_gamma*np.pi/180)**2 + 2*np.cos(random_alpha*np.pi/180)*np.cos(random_beta*np.pi/180)*np.cos(random_gamma*np.pi/180))
        a = np.cbrt(volume / scale) # scale = 0.963
        b = a
        c = a

        lattice_params = (a, b, c, random_alpha, random_beta, random_gamma)

    elif lattice_type == 'hP':
        a = np.sqrt(volume / (np.sqrt(3) / 2) / random_ratio)
        c = a * random_ratio
        lattice_params = (a, a, c, 90, 90, 120)

    elif lattice_type == 'hR':
        a = np.cbrt(volume / 0.707)
        alpha = 60
        lattice_params = (a, a, a, alpha, alpha, alpha)
        
    else:
        raise ValueError("Invalid lattice type")
    
    return lattice_params


def get_coarse_grain_framework(structure: Structure,
                                species_to_remove = None,
                                sym_list = [0.1, 0.2, 0.5, 0.8, 1.0, 1.5, 2.0],
                                angle_list = [10, 15, 15, 15, 20, 30, 30],
                                ):
    """ 
    Coarse grain the framework of structure by removing the species in the species list.
    """

    if species_to_remove is None:
        raise ValueError("Please specify the species to be removed.")

    structure.remove_oxidation_states()

    for symprec, angle_tolerance in zip(sym_list, angle_list):
        s_frame = structure.copy()
        num_species = s_frame.composition[species_to_remove]
        

        s_frame.remove_species([species_to_remove])

        try:
            analyzer_CG = SpacegroupAnalyzer(structure= s_frame, 
                                         symprec= symprec, 
                                         angle_tolerance= angle_tolerance)
            s_CG = analyzer_CG.get_conventional_standard_structure()
            symbol_CG= analyzer_CG.get_space_group_symbol()
        except:
            s_CG = s_frame
            symbol_CG = 'P1'

        # print("symmetry: ", symprec, "angle_tolerance: ", angle_tolerance)
        # print("CG spacegroup: ", symbol_CG) 
        if symbol_CG== 'P1' or symbol_CG== 'P-1':
            continue
        else:
            break

    print("CG spacegroup: ", symbol_CG) 

    return s_CG, symbol_CG, int(num_species)



def run_SDE_from_structures(model, 
                            s_init_list,
                            ld_kwargs):
    """
    Run the reverse SDE (Stochastic Differential Equation) based on a given list of initial structures.

    Args:
        model (object): The CHGGen model used for the simulation.
        s_init_list (list): A list of initial structures.
        ld_kwargs (dict): Additional keyword arguments for the SDE simulation.

    Returns:
        tuple: A tuple containing the list of transformed structures and the simulation results.
    """
    
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



def run_SDE_simpleCubic(model, 
                        comp_str, # the string of composition
                        atom_volume, # avg atom volume
                        gen_kwargs,
                        ld_kwargs):
    """
    Generate SDE from give lattice parameters. 

    Args:
        model (object): The model used for generating trajectories.
        comp_str (str): The string representation of the composition.
        atom_volume (float): The average atom volume.
        gen_kwargs (object): Additional keyword arguments for generation task
        ld_kwargs (object): Additional keyword arguments for reverse SDE simulation.

    Returns:
        tuple: A tuple containing the generated trajectories and additional results.
    """
    
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
    # rand_frac_coords = torch.rand((num_atoms.sum(), 3), device= DEVICE, requires_grad = False)
    

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
                       ld_kwargs, 
                       ratio = None):
    """
    Run the SDE with experiencing different Bravis lattices

    Args:
        model (object): The CHGGen model used for the simulation.
        comp_str (str): The string representation of the composition.
        atom_volume (float): The average atom volume.
        gen_kwargs (object): The generation keyword arguments.
        ld_kwargs (object): The SDE simulation keyword arguments.
        ratio (float): The ratio of the lattice parameters (e.g c/a) None for random value between 1 and 1.5

    Returns:
        tuple: A tuple containing the pymatgen structure list and the results dictionary.
    """
    
    num_atoms = []
    all_atom_types = []
    
    DEVICE = model.device

    lengths = []
    angles = []

    # generate the lattice cell

    comp = Composition(comp_str)
    num_atoms_formula = int(comp.num_atoms)

    if isinstance(ratio, float):
        num_atoms = [num_atoms_formula] * 7 # for 7 different lattice types

    elif isinstance(ratio, list):
        num_atoms = [num_atoms_formula] * 7 * len(ratio) # for 7 different lattice types
        print("num_atoms", num_atoms)

    


    for lattice_type in ['cubic', 'tetragonal', 'orthorhombic', 'monoclinic', 'aP', 'hP', 'hR']:
        if isinstance(ratio, float):
            lattice_params = generate_lattice_cell(lattice_type, num_atoms_formula * atom_volume, ratio = ratio)
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
        
        elif isinstance(ratio, list):
            for r in ratio:
                lattice_params = generate_lattice_cell(lattice_type, num_atoms_formula * atom_volume, ratio = r)
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
    cur_atom_types = torch.tensor(all_atom_types, device = DEVICE)

    # print("num_atoms", num_atoms)
    # print(lengths)
    # print(angles)
    # print("cur_atom_types", cur_atom_types)

    lengths = torch.tensor(lengths, device = DEVICE, dtype = torch.float32).view(-1, 3)
    angles = torch.tensor(angles, device = DEVICE, dtype = torch.float32).view(-1, 3)
    
    # print(lengths)
    # print(angles)

    # rand_frac_coords = torch.rand((num_atoms.sum(), 3), device= DEVICE, requires_grad = False)
    

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



def get_inpaint_data_fromHost(model,
                              host_structure, 
                              num_insert_ion,
                              species = None):
    

    if species is None:
        raise ValueError("Please specify the ion (as a string) to be inserted.")
    
    
    DEVICE = model.device

    insert_ion = Element(species)

    cur_atom_types = []
    atom_masks = []
    for site in host_structure.sites:
        cur_atom_types.append(site.specie.Z)
        atom_masks.append(0)

    # Add atom types and atom masks for inserted ions.
    cur_atom_types += [insert_ion.Z] * num_insert_ion
    cur_atom_types = torch.tensor(cur_atom_types, device = DEVICE, dtype = torch.int32)
    atom_masks += [1] * num_insert_ion
    atom_masks = torch.tensor(atom_masks, device = DEVICE).bool()

    # Add random coordinates for host and inserted ions.
    insert_frac_coords = torch.rand(num_insert_ion, 3, requires_grad = False, device = DEVICE, dtype = torch.float32)
    cur_frac_coords = torch.tensor(host_structure.frac_coords, device = DEVICE)
    cur_frac_coords = torch.concat((cur_frac_coords, insert_frac_coords), axis = 0)
    cur_frac_coords = torch.tensor(cur_frac_coords, dtype = torch.float32)

    # Cell information.
    num_atoms = torch.tensor([len(cur_atom_types)], device = DEVICE, dtype=torch.int32)
    angles = torch.tensor([host_structure.lattice.angles], device = DEVICE)
    lengths = torch.tensor([host_structure.lattice.lengths], device = DEVICE)

    return cur_atom_types, atom_masks, cur_frac_coords, num_atoms, angles, lengths

def get_batch_inpaint_data_fromHost(model,
                              host_structure_list, 
                              num_intercalant_list,
                              species = None):
    """
    Get batch inpaint data from a list of host structures
    """

    if species is None:
        raise ValueError("Please specify the ion (as a string) to be inserted.")
    
    cur_atom_types, atom_masks, cur_frac_coords, num_atoms, angles, lengths = \
        zip(*[get_inpaint_data_fromHost(model, host_structure, num_intercalant, species= species) for host_structure, num_intercalant in zip(host_structure_list, num_intercalant_list)])

    # Construct batch data
    cur_atom_types = torch.cat(cur_atom_types, axis=0)
    atom_masks = torch.cat(atom_masks, axis=0)
    cur_frac_coords = torch.cat(cur_frac_coords, axis=0)
    num_atoms = torch.cat(num_atoms, axis=0)
    angles = torch.cat(angles, axis=0)
    lengths = torch.cat(lengths, axis=0)

    return cur_atom_types, atom_masks, cur_frac_coords, num_atoms, angles, lengths



def run_inpaint_SDE(model, 
                    host_structure_list,
                    num_intercalant_list,
                    ld_kwargs,
                    species = 'Li',
                    ):
    

    if host_structure_list == []:
        return [], None

    gen_inputs_batch = get_batch_inpaint_data_fromHost(model = model, #  
                                       host_structure_list= host_structure_list,
                                       num_intercalant_list = num_intercalant_list,
                                       species = species,
                                    )
    


    cur_atom_types, atom_masks, cur_frac_coords, num_atoms, angles, lengths = gen_inputs_batch


    results = model.conditional_reverse_SDE(
                                                lengths=lengths,
                                                angles=angles,
                                                composition=cur_atom_types,
                                                num_atoms=num_atoms,
                                                ori_frac_coords = cur_frac_coords,
                                                mask = atom_masks,
                                                ld_kwargs = ld_kwargs,
                                            )

    # repeats = len(results['all_frac_coords'])//results['num_atoms'][0]
    repeats = 1
    lengths = results['lengths'].repeat(repeats,1)
    angles = results['angles'].repeat(repeats,1)
    num_atoms = results['num_atoms'].repeat(repeats)
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




class CSP_Generator():

    def __init__(self, 
                 chggen_path,
                 chgnet_path = None,
                 device='cuda:0',
                 ):
        
        self.chggen = None
        self.chgnet = None
        self.relaxer = None
        self.e_hull_calculator = None
        self.device = torch.device(device)

        self.load_chggen(chggen_path)
        self.load_chgnet(chgnet_path)
        self.relaxer = StructOptimizer(model=self.chgnet,
                                        use_device= self.device)


    def load_chggen(self, chggen_path):
        self.chggen = CHGGen.load_from_checkpoint(checkpoint_path=chggen_path, map_location=self.device, strict=False)

    def load_e_hull_calculator(self, ppd_path):
        self.e_hull_calculator = EHullCalculator(ppd_path)

    def load_chgnet(self, chgnet_path = None):
        if chgnet_path is None:
            self.chgnet = CHGNet.load()
        else:
            self.chgnet = CHGNet.from_file(chgnet_path)

    def generate_simple_cubic_structure(self,
                                        comp_str, # the string of composition,
                                        atom_volume, # avg atom volume,
                                        gen_kwargs, # generation keyword arguments,
                                        ld_kwargs, # SDE simulation keyword arguments,
                                        ):
        
        s_list, results = run_SDE_simpleCubic(model= self.chggen,
                                              comp_str= comp_str,
                                              atom_volume = atom_volume,
                                              gen_kwargs = gen_kwargs,
                                              ld_kwargs= ld_kwargs)
        
        return s_list


    def generate_structures_from_Bravis(self, 
                                        comp_str, # the string of composition,
                                        atom_volume, # avg atom volume,
                                        gen_kwargs, # generation keyword arguments,
                                        ld_kwargs, # SDE simulation keyword arguments,
                                        ratio_list = [0.75, 0.875, 1.125, 1.25],  # ratio of lattice parameters # list or float
                                        ):
        """
        Returen: list of pymatgen structures of seven different Bravis lattices
        """
        s_Bravis_list, results = run_SDE_fromBravis(model= self.chggen, 
                                                comp_str= comp_str,
                                                atom_volume = atom_volume,
                                                gen_kwargs = gen_kwargs,
                                                ld_kwargs= ld_kwargs,
                                                ratio =  ratio_list)
        
        return s_Bravis_list
    

    def generate_from_host_structure(self,
                                    host_structure_list,
                                    num_intercalant_list,
                                    ld_kwargs,
                                    species='Li'):
        """
        Generate a list of structures by inserting ions into a host structure.

        Args:
            host_structure (Structure): The host structure into which ions will be inserted.
            num_insert_ion (int): The number of ions to be inserted.
            ld_kwargs (dict): Additional keyword arguments for the ion insertion algorithm.
            species (str): The species of the ions to be inserted. Default is 'Li'.

        Returns:
            list: A list of structures with ions inserted.

        """
        s_list, results = run_inpaint_SDE(model=self.chggen,
                                        host_structure_list= host_structure_list,
                                        num_intercalant_list= num_intercalant_list,
                                        ld_kwargs=ld_kwargs,
                                        species=species)
        return s_list
                           


def generate_crystal(fv_dict, # dictionary of formula and volume
                     CHGGEN_PATH, # path to the chggen model
                     CUDA_DEVICE = 'cuda:0',
                     ):

    comp_str = fv_dict['formula']
    atom_volume = fv_dict['volume']


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
