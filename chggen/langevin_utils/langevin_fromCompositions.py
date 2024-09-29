from __future__ import annotations
from types import SimpleNamespace

import torch
import numpy as np

from chggen.pl_modules.model import CHGGen
from chggen.common.data_utils import mkdir, get_pymatgen_structure


from pymatgen.core import Structure, Composition, Element


def run_langevin_supercell(model, 
                           ld_kwargs, 
                           gen_kwargs):
    
    ### initilize the num_atoms and types ###

    num_atoms = []
    all_atom_types = []

    device = model.device

    for comp_str, pred_volume in zip(gen_kwargs.composition_list, gen_kwargs.pred_volume_list):
        comp = Composition(comp_str)
        num_atoms_formula = comp.num_atoms
        num_formula = int(np.round(gen_kwargs.lattice_supercell**3 / (pred_volume * num_atoms_formula), 0))

        num_atoms.append( int(num_atoms_formula * num_formula) )

        atom_types = []
        comp_dict = comp.as_dict()
        for key in comp_dict:
            amount = comp_dict[key]
            element = Element(key)
            atom_types += [element.Z] * int(amount)
        atom_types = atom_types * num_formula
        all_atom_types += atom_types


    num_atoms = torch.tensor(num_atoms, device = device)

    cur_atom_types = torch.tensor(all_atom_types, device = device)
    angles = torch.ones((len(num_atoms), 3), device = device) * 90
    lengths =  torch.ones((len(num_atoms), 3), device = device) * gen_kwargs.lattice_supercell


    rand_frac_coords = torch.rand((num_atoms.sum(), 3), device= device, requires_grad = False)
    

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

    s_list_rand = get_pymatgen_structure(
        lengths = lengths,         
        angles = angles,
        num_atoms = num_atoms,
        frac_coords = rand_frac_coords,
        atom_types = atom_types,
    )



    return s_list, s_list_rand





if __name__ == "__main__":

    device = torch.device('cuda')
    checkpoint_path = "./test_models/MP_20/last.ckpt"
    chggen = CHGGen.load_from_checkpoint(checkpoint_path = checkpoint_path, map_location= device)

    # Sampling.
    ld_kwargs = SimpleNamespace(
        n_step_each = 1,
        step_lr = 1e-4,
        min_sigma = 0,
        save_traj = False,
        disable_bar = False,                     
    )

    gen_kwargs = SimpleNamespace(
        lattice_supercell = 16,  # the cell size in A for the supercell
        composition_list = ['Li2ZrCl6', 'Li5ZrInCl12'], # composition list
        pred_volume_list = [23.66, 25], # predicted volume per atom
        save_path = './auto_generation/supercell_structures/', 
    )

    s_list, s_list_rand = run_langevin_supercell(model = chggen, 
                                                ld_kwargs = ld_kwargs, 
                                                gen_kwargs = gen_kwargs)
    
    mkdir(gen_kwargs.save_path)
    for structure_id in range(len(s_list)):
        s_save = s_list[structure_id]
        s_rand = s_list_rand[structure_id]

        formula = s_save.composition.reduced_formula
        s_save.to(filename= gen_kwargs.save_path + 'super_' + formula + '.cif')
        s_rand.to(filename= gen_kwargs.save_path + 'rand_' + formula + '.cif')


    print("Done")





