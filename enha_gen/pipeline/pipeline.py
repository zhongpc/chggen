from types import SimpleNamespace
import sys
sys.path.append('../')

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from ase.io import read, write
from ase import Atoms
from ase.build import cut
from ase.visualize.plot import plot_atoms

from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure, Lattice

from chggen.pl_modules.model import CHGGen
from chggen.common.data_utils import (
    mkdir, 
    get_pymatgen_structure,
)

from auxi_func import (
    charge_balance_from_structure,
    structure_add_atoms,
    structure_to_tensor,
    check_in_box,
    remove_close_atoms,
)

mkdir('../data/host/')
device = 'cuda:2'



# Parameter Setting
uncertain_cut = 0.07            # Thredhold for unertainty
min_dist_cut = 12               # Minimum distance for cutting
inpaint_cut = 6.0               # Cut threshold for inpainting
subcell_size = [12,12,12]       # Cut subcells

# Load pre-trained model.
chggen = CHGGen.load_from_checkpoint('../data/model/cutoff-7_epoch=49-val_loss=0.83.ckpt', strict=False, map_location=device)

# ------------------------------------------------------------
# Subcell Cutting
# ------------------------------------------------------------

# Load atoms and get force uncertain and position
atoms = read('../data/structure_1.xyz')
F_uncertain = atoms.get_array('F_uncertain')

# Convert ase atoms to pymatgen structure
adaptor = AseAtomsAdaptor()
s_super = adaptor.get_structure(atoms)

# Compute id of atoms with high uncertainty
cut_atom_ids = np.where(F_uncertain > uncertain_cut)[0]
print(f"{len(cut_atom_ids)} subcells are selected as high uncertainty (> {uncertain_cut}).")

# Remove close atoms
cut_atom_ids = remove_close_atoms(s_super, cut_atom_ids, verbose=True)

s_subs = []
for center_id in tqdm(cut_atom_ids):
    subcell_atoms = []
    dist_vecs = []
    for i, site in enumerate(s_super):
        dist_vec = check_in_box(s_super, center_id, i, subcell_size)
        if dist_vec is not None:
            subcell_atoms.append(site)
            dist_vecs.append(dist_vec)
    
    # Calculate fractional coordinates
    frac_coords = [dist_vec / subcell_size + 0.5 for dist_vec in dist_vecs]
    
    # Create a new Structure object with the new lattice and the selected atoms
    lattice = Lattice.from_parameters(
        a=subcell_size[0], 
        b=subcell_size[1], 
        c=subcell_size[2], 
        alpha=90, beta=90, gamma=90,
    )
    s_sub = Structure(
        lattice=lattice, 
        species=[site.species for site in subcell_atoms], 
        coords=frac_coords,
    )
    
    s_subs.append(s_sub)

print(f"{len(s_subs)} subcells are generated.")
print(f"Number of atoms in each subcell: {[len(s_sub) for s_sub in s_subs]}")

# Save
for i, subcell in enumerate(s_subs):
    subcell.to(filename=f'../data/host/structure_ori_{i}.cif')

for id, s_sub in enumerate(s_subs):
    # Screen
    print("*"*100, f"\nSubcell {id+1}/{len(s_subs)}\n")
    
    # ------------------------------------------------------------
    # Charge Balancing
    # ------------------------------------------------------------

    # Compute the oxidation state
    oxi_state_super = s_super.composition.oxi_state_guesses()[0]
    print(f"Oxidation state {oxi_state_super}")

    # Compute the charge balance
    atoms_to_add = charge_balance_from_structure(
        structure = s_sub,
        oxi_state = oxi_state_super,
        verbose = False,
    )

    print(f"Number of atoms to be added: {atoms_to_add}")

    # ------------------------------------------------------------
    # Manipulate Subcell
    # ------------------------------------------------------------

    s_temp = s_sub.copy()
    print(len(s_temp))

    # Add atoms to the subcell
    s_temp = structure_add_atoms(
        structure = s_temp,
        atoms_to_add = atoms_to_add,
        cut_threshold = inpaint_cut,
        verbose=True,
    )
    print(len(s_temp))

    # ------------------------------------------------------------
    # Convert Subcell
    # ------------------------------------------------------------

    (   
        lengths, 
        angles, 
        frac_coords, 
        atom_types, 
        num_atoms, 
        atom_masks
    ) = structure_to_tensor(
        structure = s_temp,
        cut_threshold = inpaint_cut,
    )

    # Map to device
    lengths, angles, frac_coords, atom_types, num_atoms, atom_masks = map(
        lambda x: x.to(device),
        (lengths, angles, frac_coords, atom_types, num_atoms, atom_masks),
    )

    # ------------------------------------------------------------
    # Inpaint
    # ------------------------------------------------------------

    # Print how many atoms are masked and no masked
    print(f"Number of masked atoms to diffusion: {sum(atom_masks)}/{len(atom_masks)}")

    # Sampling.
    ld_kwargs = SimpleNamespace(
        n_step_each = 5,
        num_noise_level = 200,
        signal_to_noise_ratio = 0.4,
        save_traj = False,
        disable_bar = False,                     
    )

    results = chggen.conditional_langevin_dynamics(
        lengths = lengths,
        angles = angles,
        composition = atom_types,
        num_atoms = num_atoms,
        ori_frac_coords = frac_coords,
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

    for i, structure in enumerate(s_list):
        structure.to(filename=f'../data/host/structure_{id}_{i}.cif')