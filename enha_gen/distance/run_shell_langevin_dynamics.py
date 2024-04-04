"""Run multiple Langevin dynamics with sphere mask to study density effect."""
from __future__ import annotations
from types import SimpleNamespace
import sys
sys.path.append('../')

import torch

from chggen.pl_modules.model import CHGGen
from chggen.common.data_utils import (
    mkdir, get_pymatgen_structure
)

from pymatgen.core import Structure

mkdir('../data/distance/')
device = 'cuda:0'

# -------------------------------------------------------------
cut_threshold = 5       # A
num_samples = 100
# -------------------------------------------------------------



# Load pre-trained model.
chggen = CHGGen.load_from_checkpoint('../../gen_org/data/test_models/mp/cutoff-7_epoch=49-val_loss=0.83.ckpt', strict=False, map_location=device)

# Load structure.
s0 = Structure.from_file(
    '../data/cut_subcell.cif',
    site_tolerance=0,
    frac_tolerance=0,
)
s0.to(filename='../data/distance/structure_ori.cif')

# Construct framework structure
angles = torch.tensor([s0.lattice.angles])
lengths = torch.tensor([s0.lattice.lengths])
ori_frac_coords = torch.tensor(s0.frac_coords)

# Compute the distance from current atom to the center of the cell
center = torch.tensor([0.5, 0.5, 0.5])
frac_dists = ori_frac_coords - center[None, :]
cart_dists = lengths[0][None, :] * frac_dists
cart_dist = torch.norm(cart_dists, dim = 1)

cur_atom_types = []
atom_masks = []
for i, site in enumerate(s0.sites):
    cur_atom_types.append(site.specie.Z)
    if cart_dist[i] < cut_threshold:
        atom_masks.append(0)
    else:
        atom_masks.append(1)

# Print how many atoms are masked and no masked
print(f"Number of masked atoms to diffusion: {sum(atom_masks)}/{len(atom_masks)}")

cur_atom_types = torch.tensor(cur_atom_types)
atom_masks = torch.tensor(atom_masks)
num_atoms = torch.tensor([len(cur_atom_types)])

# Quantization
lengths = lengths.float().to(device)
angles = angles.float().to(device)
num_atoms = num_atoms.int().to(device)
atom_masks = atom_masks.bool().to(device)
cur_atom_types = cur_atom_types.int().to(device)
ori_frac_coords = ori_frac_coords.float().to(device)

# Sampling.
ld_kwargs = SimpleNamespace(
    n_step_each = 5,
    num_noise_level = 200,
    signal_to_noise_ratio = 0.4,
    save_traj = False,
    disable_bar = False,                     
)

for num_sample in range(num_samples):
    print(f"Sample {num_sample} ...")
    results = chggen.conditional_langevin_dynamics(
        lengths=lengths,
        angles=angles,
        composition=cur_atom_types,
        num_atoms=num_atoms,
        ori_frac_coords = ori_frac_coords,
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
        structure.to(filename=f'../data/distance/structure_{num_sample}_{i}.cif')

    print("Done")