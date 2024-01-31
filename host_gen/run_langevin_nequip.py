"""Main file for running annealed Langevin dynamics for new material sampling."""

from __future__ import annotations
from types import SimpleNamespace

import torch

from chggen.pl_modules.model import CHGGen
from chggen.common.data_utils import mkdir, get_pymatgen_structure



mkdir('./data/test_models/universal/structures_test/')

# Load model.
device = torch.device('cuda')
checkpoint_path = "../gen_org/data/test_models/universal/epoch=29.ckpt"
chggen = CHGGen.load_from_checkpoint(checkpoint_path = checkpoint_path)
chggen.to(device = device)
print('Model loaded')

# Sampling.
ld_kwargs = SimpleNamespace(
    n_step_each = 5,
    min_sigma = 0,
    signal_to_noise_ratio = 0.4,
    save_traj = True,
    disable_bar = False,                     
)

# cur_frac_coords, cur_atom_types, num_atoms, lengths, angles
# cur_frac_coords = torch.rand(50, 3, requires_grad = False, device = device)
# cur_atom_types = torch.tensor([7, 7, 8, 22, 20, 7, 8, 8, 22, 20,7, 7, 8, 40, 20,7, 7, 8, 72, 20,7, 7, 8, 22, 20
#                                ,7, 7, 8, 22, 20,7, 7, 8, 28, 30,7, 7, 8, 40, 56,7, 7, 8, 22, 20,8, 8, 8, 40, 20], device = device)
# num_atoms = torch.tensor([5,5,5,5,5,5,5,5,5,5], device = device)

# length = 4
# lengths = torch.ones((10,3), device=device) * length
# angles = torch.tensor([[90, 90, 90],
#                        [90, 90, 90],
#                        [90, 90, 90],
#                        [90, 90, 90],
#                        [90, 90, 90],
#                        [90, 90, 90],
#                        [90, 90, 90],
#                        [90, 90, 90],
#                        [90, 90, 90],
#                        [90, 90, 90],], device = device)

cur_frac_coords = torch.rand(5, 3, requires_grad = False, device = device)
cur_atom_types = torch.tensor([8, 8, 8, 22, 20], device = device)
num_atoms = torch.tensor([5], device = device)
length = 4
lengths = torch.ones((1,3), device=device) * length
angles = torch.tensor([[90, 90, 90]], device = device)

results = chggen.langevin_dynamics(
    lengths=lengths,
    angles=angles,
    composition=cur_atom_types,
    num_atoms=num_atoms,
    ld_kwargs = ld_kwargs,
)

lengths = results['lengths'].repeat([199,1])
angles = results['angles'].repeat([199,1])
num_atoms = results['num_atoms'].repeat([199])
frac_coords = results['all_frac_coords']
atom_types = results['all_atom_types']

print(f"lengths: {lengths.shape}")
print(f"angles: {angles.shape}")
print(f"num_atoms: {num_atoms.shape}")
print(f"frac_coords: {frac_coords.shape}") 
print(f"atom_types: {atom_types.shape}")

s_list = get_pymatgen_structure(
    lengths = lengths,         
    angles = angles,
    num_atoms = num_atoms,
    frac_coords = frac_coords,
    atom_types = atom_types,
)

for i, structure in enumerate(s_list):
    structure.to(filename='./data/test_models/universal/structures_test/structure_' + str(i) + '.cif')

print("Done")