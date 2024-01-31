"""Main file for running annealed Langevin dynamics for new material sampling."""

from __future__ import annotations
from types import SimpleNamespace

import torch

from chggen.pl_modules.model import CHGGen
from chggen.common.data_utils import mkdir, get_pymatgen_structure



mkdir('./data/de_novo/')

# Load model.
device = torch.device('cuda')
checkpoint_path = "../gen_org/data/test_models/mp/trainer_mp.ckpt"
chggen = CHGGen.load_from_checkpoint(checkpoint_path = checkpoint_path)
chggen.to(device = device)
print('Model loaded')

# Sampling.
ld_kwargs = SimpleNamespace(
    n_step_each = 0,
    min_sigma = 0,
    signal_to_noise_ratio = 0.4,
    save_traj = False,
    disable_bar = False,                     
)

# cur_frac_coords, cur_atom_types, num_atoms, lengths, angles
cur_frac_coords = torch.rand(50, 3, requires_grad = False, device = device)
cur_atom_types = torch.tensor([7, 7, 8, 22, 20, 7, 8, 8, 22, 20,7, 7, 8, 40, 20,7, 7, 8, 72, 20,7, 7, 8, 22, 20
                               ,7, 7, 8, 22, 20,7, 7, 8, 28, 30,7, 7, 8, 40, 56,7, 7, 8, 22, 20,8, 8, 8, 40, 20], device = device)
num_atoms = torch.tensor([5,5,5,5,5,5,5,5,5,5], device = device)

length = 4
lengths = torch.ones((10,3), device=device) * length
angles = torch.tensor([[90, 90, 90],
                       [90, 90, 90],
                       [90, 90, 90],
                       [90, 90, 90],
                       [90, 90, 90],
                       [90, 90, 90],
                       [90, 90, 90],
                       [90, 90, 90],
                       [90, 90, 90],
                       [90, 90, 90],], device = device)

# cur_frac_coords = torch.rand(5, 3, requires_grad = False, device = device)
# cur_atom_types = torch.tensor([8, 8, 8, 22, 20], device = device)
# num_atoms = torch.tensor([5], device = device)
# length = 4
# lengths = torch.ones((1,3), device=device) * length
# angles = torch.tensor([[90, 90, 90]], device = device)

results = chggen.langevin_dynamics(
    lengths=lengths,
    angles=angles,
    composition=cur_atom_types,
    num_atoms=num_atoms,
    ld_kwargs = ld_kwargs,
)

# repeats = len(results['all_frac_coords'])//results['num_atoms'][0]
repeats = 1

lengths = results['lengths'].repeat([repeats,1])
angles = results['angles'].repeat([repeats,1])
num_atoms = results['num_atoms'].repeat([repeats])
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
    structure.to(filename='./data/de_novo/structure_' + str(i) + '.cif')

print("Done")