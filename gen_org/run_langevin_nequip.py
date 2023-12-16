"""Main file for running annealed Langevin dynamics for new material sampling."""

from __future__ import annotations
from types import SimpleNamespace

import torch

from chggen.pl_data.dataset import CHGNetDataset
from chggen.pl_modules.model import CHGGen
from chggen.common.data_utils import get_scaler, mkdir, get_pymatgen_structure



mkdir('./test_models/structures/')

# Load dataset for obtaining lattice_scaler for scaling in Niggli reduction.
dataset = CHGNetDataset(
    path='./data/perov_5/test_zpc.csv',
    name = 'A_good_name',
    prop_list = ['heat_all'],
)
lattice_scaler = get_scaler(dataset= dataset)

# Load model.
device = torch.device('cuda')

checkpoint_path = "./test_models/perov/trainer_perov.ckpt"

chggen = CHGGen.load_from_checkpoint(checkpoint_path = checkpoint_path)
chggen.lattice_scaler = lattice_scaler
chggen.to(device = device)
print('Model loaded')

# Sampling.
ld_kwargs = SimpleNamespace(
    n_step_each = 1,
    step_lr = 1e-4,
    min_sigma = 0,
    save_traj = False,
    disable_bar = False,                     
)

#######################################################################################
# Supercell
#######################################################################################
# cur_frac_coords = torch.rand(80, 3, requires_grad = False, device = device)
# cur_atom_types1 = torch.tensor([7, 7, 8, 22, 20], device = device)
# cur_atom_types1 = cur_atom_types1.repeat(8)
# cur_atom_types2 = torch.tensor([8, 8, 8, 28, 56], device = device)
# cur_atom_types2 = cur_atom_types2.repeat(8)
# cur_atom_types = torch.cat((cur_atom_types1, cur_atom_types1))

# num_atoms = torch.tensor([40, 40], device = device)

# length = 4
# lengths = torch.ones((2,3), device=device) * length * 2
# angle = 90
# angles = torch.ones((2,3), device=device) * angle


#######################################################################################
# lattice parameter = 4
#######################################################################################
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

results = chggen.langevin_dynamics(
    lengths=lengths,
    angles=angles,
    composition=cur_atom_types,
    num_atoms=num_atoms,
    ld_kwargs = ld_kwargs,
)

# results = chggen.sample(
#     lattice_size=4,
#     num_samples=10,
#     ld_kwargs=ld_kwargs,
# )

lengths = results['lengths']
angles = results['angles']
num_atoms = results['num_atoms']
frac_coords = results['frac_coords']
atom_types = results['atom_types']

print(lengths)
print(angles)
print(num_atoms)
print(frac_coords)
print(atom_types)

# # Save final structures.
# for structure_id in range(len(num_atoms)):
#     s = chggen.get_pymatgen_structure(
#         lengths = lengths[structure_id].reshape(1,3),         
#         angles = angles[structure_id].reshape(1,3),
#         num_atoms = num_atoms[structure_id].reshape(1),
#         frac_coords = frac_coords[structure_id:structure_id+5],
#         atom_types = atom_types[structure_id:structure_id+5],
#     )
#     s.to(filename='./test_models/table_structures/structure_' + str(structure_id) + '.cif')

s_list = get_pymatgen_structure(
    lengths = lengths,         
    angles = angles,
    num_atoms = num_atoms,
    frac_coords = frac_coords,
    atom_types = atom_types,
)

for structure_id in range(len(num_atoms)):
    s_list[structure_id].to(filename='./test_models/structures/structure_' + str(structure_id) + '.cif')

print("Done")