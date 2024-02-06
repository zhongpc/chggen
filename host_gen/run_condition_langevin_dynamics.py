from __future__ import annotations
from types import SimpleNamespace

import torch

from chggen.pl_modules.model import CHGGen
from chggen.common.data_utils import mkdir, get_pymatgen_structure

from pymatgen.core import Structure, Element

from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation

mkdir('./data/host/')
device = 'cuda:6'
# Load pre-trained model.
chggen = CHGGen.load_from_checkpoint('../gen_org/data/test_models/mp/epoch=9.ckpt', strict=False, map_location=device)

# Load structure.
s0 = Structure.from_file('../gen_org/LaCl3.cif')*[2,2,2]

s0.replace_species({'La3+': {'Nb5+':0.5, 'La3+': 0.5}})
order = OrderDisorderedStructureTransformation(algo = 2)
s0 = order.apply_transformation(s0, return_ranked_list = False)
s0.to(filename='./data/host/structure_ori.cif')

# Construct framework structure.
num_insert_ion = 8
insert_ion = Element('Li')
ori_frac_coords = torch.tensor(s0.frac_coords)

cur_atom_types = []
atom_masks = []
for site in s0.sites:
    cur_atom_types.append(site.specie.Z)
    atom_masks.append(0)

cur_atom_types += [insert_ion.Z] * num_insert_ion
cur_atom_types = torch.tensor(cur_atom_types)

atom_masks += [1] * num_insert_ion
atom_masks = torch.tensor(atom_masks)

insert_frac_coords = torch.rand(num_insert_ion, 3, requires_grad = False)

ori_frac_coords = torch.cat((ori_frac_coords, insert_frac_coords), axis = 0)
ori_frac_coords = torch.tensor(ori_frac_coords)

num_atoms = torch.tensor([len(cur_atom_types)])

angles = torch.tensor([s0.lattice.angles])
lengths = torch.tensor([s0.lattice.lengths])*1.0

# Quantization
lengths = lengths.float().to(device)
angles = angles.float().to(device)
num_atoms = num_atoms.int().to(device)
frac_coords = ori_frac_coords.float().to(device)
atom_masks = atom_masks.bool().to(device)
cur_atom_types = cur_atom_types.int().to(device)
ori_frac_coords = ori_frac_coords.float().to(device)

# Sampling.
ld_kwargs = SimpleNamespace(
    n_step_each = 1,
    min_sigma = 0,
    signal_to_noise_ratio = 0.4,
    save_traj = False,
    disable_bar = False,                     
)

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
    structure.to(filename='./data/host/structure_' + str(i) + '.cif')

print("Done")