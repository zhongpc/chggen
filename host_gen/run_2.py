from __future__ import annotations
from types import SimpleNamespace
import pandas as pd

import torch

from chggen.pl_modules.model import CHGGen
from chggen.common.data_utils import mkdir, get_pymatgen_structure

from pymatgen.core import Structure, Element



mkdir('./data/recon/')

# Load pre-trained model.
chggen = CHGGen.load_from_checkpoint('../gen_org/data/test_models/mp/trainer_mp.ckpt', strict=False)

# Load data.
mp_O_Li = pd.read_csv('../lithium/data/mp_O_Li.csv', keep_default_na=False, na_values=[''])

num = 20
s0 = Structure.from_str(mp_O_Li.iloc[num]['cif'], fmt='cif', frac_tolerance=0, site_tolerance=0)
s0.to(filename='./data/recon/structure_ori.cif')

num_Li = int(s0.composition.get_el_amt_dict()['Li'])
print(f"{num_Li} Li atoms in s0 with total {len(s0)} atoms")
s0.remove_species(['Li'])

# Print details
print(f'structure: {s0}')

# Construct framework structure.
num_insert_ion = num_Li
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
lengths = torch.tensor([s0.lattice.lengths])

# Quantization
lengths = lengths.float()
angles = angles.float()
num_atoms = num_atoms.int()
frac_coords = ori_frac_coords.float()
atom_masks = atom_masks.bool()
cur_atom_types = cur_atom_types.int()
ori_frac_coords = ori_frac_coords.float()

# Sampling.
ld_kwargs = SimpleNamespace(
    n_step_each = 3,            # Corrector
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
    structure.to(filename='./data/recon/structure_' + str(i) + '.cif')

print("Done")