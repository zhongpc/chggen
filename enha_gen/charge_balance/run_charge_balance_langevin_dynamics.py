from types import SimpleNamespace
import sys
sys.path.append('../')

from pymatgen.core import Structure

from auxi_func import (
    charge_balance_from_structure,
    structure_add_atoms,
    structure_to_tensor,
)

from chggen.pl_modules.model import CHGGen
from chggen.common.data_utils import (
    mkdir, 
    get_pymatgen_structure,
    get_pbc_distances,
)

mkdir('../data/host/')
device = 'cuda:0'


# ------------------------------------------------------------
# Structure Information
# ------------------------------------------------------------
# Load supercell
s_super = Structure.from_file(
    filename = "../data/supercell.cif",
    site_tolerance = 0,
    frac_tolerance = 0,
)

# Load subcell
s_sub = Structure.from_file(
    filename = "../data/cut_subcell.cif",
    site_tolerance = 0,
    frac_tolerance = 0,
)

# ------------------------------------------------------------
# Charge Balance
# ------------------------------------------------------------
# Compute the oxidation state
oxi_state_super = s_super.composition.oxi_state_guesses()[0]
print(f"Oxidation state {oxi_state_super}")

# Compute the charge balance
atoms_to_add = charge_balance_from_structure(
    structure = s_sub,
    oxi_state = oxi_state_super,
    verbose = True,
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
    cut_threshold = 5.0,
)

# Map to device
lengths, angles, frac_coords, atom_types, num_atoms, atom_masks = map(
    lambda x: x.to(device),
    (lengths, angles, frac_coords, atom_types, num_atoms, atom_masks),
)

# ------------------------------------------------------------
# Inpaint
# ------------------------------------------------------------
# Load pre-trained model.
chggen = CHGGen.load_from_checkpoint('../../gen_org/data/test_models/mp/cutoff-7_epoch=49-val_loss=0.83.ckpt', strict=False, map_location=device)

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
    structure.to(filename='../data/host/structure_' + str(i) + '.cif')