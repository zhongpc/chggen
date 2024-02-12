from __future__ import annotations
from types import SimpleNamespace
from time import time
import sys
sys.path.append('../')

import numpy as np
import pandas as pd
import torch

from pymatgen.io.cif import CifWriter

from chggen.pl_modules.model import CHGGen
from chggen.common.data_utils import mkdir, get_pymatgen_structure

from aux_funcs_inpaint import get_batch_data



device = 'cuda:6'

# Load pre-trained model.
chggen = CHGGen.load_from_checkpoint('../../gen_org/data/test_models/mp_pretrain_pte/mp_pretrain_trainer.ckpt', strict=False, map_location=device)

# Load csv data.
data_inpaint = pd.read_csv('../data/dataset/data_inpaint_Li.csv', keep_default_na=False, na_values=[''])
data_inpaint_new = data_inpaint.copy()
data_inpaint_new['cif_inpaint'] = ''

# Data loader parameter
batch_size = 32
num_total = len(data_inpaint)
ptr = np.arange(0, num_total, batch_size)
ptr = np.append(ptr, num_total) if ptr[-1] != num_total else ptr

# Sampling.
ld_kwargs = SimpleNamespace(
    sigma_begin = 10.0,
    sigma_end = 0.01,
    num_noise_level = 200,
    n_step_each = 5,            # Corrector
    min_sigma = 0,
    signal_to_noise_ratio = 0.4,
    save_traj = False,
    disable_bar = False,  
)

start_time = time()

for i in range(len(ptr) - 1):
    cur_atom_types, atom_masks, cur_frac_coords, num_atoms, angles, lengths = get_batch_data(data_inpaint, ptr[i], ptr[i+1], device)

    results = chggen.conditional_langevin_dynamics(
        lengths=lengths,
        angles=angles,
        composition=cur_atom_types,
        num_atoms=num_atoms,
        ori_frac_coords = cur_frac_coords,
        mask = atom_masks,
        ld_kwargs = ld_kwargs,
    )
    
    s_list = get_pymatgen_structure(
        lengths = results['lengths'],
        angles = results['angles'],
        num_atoms = results['num_atoms'],
        frac_coords = results['frac_coords'],
        atom_types = results['atom_types'],
    )
    
    # Convert into cif file with CifWriter
    cifs = [CifWriter(s).__str__() for s in s_list]
    
    # Save cif file to dataframe 
    data_inpaint_new['cif_inpaint'][ptr[i]:ptr[i+1]] = cifs

    # Save to CSV every 5 minutes
    if time() - start_time > 1 * 60:
        print('Saving to CSV...')
        data_inpaint_new.to_csv('../data/data_inpaint.csv', index=False)
        start_time = time()

# Save to CSV.
data_inpaint_new.to_csv('../data/data_inpaint.csv', index=False)
print("Done")