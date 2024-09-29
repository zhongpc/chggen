from chggen.common.sample_utils import CSP_Generator
from chggen.common.data_utils import mkdir
from chggen.common.sample_utils import get_inpaint_data_fromHost
from chggen.common.sample_utils import get_batch_inpaint_data_fromHost
from chggen.common.sample_utils import get_coarse_grain_framework

from types import SimpleNamespace
import numpy as np

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pandas as pd

import time
from datetime import datetime

csp = CSP_Generator(chggen_path = "./files/cut_7_conv_3_epoch=27-val_loss=0.87.ckpt",
                    device='cuda:6')


ld_kwargs = SimpleNamespace(
        n_step_each = 5,            # Corrector
        min_sigma = 0.01,
        num_noise_level = 200,
        signal_to_noise_ratio = 0.4,
        save_traj = False,
        disable_bar = False,
    )

gen_kwargs = SimpleNamespace(
        num_gen = 3, # number of structures generated from the cubic lattice
        num_mutation = 2, # number of mutations during the relax-generation iteration
        num_cell = 1, # number of times to the formula
        ehull_cutoff = 0.06,
        )


chemical_formula = 'Li3PS4'
atomic_volume = 20
#  Generate seven different bravis lattices via diffusion
s_list_Bravis = csp.generate_structures_from_Bravis(comp_str= chemical_formula, atom_volume= atomic_volume,
                                                    gen_kwargs=gen_kwargs, ld_kwargs=ld_kwargs)

for s in s_list_Bravis:
    try:

        analyzer_init = SpacegroupAnalyzer(structure= s, symprec= 0.15, angle_tolerance= 15)
        analyzer_init.get_space_group_symbol()
    except:
        print(s)
        continue
