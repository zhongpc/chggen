import torch
import numpy as np
import pandas as pd

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

"""
site_tolerance (float): This tolerance is used to determine if two sites are sitting in the same position,
    in which case they will be combined to a single disordered site. Defaults to 1e-4.
frac_tolerance (float): This tolerance is used to determine is a coordinate should be rounded to an ideal
    value. E.g., 0.6667 is rounded to 2/3. This is desired if symmetry operations are going to be applied.
    However, for very large CIF files, this may need to be set to 0.
"""

def generate_cif(row):
    """Generate cif file from row of dataframe."""
    crystal_str = row['cif']
    structure = Structure.from_str(
        crystal_str, 
        fmt='cif', 
        site_tolerance=0,       # CifParser parameter for not merge close atoms.
        frac_tolerance=0,        # CifParser parameter for not shift frac coords for geometric matcher.
    )
    
    a = structure.lattice.a
    b = structure.lattice.b
    c = structure.lattice.c

    max_lattice = 15 # 16 A
    
    supercell_matrix = [int(np.ceil(max_lattice / a)), 
                        int(np.ceil(max_lattice / b)), 
                        int(np.ceil(max_lattice / c))]
    
    structure.make_supercell(supercell_matrix)

    cif_writer = CifWriter(structure)
    cif_content = str(cif_writer)
    return cif_content

df = pd.read_csv('./train.csv')
df['cif'] = df.apply(generate_cif, axis=1)

output_csv_file = './SC-15_train.csv'
df.to_csv(output_csv_file, index=False)

print(f"CSV with updated CIF saved to {output_csv_file}")
