import torch
import numpy as np
import pandas as pd

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

def generate_cif(row):
    """Generate cif file from row of dataframe."""
    crystal_str = row['cif']
    structure = Structure.from_str(crystal_str, fmt='cif')
    
    a = structure.lattice.a
    b = structure.lattice.b
    c = structure.lattice.c

    max_lattice = 15 # 16 A
    
    supercell_matrix = [int(np.ceil(15 / a)), int(np.ceil(15 / b)), int(np.ceil(15 / c))]
    structure.make_supercell(supercell_matrix)

    cif_writer = CifWriter(structure)
    cif_content = str(cif_writer)
    return cif_content


df = pd.read_csv('./train.csv')
df['cif'] = df.apply(generate_cif, axis=1)

output_csv_file = './SC-15_train.csv'
df.to_csv(output_csv_file, index=False)

print(f"CSV with updated CIF saved to {output_csv_file}")
