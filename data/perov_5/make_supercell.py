import torch
import numpy as np
import pandas as pd

from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

def generate_cif(row):
    """Generate cif file from row of dataframe."""
    crystal_str = row['cif']
    structure = Structure.from_str(crystal_str, fmt='cif')
    structure.make_supercell([2,2,2])
    print(structure)
    cif_writer = CifWriter(structure)
    cif_content = str(cif_writer)
    return cif_content

def generate_comp(row):
    crystal_str = row['cif']
    structure = Structure.from_str(crystal_str, fmt='cif')
    structure.make_supercell([2,2,2])
    print(structure)
    cif_writer = CifWriter(structure)
    cif_content = str(cif_writer)
    return cif_content


# test_zpc.
df = pd.read_csv('./test_zpc.csv')
df['cif'] = df.apply(generate_cif, axis=1)

output_csv_file = './cs2_test_zpc.csv'
df.to_csv(output_csv_file, index=False)

print(f"CSV with updated CIF saved to {output_csv_file}")
