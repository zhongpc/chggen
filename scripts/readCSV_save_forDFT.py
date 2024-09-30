import numpy as np
from pymatgen.core import Structure, Composition, Element, Lattice
from pymatgen.io.cif import CifWriter
from chggen.common.data_utils import mkdir

import argparse
import json
import pandas as pd
import os
import time
from datetime import datetime


def main(file_path, save_root_string = './forVASP_'):

    df_all = pd.read_csv(file_path)

    df_all = df_all.sort_values(['formula', 'e_hull_chgnet'])
    df_all['min_e_hull'] = df_all.groupby('formula')['e_hull_chgnet'].transform('min')
    df_all['filtered'] = (df_all['e_hull_chgnet'] <= 0.03) | (df_all['e_hull_chgnet'] == df_all['min_e_hull'])
    df_all = df_all[df_all['filtered']].drop(['min_e_hull', 'filtered'], axis=1)


    df_all = df_all.sort_values('e_hull_chgnet')
    df_all['e_hull_diff'] = df_all.groupby(['formula', 'spacegroup_refine'])['e_hull_chgnet'].diff()
    # df_all
    df_all = df_all[(df_all['e_hull_diff'].isna()) | (df_all['e_hull_diff'].abs() >= 0.01)]

    df_all = df_all.drop('e_hull_diff', axis=1)

    print(len(df_all))
    save_root = save_root_string + datetime.now().strftime('%Y-%m-%d')

    for idx, row in df_all.iterrows():
        structure = Structure.from_str(row['s_relax_refine_cif'], fmt='cif')
        save_json = {'material_id': row['material_id'],
                    'e_hull_chgnet': row['e_hull_chgnet'],
                    'structure': structure.as_dict(),}

        mkdir(save_root + '/INPUT_' + row['material_id'])


        with open(save_root + '/INPUT_' + row['material_id'] + '/s.json', 'w') as fp:
            json.dump(save_json, fp, indent=4)


    df_all.to_csv(save_root + '/generate_summary.csv', index=False)


    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, default= "./gen_summary.csv", help="The csv file to be read")
    parser.add_argument("-o", "--output", type=str, default= "./forVASP_", help="The csv file to be read")

    args = parser.parse_args()

    main(file_path = args.input, save_root_string=args.output)



