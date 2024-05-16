"""Modules for dataset and dataloader."""

from typing import List
from p_tqdm import p_umap

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from pymatgen.core import Structure



def process_one(
    row: pd.DataFrame,
    prop_list: list,
) -> dict:
    """Process one row of the csv file."""
    crystal_str = row['cif']                    # cif file

    structure = Structure.from_str(
        crystal_str, 
        fmt = 'cif', 
        site_tolerance=0,
        frac_tolerance=0,
    )

    if structure.num_sites >= 1000:             # Prevent GPU memory overflow.
        print(f"Too many atoms in {row['material_id']}, skip this structure.")
        return None

    properties = {k: row[k] for k in prop_list if k in row.keys()}
    result_dict = {
        'mp_id': row['material_id'],
        'cif': crystal_str,
        'lattice': structure.lattice.matrix,
        'frac_coords': structure.frac_coords,
        'atom_types': structure.atomic_numbers,
        'lengths': np.array(structure.lattice.lengths),
        'angles': np.array(structure.lattice.angles),
        'num_atoms': structure.num_sites,
        'atom_volume': structure.volume / structure.num_sites,
    }
    result_dict.update(properties)
    return result_dict


def process_pmg_structure(
    structure: Structure,
    # prop_list: list,
) -> dict:
    """Process one pymatgen structure"""  

    mp_id = 0

    # properties = {k: row[k] for k in prop_list if k in row.keys()}
    data_dict = {
        'mp_id': mp_id,
        # 'cif': crystal_str,
        'lattice': structure.lattice.matrix,
        'frac_coords': structure.frac_coords,
        'atom_types': structure.atomic_numbers,
        'lengths': np.array(structure.lattice.lengths),
        'angles': np.array(structure.lattice.angles),
        'num_atoms': structure.num_sites,
        'atom_volume': structure.volume / structure.num_sites,
    }

    lattice = data_dict['lattice']                      # (num of structure, 9)
    frac_coords = data_dict['frac_coords']
    atom_types = data_dict['atom_types']
    lengths = data_dict['lengths']
    angles = data_dict['angles']
    num_atoms = data_dict['num_atoms']
    atom_volume = data_dict['atom_volume'] 

    # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
    data = Data(
        x=torch.LongTensor(atom_types),
        lattices=torch.Tensor(lattice).view(1, -1),
        frac_coords=torch.Tensor(frac_coords),
        atom_types=torch.LongTensor(atom_types),
        lengths = torch.Tensor(lengths).view(1, -1),
        angles = torch.Tensor(angles).view(1, -1),
        num_atoms = num_atoms,
        atom_volume = torch.Tensor([atom_volume]).view(1, -1),
        # properties = torch.Tensor(prop_list).view(1, -1),
    )
    
    # result_dict.update(properties)
    return data


def process_csv(
    input_file: str, 
    num_workers: int,                          
    prop_list: list,
) -> List[dict]:
    """Process csv file to get the list of dict containing infomation for one
    structure."""
    df = pd.read_csv(input_file, keep_default_na=False, na_values=[''])

    unordered_results = p_umap(
        process_one,
        [df.iloc[idx] for idx in range(len(df))],
        [prop_list] * len(df),
        num_cpus=num_workers)
    
    unordered_results = [item for item in unordered_results if item is not None]

    return unordered_results


class CHGNetDataset(Dataset):
    def __init__(
        self, 
        name: str, 
        path: str,
        prop_list: list, 
        preprocess_workers: int = 16,
        **kwargs
    ) -> None:
        """Initialize CHGNetDataset.
        
        Args:
            name (str): name of the dataset.
            path (str): path to the csv file.
            prop_list (list): list of properties to be included in the dataset.
            preprocess_workers (int, optional): Number of workers for preprocessing.
                Defaults to 16.
        """
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop_list = prop_list
        self.cached_data = process_csv(
            self.path,
            preprocess_workers,
            prop_list= prop_list)
        
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        prop_list = []
        for key in self.prop_list:
            prop = data_dict[key]
            prop_list.append(prop)

        lattice = data_dict['lattice']                      # (num of structure, 9)
        frac_coords = data_dict['frac_coords']
        atom_types = data_dict['atom_types']
        lengths = data_dict['lengths']
        angles = data_dict['angles']
        num_atoms = data_dict['num_atoms']
        atom_volume = data_dict['atom_volume'] 

        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            x=torch.LongTensor(atom_types),
            lattices=torch.Tensor(lattice).view(1, -1),
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths = torch.Tensor(lengths).view(1, -1),
            angles = torch.Tensor(angles).view(1, -1),
            num_atoms = num_atoms,
            atom_volume = torch.Tensor([atom_volume]).view(1, -1),
            properties = torch.Tensor(prop_list).view(1, -1),
        )
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"
