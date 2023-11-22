"""Modules for dataset and dataloader."""

from typing import List
from p_tqdm import p_umap

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from pymatgen.core import Structure

from chggen.common.data_utils import (
    add_reduced_lattice_prop, 
    add_reduced_lengths_and_angles_prop
)
from chgnet.graph.converter import CrystalGraphConverter



crys_converter = CrystalGraphConverter(
    atom_graph_cutoff = 6, bond_graph_cutoff = 3
)

def process_one(
    row: pd.DataFrame,
    niggli: bool,                               # TODO: Consider Niggli reduction.
    primitive: bool,                            # TODO: Consider primitive cell or conventional cell.
    prop_list: list,
) -> dict:
    """Process one row of the csv file."""
    crystal_str = row['cif']                    # cif file
    structure = Structure.from_str(crystal_str, fmt = 'cif')

    crys_graph = crys_converter(structure)

    # try:
    #     crys_graph = crys_converter(structure)
    # except:
    #     print("Crystal graph construction failed in CHGNet. Check the process_csv.")
    #     return None

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
        'crys_graph': crys_graph,
    }
    result_dict.update(properties)
    return result_dict

def process_csv(
    input_file: str, 
    num_workers: int, 
    niggli: bool,                               
    primitive: bool,                           
    prop_list: list,
) -> List[dict]:
    """Process csv file to get the list of dict containing infomation for one
    structure."""
    df = pd.read_csv(input_file)

    unordered_results = p_umap(
        process_one,
        [df.iloc[idx] for idx in range(len(df))],
        [niggli] * len(df),
        [primitive] * len(df),
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
        niggli: bool = True, 
        primitive: bool = False,
        preprocess_workers: int = 16,
        lattice_scale_method: str = 'scale_length',
        **kwargs
    ) -> None:
        """Initialize CHGNetDataset.
        
        Args:
            name (str): name of the dataset.
            path (str): path to the csv file.
            prop_list (list): list of properties to be included in the dataset.
            niggli (bool, optional): Whether to perform Niggli reduction. 
                Defaults to True.
            primitive (bool, optional): Whether to use primitive cell.
                Defaults to False.
            preprocess_workers (int, optional): Number of workers for preprocessing.
                Defaults to 16.
            lattice_scale_method (str, optional): Method for scaling the lattice.
                Defaults to 'scale_length'.
        """
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop_list = prop_list
        self.niggli = niggli
        self.primitive = primitive
        self.lattice_scale_method = lattice_scale_method
        self.cached_data = process_csv(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            prop_list= prop_list)
        
        # Niggli reduction on lattice, lengths and angles - scale lengths and angles by num of atoms.
        add_reduced_lattice_prop(self.cached_data, lattice_scale_method)
        add_reduced_lengths_and_angles_prop(self.cached_data, lattice_scale_method)
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

        crys_graph = data_dict['crys_graph']
        lattice = data_dict['lattice']                      # (num of structure, 9)
        reduced_lattice = data_dict['reduced_lattice']      # (num of structure, 9)
        frac_coords = data_dict['frac_coords']
        atom_types = data_dict['atom_types']
        lengths = data_dict['lengths']
        angles = data_dict['angles']
        num_atoms = data_dict['num_atoms']
        atom_volume = data_dict['atom_volume'] 

        edge_index = [(i, j) for i in range(len(atom_types)) for j in range(len(atom_types)) if i != j]

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            x=torch.LongTensor(atom_types),
            crys_graph=crys_graph,
            edge_index=torch.LongTensor(edge_index).T,       # edge index for fully connected graph
            lattices=torch.Tensor(lattice).view(1, -1),
            reduced_lattices=torch.Tensor(reduced_lattice).view(1, -1),
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
