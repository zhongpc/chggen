"""Class for creating composition dataset."""

from __future__ import annotations

from typing import List
import functools
import os.path as osp

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data

from pymatgen.core import Composition

from compgen.utils import get_scaler
from .featurizer import Featurizer



class CompositionDataset(Dataset):
    """The CompositionDataset is a wrapper for a dataset data points are
    automatically constructed from composition strings.
    
    Args:
        data_path (str): The path to the csv dataset file.
        com_tag (str): The column name for the composition strings. 
            Defaults = 'formula'.
        ave_v_tag (str): The column name for the average voltage.
            Defaults = 'avg_volume'.
        identifiers (list): The column names for the identifiers.
    """
    def __init__(
        self,
        data_path: str | None,
        com_tag = 'formula',
        ave_v_tag = 'avg_volume',
        identifiers = ["material_id", "formula"],
        **kwargs,
    ) -> None:
        """Initialize a CompositionDataset with a given data path."""
        super(CompositionDataset, self).__init__()
        
        if data_path is None:
            return None
        
        assert len(identifiers) == 2, "Two identifiers are required"
        assert osp.exists(data_path), f"{data_path} does not exist!"
        self.com_tag = com_tag
        self.ave_v_tag = ave_v_tag
        self.identifiers = identifiers
        self.df = pd.read_csv(data_path, keep_default_na=False, na_values=[])
        
        # Read matscholar embedding.
        fea_name = "matscholar-embedding.json"
        fea_path = osp.join(osp.dirname(osp.abspath(__file__)), fea_name)
        self.elem_features = Featurizer.from_json(fea_path)
        self.elem_emb_len = self.elem_features.embedding_size
        
        # Calculate scaler for ave_v.
        if kwargs.get('scaler_path', None) is not None:
            self.v_scaler = get_scaler(scaler_path=kwargs.get('scaler_path'))
            print(f"ave_v scaler loaded from {kwargs.get('scaler_path')}")
        else:
            self.v_scaler = get_scaler(self)
        self.df['avg_volume'] = self.v_scaler.transform(torch.tensor(self.df['avg_volume']))
        
        return None
    
    def __len__(self):
        return len(self.df)
    
    @functools.lru_cache(maxsize=None)  # Cache data for faster training
    def __getitem__(self, idx):
        """Get item with id.
        
        Args:
            idx (int): dataset index

        Raises:
            AssertionError: [description]
            ValueError: [description]

        Returns:
            atom_weights: torch.Tensor shape (M, 1)
                weights of atoms in the material
            atom_fea: torch.Tensor shape (M, n_fea)
                features of atoms in the material
            self_fea_idx: torch.Tensor shape (M*M, 1)
                list of self indices
            nbr_fea_idx: torch.Tensor shape (M*M, 1)
                list of neighbor indices
            target: torch.Tensor shape (1,)
                target value for material
            cry_id: torch.Tensor shape (1,)
                input id for the material
        """
        df_idx = self.df.iloc[idx]
        composition = df_idx[self.com_tag]
        ave_v = df_idx[self.ave_v_tag]
        cry_id = df_idx[self.identifiers].values[0]
        cry_name = df_idx[self.identifiers].values[1]
        
        comp = Composition(composition)
        comp_dict = comp.get_el_amt_dict()
        elements = comp.elements
        weights = list(comp_dict.values())
        weights = np.atleast_2d(weights).T / np.sum(weights)
        
        try:
            atom_fea = np.vstack(
                [self.elem_features.get_fea(element.name) for element in elements]
            )
        except AssertionError:
            raise AssertionError(
                f"cry-id {cry_id} [{composition}] contains element types not in embedding"
            )
        except ValueError:
            raise ValueError(
                f"cry-id {cry_id} [{composition}] composition cannot be parsed into elements"
            )
        # Create densely commected graph. 
        nele = len(elements)
        self_fea_idx = []
        nbr_fea_idx = []
        for i, _ in enumerate(elements):
            self_fea_idx += [i] * nele
            nbr_fea_idx += list(range(nele))
        
        # convert all data to tensors
        elem_weights = torch.FloatTensor(weights)
        elem_fea = torch.FloatTensor(atom_fea)
        self_fea_idx = torch.LongTensor(self_fea_idx)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx)
        
        # Compute target composition.
        comp = torch.LongTensor([element.number for element in elements])
        comp = F.one_hot(comp-1, num_classes = 103)    # Predict Z number - 1.
        comp = comp.sum(axis=0)/comp.sum()
        comp = comp.view(1,-1)
        
        # Compute average volume.
        ave_v = torch.FloatTensor([ave_v]) if ave_v is not None else None
        
        # Save into PyG Data.
        data = Data(
            x = elem_fea,
            edge_index = torch.stack([self_fea_idx, nbr_fea_idx], dim=0),
            elem_weights = elem_weights,
            comp = comp,
            ave_v = ave_v,
            cry_id = cry_id,
            cry_name = cry_name,
        )

        return data
    
    def save_scaler(
        self, scaler_path: str,
    ) -> None:
        """Save scaler to scaler_path."""
        torch.save(self.v_scaler, scaler_path)
    
    @classmethod
    def from_formulas(
        cls, formulas: List[str],
    ) -> Data:
        """Initialize a composition dataset from atom types, like 
        generated from composition sampler."""
        df = {
            "material_id": None,
            "cif": None,
            "formula": formulas,
            "avg_volume": None,
        }
        df = pd.DataFrame(df)
        
        # Create a new instance of the class.
        instance = cls(data_path=None)
        instance.com_tag = "formula"
        instance.ave_v_tag = "avg_volume"
        instance.identifiers = ["material_id", "formula"]
        instance.df = df
        
        # Read matscholar embedding.
        fea_name = "matscholar-embedding.json"
        fea_path = osp.join(osp.dirname(osp.abspath(__file__)), fea_name)
        instance.elem_features = Featurizer.from_json(fea_path)
        instance.elem_emb_len = instance.elem_features.embedding_size
        return instance