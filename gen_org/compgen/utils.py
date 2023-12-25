"""Class containing common utilities."""

from __future__ import annotations

from pathlib import Path
import json

import torch
import numpy as np


from pymatgen.core import Composition, Element

EPSILON = 1e-7


        
def read_json(
    json_path: str,
) -> list:
    """Read the json file.
    
    Args:
        json_path (str): path to the json file.
    
    Returns:
        dic: dictionary stored in json file.
    """
    with open(json_path, 'r') as file:
        ls = json.load(file)
    return ls
        
def write_json(
    dic: dict,
    json_path: str,
) -> None:
    """Write the dictionary to json file.
    
    Args:
        dic (dict): dictionary to be written to json file.
        json_path (str): path to the json file.
    """
    with open(json_path, 'w') as file:
        json.dump(dic, file)
    return None

def get_scaler_from_data_list(df, key):
    """Get scaler (std, mean) assigned by key from data_list."""
    targets = torch.tensor(df[key])
    scaler = StandardScalerTorch()
    scaler.fit(targets)
    return scaler

def get_scaler(
    dataset = None, scaler_path = None,
):
    """Get scalers (mean, std) from scaler_path or from data."""
    if scaler_path is None:     # Compute
        lattice_scaler = get_scaler_from_data_list(
            dataset.df,
            key='avg_volume',
        )
    else:                       # Load
        lattice_scaler = torch.load(scaler_path)
    return lattice_scaler

def atom_types2compositions(
    atom_types: List[torch.Tensor], 
) -> List[Composition]:
    """Convert list of atom types to list of compositions."""
    compositions = []
    for atom_type in atom_types:
        # Convert atomic numbers to compositions
        compositions.append(Composition("".join([str(Element.from_Z(int(atom))) for atom in atom_type])))
    return compositions

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.ave = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.ave = self.sum / self.count

class StandardScalerTorch(object):
    """Normalizes the targets of a dataset."""

    def __init__(
        self, means = None, stds = None
    ) -> None:
        self.means = means
        self.stds = stds

    def fit(self, X) -> None:
        X = X.clone().detach().float()
        self.means = torch.mean(X, dim=0)
        # https://github.com/pytorch/pytorch/issues/29372
        self.stds = torch.std(X, dim=0, unbiased=False) + EPSILON
        return None

    def transform(self, X) -> torch.Tensor:
        X = X.clone().detach().float()
        return (X - self.means) / self.stds

    def inverse_transform(self, X) -> torch.Tensor:
        X = X.clone().detach().float()
        return X * self.stds + self.means

    def match_device(self, tensor) -> None:
        if self.means.device != tensor.device:
            self.means = self.means.to(tensor.device)
            self.stds = self.stds.to(tensor.device)
        return None
    
    def copy(self) -> StandardScalerTorch:
        return StandardScalerTorch(
            means = self.means.clone().detach(),
            stds = self.stds.clone().detach()
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"means: {self.means.tolist()}, "
            f"stds: {self.stds.tolist()})"
        )