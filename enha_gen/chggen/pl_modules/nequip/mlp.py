"""MLP modules for generate compisition (c), lattice (l) and number of atom (n) from latent space z."""

import torch
from torch import nn

class MLP_c(nn.Module):
    """MLP predictor for composition from z.
    
    Args:
        dim_z (int): dimension of the latend space (z).
        dim_composition (int): total number of composition for the material.
    """
    
    def __init__(
        self, dim_z: int, dim_composition: int,
    ) -> None:
        """Initialieze composition prediction MLP."""
        self.linear = nn.Linear(dim_z, dim_composition)
        self.softmax = nn.Softmax()
        return None
    
    def forward(
        self, z: torch.Tensor,
    ) -> torch.Tensor:
        """Predict composition from latend variable z."""
        out = self.softmax(self.linear(z))
                
class MLP_l(nn.Module):
    """MLP predictor for lattice parameter from z.
    
    Args:
        dim_z (int): dimension of the latend space (z).
    """
    
    def __init__(
        self, dim_z: int, dim_composition: int,
    ) -> None:
        """Initialieze lattice prediction MLP."""
        self.linear = nn.Linear(dim_z, 9)
        return None
    
    def forward(
        self, z: torch.Tensor,
    ) -> torch.Tensor:
        """Predict lattice from latend variable z."""
        out = self.linear(z).reshape(-1,3,3)
        
class MLP_n(nn.Module):
    """MLP predictor for number of atoms from z.
    
    Args:
        dim_z (int): dimension of the latend space (z).
    """
    
    def __init__(
        self, dim_z: int, dim_composition: int,
    ) -> None:
        """Initialieze composition prediction MLP."""
        self.linear = nn.Linear(dim_z, 1)
        return None
    
    def forward(
        self, z: torch.Tensor,
    ) -> torch.Tensor:
        """Predict composition from latent variable z."""
        out = self.linear(z)