"""Module for creating operations needed in DiffCSP."""
from __future__ import annotations
import torch

class PModulo:
    """Plus and minus considering periodicity."""
    def __init__(self) -> None:
        """Initialize PModulo."""
        self.pbc = torch.tensor([1, 1, 1])  
        
    def to(self, device: torch.device) -> PModulo:
        """Move to device."""
        self.pbc = self.pbc.to(device)
        return self
        
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the modulo considering periodicity."""
        return self.modulo(x)
    
    def modulo(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the modulo considering periodicity.

        Args:
            x (Tensor): tensor

        Returns:
            mod (Tensor): modulo considering periodicity
        """
        return x - torch.floor(x / self.pbc) * self.pbc
    
    def sub(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the sub considering periodicity.

        Args:
            x (Tensor): tensor 1
            y (Tensor): tensor 2

        Returns:
            diff (Tensor): summation considering periodicity
        """
        return x + y - torch.floor((x + y) / self.pbc) * self.pbc
    
    def minus(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the difference considering periodicity.

        Args:
            x (Tensor): tensor 1
            y (Tensor): tensor 2

        Returns:
            diff (Tensor): difference considering periodicity
        """
        return x - y - torch.floor((x - y) / self.pbc) * self.pbc
    
    def __repr__(self):
        return f"PMinus({self.pbc=})" 

if __name__ == "__main__":
    # Test PModulo.
    pmodulo = PModulo()
    x = torch.randn(9).reshape(3,3)
    y = torch.randn(9).reshape(3,3)
    
    # Sub.
    print('x+y:', x+y)
    print('x+y (p):', pmodulo.sub(x, y))
    
    # Minus.
    print('x-y:', x-y)
    print('x-y (p):', pmodulo.minus(x, y))
    
    # Modulo.
    print('x:', x)
    print('x (p):', pmodulo.modulo(x))