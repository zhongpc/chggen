"""Modules for generation sinusoidal positional embedding."""
from __future__ import annotations
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



class PositionEmbedding(nn.Module):
    """Position embedding by sinusoidal position encoding.
    
    Args:
        max_position_len (int): Maximum length of position.
        model_dim (int): Embedding length.
        
    Mathematics:
        PE_(pos, 2i) = sin(pos/10000^(2i/model_dim))
        PE_(pos, 2i+1) = cos(pos/10000^(2i/model_dim)
    """
    def __init__(
        self, max_position_len: int, model_dim: int,
    ) -> None:
        super(PositionEmbedding, self).__init__()
        """Initialize PositionEmbedding."""
        self.pos_mat = torch.arange(max_position_len).reshape(-1,1)
        self.i_mat = torch.pow(10000, torch.arange(0, model_dim, 2).reshape(1,-1)/model_dim)
        self.pe_table = torch.zeros(max_position_len, model_dim)
        self.pe_table[:,0::2] = torch.sin(self.pos_mat/self.i_mat)
        self.pe_table[:,1::2] = torch.cos(self.pos_mat/self.i_mat)
    
    def to(self, device: torch.device) -> PositionEmbedding:
        """Move to device."""
        self.pe_table = self.pe_table.to(device)
        return self  
    
    def forward(
        self, position: int,
    ) -> torch.Tensor:
        """Fetch position embedding corresponding to the position given.
        
        Args:
            position (int): Position that needs to be embedded.
            
        Returns:
            emb (Tensor): Position embedding.
        """
        emb = self.pe_table[position]
        return emb
    
    
    
if __name__ == '__name__':

    max_position_len = 50       # Maximum length of position.
    model_dim = 100              # Embedding length.
    pe = PositionEmbedding(max_position_len, model_dim)
    plt.imshow(pe.pe_table)