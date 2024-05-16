"""Initialize the graph network model."""
import torch
from torch import nn
from functools import partial
from .basis import bessel
from ..embeddings.element_table import PeriodicTable


def condition_embedding(x, embedding_dim=16, const=1000):
    """
    Forward pass of the ConditionEncoder.
    Args:
    - x (torch.Tensor): A tensor of shape (..., 1) where `...` denotes any number of dimensions, representing the scalar values to encode.
    Returns:
    - torch.Tensor: The sinusoid embedding of x with shape (..., embedding_dim).
    """
    # Apply the sinusoid encoding
    div_term = torch.pow(2.0, torch.arange(0., embedding_dim, device=x.device) / embedding_dim) * const
    encoding = torch.sin(x / div_term)
    return encoding


# radial embedding
class InitialEmbedding_EE(nn.Module):
    def __init__(self, num_species, cutoff, emb_dim, radical_dim):
        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, emb_dim)
        self.embed_node_z = nn.Embedding(num_species, emb_dim)
        self.embed_edge   = partial(bessel, start=0.0, end=cutoff, num_basis= radical_dim)
    
    def forward(self, data):
        # Embed node
        x = data.x
        # x[(x>=57) * (x<=70)] = 57   # put La series in one element embedding
        # x[(x>=89) * (x<=102)] = 89  # put Ac series in one element embedding
        x = x - 1                   # index from 0.
        
        data.h_node_x = self.embed_node_x(x)
        data.h_node_z = self.embed_node_z(x)

        # Embed edge
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        
        return data


# radial embedding
class InitialEmbedding_condition(nn.Module):
    def __init__(self, num_species, cutoff, emb_dim, radical_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embed_node_x = nn.Embedding(num_species, emb_dim)
        self.embed_node_z = nn.Embedding(num_species, emb_dim)
        self.scale_W = nn.Linear(emb_dim, emb_dim, bias=False)
        # self.embed_condition = nn.Embedding(1, cond_dim)
        self.embed_edge   = partial(bessel, start=0.0, end=cutoff, num_basis= radical_dim)
    
    def forward(self, data):
        # Embed node
        x = data.x
        # x[(x>=57) * (x<=70)] = 57   # put La series in one element embedding
        # x[(x>=89) * (x<=102)] = 89  # put Ac series in one element embedding
        x = x - 1                   # index from 0.

        condition = data.property.view(-1, 1)

        # print('condition:', condition)
        # print('condition_shape:', condition.shape)

        node_x = self.embed_node_x(x)
        node_z = self.embed_node_z(x)
        node_cond = self.scale_W(condition_embedding(condition, embedding_dim= self.emb_dim, const=1000))
        
        data.h_node_x = node_x + node_cond # ), axis = 1)
        data.h_node_z = node_z + node_cond # ), axis = 1)

        # Embed edge
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        
        return data