"""Initialize the graph network model."""
import torch
from torch import nn
from functools import partial
from .basis import bessel
from ..embeddings.element_table import PeriodicTable


# class ConditionEncoder(nn.Module):
#     def __init__(self, embedding_dim=16, const=1000):
#         """
#         Initialize the ConditionEncoder module.
#         Args:
#         - embedding_dim (int): The dimensionality of the sinusoid embedding.
#         - const (float): A constant used to adjust frequencies of the sinusoid functions.
#         """
#         super(ConditionEncoder, self).__init__()
#         self.embedding_dim = embedding_dim
#         self.const = const
#         # Generate divisors for the denominator based on the embedding dimension.
#         # These are powers of 2, scaled by the constant.
#         self.div_term = torch.pow(2.0, torch.arange(0., embedding_dim) / embedding_dim) * const


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
class InitialEmbedding_PTE(nn.Module):
    def __init__(self, num_periods, num_groups, cutoff, emb_dim):
        super().__init__()
        self.periodic_table = PeriodicTable()
        self.embed_node_x_period = nn.Embedding(num_periods, emb_dim)
        self.embed_node_z_period = nn.Embedding(num_periods, emb_dim)
        self.embed_node_x_group = nn.Embedding(num_groups, emb_dim)
        self.embed_node_z_group = nn.Embedding(num_groups, emb_dim)
        self.embed_edge   = partial(bessel, start=0.0, end=cutoff, num_basis=16)
    
    def forward(self, data):
        # Embed node
        x_period = self.periodic_table.get_property_from_atomic_numbers(data.x, 'period')-1 # index from 0.
        x_group = self.periodic_table.get_property_from_atomic_numbers(data.x, 'group')-1  # index from 0.
        
        data.h_node_x_period = self.embed_node_x_period(x_period)
        data.h_node_z_period = self.embed_node_z_period(x_period)
        data.h_node_x_group = self.embed_node_x_group(x_group)
        data.h_node_z_group = self.embed_node_z_group(x_group)

        # Embed edge
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        
        return data
    
    

# radial embedding
class InitialEmbedding_EE(nn.Module):
    def __init__(self, num_species, cutoff, emb_dim):
        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, emb_dim)
        self.embed_node_z = nn.Embedding(num_species, emb_dim)
        self.embed_edge   = partial(bessel, start=0.0, end=cutoff, num_basis=16)
    
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
    def __init__(self, num_species, cutoff, emb_dim, cond_dim):
        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, emb_dim)
        self.embed_node_z = nn.Embedding(num_species, emb_dim)
        self.embed_edge   = partial(bessel, start=0.0, end=cutoff, num_basis=16)
    
    def forward(self, data):
        # Embed node
        x = data.x
        # x[(x>=57) * (x<=70)] = 57   # put La series in one element embedding
        # x[(x>=89) * (x<=102)] = 89  # put Ac series in one element embedding
        x = x - 1                   # index from 0.

        condition = data.property

        node_x = self.embed_node_x(x)
        node_z = self.embed_node_z(x)
        node_cond = condition_embedding(condition, embedding_dim=16, const=1000)
        # node_cond = node_cond.repeat_interleave(datas.num_atoms, dim=0)
        
        data.h_node_x = torch.cat((node_x, node_cond), axis = 1)
        data.h_node_z = torch.cat((node_z, node_cond), axis = 1)

        # Embed edge
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        
        return data