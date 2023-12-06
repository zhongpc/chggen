"""Initialize the graph network model."""

from torch import nn
from functools import partial
from .basis import bessel
from element_dict.element_dict import PeriodicTable



# radial embedding
class InitialEmbedding(nn.Module):
    def __init__(self, num_periods, num_groups, cutoff, emb_dim):
        super().__init__()
        self.periodic_table = PeriodicTable('/home/xzdai/ceder_group/material_dircovery/chggen/element_dict/elements.csv')
        self.embed_node_x_period = nn.Embedding(num_periods, emb_dim)
        self.embed_node_z_period = nn.Embedding(num_periods, emb_dim)
        self.embed_node_x_group = nn.Embedding(num_groups, emb_dim)
        self.embed_node_z_group = nn.Embedding(num_groups, emb_dim)
        self.embed_edge   = partial(bessel, start=0.0, end=cutoff, num_basis=16)
    
    def forward(self, data):
        # Embed node
        x_period = self.periodic_table.get_property_from_atomic_numbers(data.x, 'period')-1 # index from 0.
        x_group = self.periodic_table.get_property_from_atomic_numbers(data.x, 'period')-1  # index from 0.
        
        data.h_node_x_period = self.embed_node_x_period(x_period)
        data.h_node_z_period = self.embed_node_z_period(x_period)
        data.h_node_x_group = self.embed_node_x_group(x_group)
        data.h_node_z_group = self.embed_node_z_group(x_group)

        # Embed edge
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        
        return data