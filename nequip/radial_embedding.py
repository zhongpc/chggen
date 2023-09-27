"""Initialize the graph network model."""

from torch import nn
from functools import partial
from nequip.basis import bessel

# radial embedding
class InitialEmbedding(nn.Module):
    def __init__(self, num_species, cutoff, emb_dim):
        super().__init__()
        self.embed_node_x = nn.Embedding(num_species, emb_dim)
        self.embed_node_z = nn.Embedding(num_species, emb_dim)
        self.embed_edge   = partial(bessel, start=0.0, end=cutoff, num_basis=16)
    
    def forward(self, data):
        # Embed node
        data.h_node_x = self.embed_node_x(data.x)
        data.h_node_z = self.embed_node_z(data.x)

        # Embed edge
        data.h_edge = self.embed_edge(data.edge_attr.norm(dim=-1))
        
        return data