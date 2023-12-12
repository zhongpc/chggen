import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from chggen.pl_modules.embeddings import MAX_ATOMIC_NUM
from chggen.pl_modules.gemnet.gemnet import GemNetT
from chggen.common.data_utils import get_pbc_distances, frac_to_cart_coords, cart_to_frac_coords, radius_graph_pbc
from chggen.pl_modules.nequip.radial_embedding_table import InitialEmbedding
from chggen.pl_modules.nequip.e3nn_nequip_table import NequIP




def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)

class NequipTableDecoder(nn.Module):
    """Decoder with nequip equipped with periodic table information."""

    def __init__(
        self,
        max_neighbors=20,
        cutoff = 6.,
    ):
        super(NequipTableDecoder, self).__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_neighbors
        self.nequip = NequIP(init_embed     = InitialEmbedding(num_periods=7, num_groups=18, cutoff=cutoff, emb_dim= 32),
                            irreps_node_x  = '32x0e',
                            irreps_node_z  = '32x0e',
                            irreps_hidden  = '32x0e + 16x1e + 8x2e',
                            irreps_edge    = '32x0e + 16x1e + 8x2e',
                            irreps_out     = '1x1e',
                            num_convs      = 3,
                            radial_neurons = [16, 64],
                            num_neighbors  = 12,
                        )

        # self.fc_atom = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)

    def forward(self, pred_frac_coords, pred_atom_types, num_atoms,
                lengths, angles):
        """
        args:
            z: (N_cryst, num_latent)
            pred_frac_coords: (N_atoms, 3)
            pred_atom_types: (N_atoms, ), need to use atomic number e.g. H = 1
            num_atoms: (N_cryst,)
            lengths: (N_cryst, 3)
            angles: (N_cryst, 3)
        returns:
            atom_frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, MAX_ATOMIC_NUM)
        """
        # cutoff = 6.0
        # max_neighbors = 12
        edge_index, to_jimages, num_bonds = radius_graph_pbc(
                        pred_frac_coords, lengths, angles, num_atoms, self.cutoff, self.max_num_neighbors,
                        device=num_atoms.device)
        
        out = get_pbc_distances(
                                pred_frac_coords,
                                edge_index,
                                lengths,
                                angles,
                                to_jimages,
                                num_atoms,
                                num_bonds,
                                coord_is_cart=True,
                                return_offsets=True,
                                return_distance_vec=True,
                            )

        data = Data(
            x       = pred_atom_types, # do not need minus 1 to accomodate the index. This is done in Nequip class. 
            pos     = pred_frac_coords,
            # cell    = all_cells,
            pbc     = True,
            edge_index = edge_index,
            edge_attr = out['distance_vec']
        )

        pred_cart_coord_diff = self.nequip(data)
        
        return pred_cart_coord_diff