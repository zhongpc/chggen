import torch
import torch.nn as nn

from e3nn       import o3
from e3nn.nn    import Gate

from nequip.radial_embedding_table import InitialEmbedding
from chggen.common.data_utils import get_pbc_distances, frac_to_cart_coords, cart_to_frac_coords, radius_graph_pbc
from torch_geometric.data import Data

from nequip.e3nn_nequip_table import NequIP



if __name__ == '__main__':
    nequip = NequIP(init_embed = InitialEmbedding(num_periods = 7, num_groups = 18, cutoff = 6.0, emb_dim= 32),
                            irreps_node_x  = '32x0e',
                            irreps_node_z  = '32x0e',
                            irreps_hidden  = '32x0e + 16x1e + 8x2e',
                            irreps_edge    = '32x0e + 16x1e + 8x2e',
                            irreps_out     = '1x0e',
                            num_convs      = 3,
                            radial_neurons = [16, 64],
                            num_neighbors  = 12,
                        )
    
    pred_frac_coords = torch.tensor([[0,0,0], [0.5,0.5,0.5], [0.25,0.25,0.25], [0.75,0.75,0.75]])
    lengths = torch.tensor([5,5,5]).view(-1, 3)
    angles = torch.tensor([90, 90, 90]).view(-1, 3)
    num_atoms = torch.tensor([4])
    
    edge_index, to_jimages, num_bonds = radius_graph_pbc(
        pred_frac_coords, 
        lengths, 
        angles, 
        num_atoms, 
        radius=6.0, 
        max_num_neighbors_threshold=12, 
        device=num_atoms.device,
    )
    
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
            x = torch.tensor([1,1,1,1]) , # minus 1 to accomodate the index 
            pos = pred_frac_coords,
            # cell    = all_cells,
            pbc = True,
            edge_index = edge_index,
            edge_attr = out['distance_vec']
        )

    pred_energy = nequip(data)
    print(pred_energy)
    print(len(pred_energy))