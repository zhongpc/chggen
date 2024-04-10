import torch
import torch.nn as nn
import pickle
from torch_geometric.data import Data

from chggen.common.data_utils import (
    get_pbc_distances, 
    radius_graph_pbc,
    cart_to_frac_coords,
)
from chggen.pl_modules.nequip.radial_embedding import InitialEmbedding_PTE, InitialEmbedding_EE, InitialEmbedding_condition
from chggen.pl_modules.nequip.e3nn_nequip import NequIP_PTE, NequIP_EE, NequIP_condition
from chggen.pl_modules.gemnet.gemnet import GemNetT


class NequipTableDecoder(nn.Module):
    """Decoder with nequip equipped with periodic table information."""

    def __init__(
        self,
        max_neighbors = 40,
        cutoff = 6.,
        model_version = 'nequip_pte',
        irreps_node_x  = '32x0e',
        irreps_node_z  = '32x0e',
        irreps_hidden  = '32x0e + 32x1e + 8x2e',
        irreps_edge    = '32x0e + 32x1e + 8x2e',
        irreps_out     = '1x1e',
        num_convs      = 5,
        radial_neurons = [16, 64],
    ):
        super(NequipTableDecoder, self).__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_neighbors

        if model_version == 'nequip_pte':       # Periodic table embedding
            self.nequip = NequIP_PTE(
                init_embed = InitialEmbedding_PTE(num_periods=7, num_groups=18, cutoff=cutoff, emb_dim=32),
                irreps_node_x  = irreps_node_x,
                irreps_node_z  = irreps_node_z,
                irreps_hidden  = irreps_hidden,
                irreps_edge    = irreps_edge,
                irreps_out     = irreps_out,
                num_convs      = num_convs ,
                radial_neurons = radial_neurons,
                num_neighbors  = self.max_num_neighbors / 2,
            )
            
        elif model_version == 'nequip_ee':      # Element embedding
            self.nequip = NequIP_EE(
                init_embed = InitialEmbedding_EE(num_species=89, cutoff=cutoff, emb_dim=32),
                irreps_node_x  = irreps_node_x,
                irreps_node_z  = irreps_node_z,
                irreps_hidden  = irreps_hidden,
                irreps_edge    = irreps_edge,
                irreps_out     = irreps_out,
                num_convs      = num_convs ,
                radial_neurons = radial_neurons,
                num_neighbors  = self.max_num_neighbors / 2,
            )

        elif model_version == 'nequip_cond':      # Element embedding
            self.nequip = NequIP_condition(
                init_embed = InitialEmbedding_condition(num_species=89, cutoff=cutoff, emb_dim=16, cond_dim = 16),
                irreps_node_x  = irreps_node_x,
                irreps_node_z  = irreps_node_z,
                irreps_hidden  = irreps_hidden,
                irreps_edge    = irreps_edge,
                irreps_out     = irreps_out,
                num_convs      = num_convs ,
                radial_neurons = radial_neurons,
                num_neighbors  = self.max_num_neighbors / 2,
            )
            
        else:
            raise NotImplementedError(f"Model version {model_version} not implemented")

    def forward(
        self, 
        pred_cart_coords, 
        pred_atom_types, 
        num_atoms,
        lengths, 
        angles,
        properties = None, # the default is no property guidance
    ):
        """Forward pass of the decoder.
        
        Args:
            z: (N_cryst, num_latent)
            pred_cart_coords: (N_atoms, 3)
            pred_atom_types: (N_atoms, ), need to use atomic number e.g. H = 1
            num_atoms: (N_cryst,)
            lengths: (N_cryst, 3)
            angles: (N_cryst, 3)
        
        Returns:
            atom_frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, MAX_ATOMIC_NUM)
        """
        
        edge_index, to_jimages, num_bonds = radius_graph_pbc(
            pred_cart_coords, 
            lengths, 
            angles, 
            num_atoms, 
            self.cutoff, 
            self.max_num_neighbors,
            device=num_atoms.device,
        )
        
        out = get_pbc_distances(
            pred_cart_coords,
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

        

        if properties is None:
            data = Data(
            x       = pred_atom_types, # do not need minus 1 to accomodate the index. This is done in Nequip class. 
            pbc     = True,
            edge_index = edge_index,
            edge_attr = out['distance_vec'],
            property = torch.zeros_like(pred_atom_types, dtype = torch.float32, device = pred_atom_types.device)
        )
        else:       
            properties = properties.repeat_interleave(num_atoms, dim=0)
            data = Data(
                x       = pred_atom_types, # do not need minus 1 to accomodate the index. This is done in Nequip class. 
                pbc     = True,
                edge_index = edge_index,
                edge_attr = out['distance_vec'],
                property = properties
            )

        pred_cart_coord_diff = self.nequip(data)
            
        return pred_cart_coord_diff

class GemNetTDecoder(nn.Module):
    """Decoder with GemNetT."""

    def __init__(
        self,
        hidden_dim=128,
        latent_dim=256,
        max_neighbors=20,
        radius=6.,
    ):
        super(GemNetTDecoder, self).__init__()
        self.cutoff = radius
        self.max_num_neighbors = max_neighbors

        self.gemnet = GemNetT(
            num_targets=1,
            latent_dim=latent_dim,
            emb_size_atom=hidden_dim,
            emb_size_edge=hidden_dim,
            regress_forces=True,
            cutoff=self.cutoff,
            max_neighbors=self.max_num_neighbors,
            otf_graph=True,
        )

    def forward(
        self, 
        pred_cart_coords, 
        pred_atom_types,
        num_atoms,
        lengths, 
        angles,
        z = None,              # Z need to be None for no latent variable and decoder-only structure
    ):
        """ Forward pass of the decoder.
        
        Args:
            z: (N_cryst, num_latent)
            pred_cart_coords: (N_atoms, 3)
            pred_atom_types: (N_atoms, ), need to use atomic number e.g. H = 1
            num_atoms: (N_cryst,)
            lengths: (N_cryst, 3)
            angles: (N_cryst, 3)
        
        Returns:
            atom_frac_coords: (N_atoms, 3)
        """
        # (num_atoms, hidden_dim) (num_crysts, 3)
        pred_frac_coords = cart_to_frac_coords(
            pred_cart_coords, 
            lengths, 
            angles, 
            num_atoms
        )
        _, pred_cart_coord_diff = self.gemnet(
            z=z,
            frac_coords=pred_frac_coords,
            atom_types=pred_atom_types,
            num_atoms=num_atoms,
            lengths=lengths,
            angles=angles,
            edge_index=None,
            to_jimages=None,
            num_bonds=None,
        )
        return pred_cart_coord_diff

def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)