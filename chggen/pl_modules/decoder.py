import torch
import torch.nn as nn
import torch.nn.functional as F

from chggen.pl_modules.embeddings import MAX_ATOMIC_NUM
from chggen.pl_modules.gemnet.gemnet import GemNetT
from chggen.common.data_utils import get_pbc_distances, frac_to_cart_coords, cart_to_frac_coords, radius_graph_pbc
from nequip.radial_embedding import InitialEmbedding
from nequip.e3nn_nequip import NequIP
from ase import Atoms

from torch_geometric.data import Data



def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


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
        self.fc_atom = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)

    def forward(self, z, pred_frac_coords, pred_atom_types, num_atoms,
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
        # (num_atoms, hidden_dim) (num_crysts, 3)
        h, pred_cart_coord_diff = self.gemnet(
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
        pred_atom_types = self.fc_atom(h)
        return pred_cart_coord_diff, pred_atom_types

class NequipDecoder(nn.Module):
    """Decoder with nequip """

    def __init__(
        self,
        hidden_dim = 128,
        latent_dim = 64,
        max_neighbors=20,
        cutoff = 6.,
    ):
        super(NequipDecoder, self).__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_neighbors
        self.nequip = NequIP(init_embed     = InitialEmbedding(num_species= MAX_ATOMIC_NUM, cutoff=cutoff, emb_dim= 32),
                            irreps_node_x  = '32x0e',
                            irreps_node_z  = '32x0e',
                            irreps_hidden  = '32x0e + 16x1e + 8x2e',
                            irreps_edge    = '32x0e + 16x1e + 8x2e',
                            irreps_out     = '1x1e',
                            irreps_type    = '94x0e',
                            num_convs      = 3,
                            radial_neurons = [16, 64],
                            num_neighbors  = 12,
                        )

        # self.fc_atom = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)

    def forward(self, z, pred_frac_coords, pred_atom_types, num_atoms,
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

        # # compute the cell 3*3 matrix via ASE
        # all_cells = []
        # for length, angle in zip(lengths, angles):
        #     params = torch.cat((length, angle), dim = 0)
        #     atoms = Atoms(cell=params, pbc=True)
        #     cell = atoms.get_cell() 
        #     all_cells.append(cell)

        # all_cells = torch.tensor(all_cells)
        # all_cells = all_cells.repeat_interleave(num_atoms, dim = 0)

        # print(pred_atom_types)

        data = Data(
            x       = pred_atom_types - 1, # minus 1 to accomodate the index 
            pos     = pred_frac_coords,
            # cell    = all_cells,
            pbc     = True,
            edge_index = edge_index,
            edge_attr = out['distance_vec']
        )

        pred_cart_coord_diff, pred_atom_types = self.nequip(data)
        
        return pred_cart_coord_diff, pred_atom_types


class NequipLatticeDecoder(nn.Module):
    """Decoder with nequip """

    def __init__(
        self,
        hidden_dim = 128,
        latent_dim = 64,
        max_neighbors=20,
        cutoff = 6.,
    ):
        super(NequipDecoder, self).__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_neighbors
        self.nequip = NequIP(init_embed     = InitialEmbedding(num_species= MAX_ATOMIC_NUM, cutoff=cutoff, emb_dim= 32),
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

    def forward(self, z, pred_frac_coords, pred_atom_types, num_atoms,
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

        # # compute the cell 3*3 matrix via ASE
        # all_cells = []
        # for length, angle in zip(lengths, angles):
        #     params = torch.cat((length, angle), dim = 0)
        #     atoms = Atoms(cell=params, pbc=True)
        #     cell = atoms.get_cell() 
        #     all_cells.append(cell)

        # all_cells = torch.tensor(all_cells)
        # all_cells = all_cells.repeat_interleave(num_atoms, dim = 0)

        # print(pred_atom_types)

        data = Data(
            x       = pred_atom_types - 1, # minus 1 to accomodate the index 
            pos     = pred_frac_coords,
            # cell    = all_cells,
            pbc     = True,
            edge_index = edge_index,
            edge_attr = out['distance_vec']
        )

        pred_cart_coord_diff, pred_atom_types = self.nequip(data)
        return pred_cart_coord_diff, pred_atom_types