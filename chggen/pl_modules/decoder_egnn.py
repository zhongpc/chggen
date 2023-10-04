import torch
import torch.nn as nn
import torch.nn.functional as F

from chggen.pl_modules.embeddings import MAX_ATOMIC_NUM
from chggen.common.data_utils import get_pbc_distances, frac_to_cart_coords, cart_to_frac_coords, radius_graph_pbc

from pymatgen.core import Structure
from chgnet.model.model import BatchedGraph
from chgnet.graph.converter import CrystalGraphConverter
from chggen.pl_modules.encoder import CHGNet_encoder
from typing import Tuple
import numpy as np

NUM_SPECIES = 94                                                # Number of species in the dataset.

def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)



def unsorted_segment_sum(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Compute the unsorted segment by sum.
    
    Args:
        data (Tensor): data to be segmented.
        segment_ids (Tensor): indices of segments.
        num_segments (int): number of segments.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Initialize empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(
    data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Compute the unsorted segment by mean.
    
    Args:
        data (Tensor): data to be segmented.
        segment_ids (Tensor): indices of segments.
        num_segments (int): number of segments.
    """
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Initialize empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    count = data.new_full(result_shape, 0)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

def get_reduced_id_from_tensor(
    tensor: torch.Tensor,
) -> torch.Tensor:
    """Get the index of value that first appears in the integer tensor 
    in sequence.
    
    Example:
        a = torch.tensor([0,0,0,0,1,1,1,1,1,2,2,2,])
        ids = get_reduced_id_from_tensor(a)
        print(ids)      # tensor([0, 1, 2])
    """
    tensor = [torch.where(tensor == i)[0][0].item() for i in range(tensor.max().item()+1)]
    tensor = torch.tensor(tensor)
    return tensor


class E_GCL(nn.Module):
    """E(n) equivariant convolutional layer.
    
    Args:
        input_nf (int): the number of features of input.
        output_nf (int): the number of features of output.
        hidden_nf (int): the number of features of hidden layer.
        edges_in_d (int): the number of features of edge attribute.
        ft_basis (int): the number of Fourier transform basis expansion.
        x_dim (int): dimension of x in real space.
        act_fn (nn.Module): activate function.
        residual (bool): residual connection or not.
        attention (bool): attention mechanism or not.
        normalize (bool): normalize or not.
        coords_agg (str): coordinates aggregate method.
        tanh (bool): add tanh activation function at the output of phi_x(m_ij).
    """
    def __init__(
        self,
        input_nf: int,
        output_nf: int,
        hidden_nf: int,
        edges_in_d: int = 0,
        ft_basis: int = 10,
        x_dim: int = 3,
        act_fn: nn.Module = nn.SiLU(),
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        coords_agg: str = "mean",
        tanh: bool = False,
    ) -> None:
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8                     # Small number to avoid division by zero.
        self.ft_basis = ft_basis                # Number of FT basis to compute.
        edge_coords_nf = x_dim * ft_basis       # Number of features in the edge coordinates.
        edge_lattice_nf = x_dim **2             # Number of features in the edge lattice.
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edge_lattice_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )
        
        layer = nn.Linear(hidden_nf, 1, bias=False)             # ??? Why no bias?
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001) # ??? Why gain is 0.001?

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid(),
            )
        
    def edge_model(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        l_feat: torch.Tensor,
        ft: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """Edge model to compute edge features for a given graph.
        
        Args:
            source (Tensor): source node features.
            target (Tensor): target node features.
            l_feat (Tensor): l_feat features with shape (len(edge_index), l_feat per edge (9)).
            ft (Tensor): FT bases expansion of relative fractional distance.
            edge_attr (Tensor): edge attribute.
        
        Returns:
            out (Tensor): next layer edge attribute.
        """
        if edge_attr is None:   # Not used
            out = torch.cat([source, target, l_feat, ft], dim=1)
        else:
            out = torch.cat([source, target, l_feat, ft, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out
    
    def lattice2feat(
        self,
        l: torch.Tensor,
    ) -> torch.Tensor:
        """Compute lattice feature LL^T from batched lattice.
        
        Args:
            l (Tensor): lattice vectors with shape (batched number of nodes, 
                lattice index (3), components (3)).
        
        Returns:
            l_feat (Tensor): lattice feature LL^T with shape (batched number 
            of nodes, reshaped lattice feature LL^T (9)).
            
        NOTE: If the lattice is defined as:
            L = [[l1],
                [l2],
                [l3]]
              = [[l11, l12, l13],
                [l21, l22, l23],
                [l31, l32, l33]]
            Then the lattice feature should be LL^T to accommodate equivariance.
            (LR)@(LR)^T = LRR^TL^T = LL^T
            
            If the lattice is defined as:
            L = [[l1, l2, l3]]
              = [[l11, l21, l31],
                [l12, l22, l32],
                [l13, l23, l33]]
            Then the lattice feature should be L^TL to accommodate equivariance.
            (RL)^T@(RL) = L^TR^TRL = L^TL
            
        TODO: Adapt it with Niggli strategy.
        """                    
        # l_T = torch.transpose(l, dim0=1, dim1=2)            # l_T [batch, abc (3), xyz (3)]
        l_feat = torch.einsum('iax,ibx->iab', l, l)       
        l_feat = l_feat.reshape(-1, l.shape[1]*l.shape[2])  # l_feat [batch, feature (9)]
        return l_feat
        
    def node_model(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        node_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Node model to compute node features for a given graph.
        
        Args:
            h (Tensor): hidden node features.
            edge_index (Tensor): edge indices.
            edge_feat (Tensor): edge features.
            node_attr (Tensor): node attribute.
        
        Returns:
            Tensor: next layer node features.
            Tensor: aggregated node features.
        """
        row, col = edge_index
        agg = unsorted_segment_sum(edge_feat, row, num_segments=h.size(0))
        if node_attr is not None:
            agg = torch.cat([h, agg, node_attr], dim=1)
        else:
            agg = torch.cat([h, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = h + out
        return out, agg
    
    def coord2strucfactor(
        self,
        edge_index: torch.Tensor,
        coord: torch.Tensor,
        ft_basis: int,
    ) -> torch.Tensor:
        """Compute FT bases expansion of relative fractional distance.
        (f1, f2, f3) -> (cos(2pi*f1), sin(2pi*f1), cos(2pi*f2), sin(2pi*f2), 
        cos(2pi*f3), sin(2pi*f3)), ...
        
        Args:
            edge_index (Tensor): edge indices.
            coord (Tensor): coordinates.
            ft_basis (int): number of FT basis to compute.
        """
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        ft_feat = torch.arange(2, ft_basis+2, 2) * torch.pi * coord_diff.unsqueeze(-1)
        ft_feat = torch.cat([torch.sin(ft_feat), torch.cos(ft_feat)], dim = -1) # [batch, x_dim, f_basis]
        ft_feat = ft_feat.reshape(-1, ft_basis * ft_feat.shape[1])
        return ft_feat 
    
    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        l: torch.Tensor,
        coord: torch.Tensor,
        edge_attr: [torch.Tensor, None] = None,
        node_attr: [torch.Tensor, None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of one E_GCN layer.
        
        Args:
            h (Tensor): hidden node features.
            edge_index (Tensor): edge indices.
            l (Tensor): lattice vectors with shape like (batched number of nodes, 
                lattice index (3), components (3)).
            coord (Tensor): coordinates.
            edge_attr (Tensor, optional): edge attribute.
            node_attr (Tensor, optional): node attribute.
        
        Returns:
            h (Tensor): next layer node features.
        """
        row, col = edge_index
        l_feat = self.lattice2feat(l)                                                   # lattice feature
        ft_feat = self.coord2strucfactor(edge_index, coord, ft_basis= self.ft_basis)    # structure factor feature
        edge_feat = self.edge_model(h[row], h[col], l_feat[row], ft_feat, edge_attr)    
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        
        return h
            
class EGNN(nn.Module):
    """Equivariant GNN implementation.
    
    Args:
        in_node_nf (int): the number of attribute for 'h' at the input.
        hidden_nf (int): the number of hidden features.
        in_edge_nf (int): the number of features for the edge attribute.
        ft_basis (int): the number of Fourier transform basis expansion coordination feature.
        x_dim (int): dimension of x in real space.
        device (str): device to run the model. (e.g. 'cpu', 'cuda:0', ...)
        act_fn (nn.Module): activate function.
        n_layers (int): number of layers for the EGNN.
        residual (bool): use residual connections or not. Default True is better in general.
        attention (bool): use attention mechanism or not.
        normalize (bool): normalize the coordinates messages such that:
            instead of: x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)
            we get:     x^{l+1}_i = x^{l}_i + Σ(x_i - x_j)phi_x(m_ij)/||x_i - x_j||
            We noticed it may help in the stability or generalization in some future works.
            We didn't use it in our paper.
        tanh (bool): Sets a tanh activation function at the output of phi_x(m_ij). I.e. it bounds 
            the output of phi_x(m_ij) which definitely improves in stability but it may decrease in 
            accuracy. We didn't use it in our paper.
    """
    def __init__(
        self,
        in_node_nf: int,
        hidden_nf: int,
        out_node_nf: int,
        in_edge_nf: int = 0,
        ft_basis: int = 10,
        x_dim: int = 3,
        device: str = "cpu",
        act_fn: nn.Module = nn.SiLU(),
        n_layers: int = 4,
        residual: bool = True,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
    ) -> None:
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.x_dim = x_dim
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out_l_weight = nn.Linear(self.hidden_nf, x_dim**2)
        self.embedding_out_x = nn.Linear(self.hidden_nf, x_dim)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    ft_basis=ft_basis,
                    x_dim= x_dim,
                    act_fn=act_fn,
                    residual=residual,
                    attention=attention,
                    normalize=normalize,
                    tanh=tanh,
                ),
            )
        self.to(self.device)
        
    def forward(
        self,
        h: torch.Tensor,
        l: torch.Tensor,
        x: torch.Tensor,
        edges: torch.Tensor,
        edge_attr: torch.Tensor,
        atom_owners: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for EGNN model.
        
        Args:
            h (Tensor): node features.
            l (Tensor): lattice vectors with shape like (batched number of nodes, 
                lattice index (3), components (3)).
            x (Tensor): coordinates.
            edges (Tensor): edge indices.
            edge_attr (Tensor): edge features.
            atom_owner (Tensor): owners of atoms showing which crystal structures 
                the atoms belongs to.
        """
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h = self._modules["gcl_%d" % i](h, edges, l, x, edge_attr=edge_attr)
        
        agg = unsorted_segment_mean(h, atom_owners, num_segments=atom_owners.max().item()+1)
        l_weight = self.embedding_out_l_weight(agg).reshape(-1, self.x_dim, self.x_dim)
        l = l[get_reduced_id_from_tensor(atom_owners)]          # Reduce l to each structure one lattice.
        l = torch.einsum('iax,ixb->iab', l_weight, l)                        # [# structure, abc (3), xyz (3)]
        x = self.embedding_out_x(h)
        return l, x
    

class EGNNDecoder(nn.Module):
    """Decoder with nequip """

    def __init__(
        self,
        hidden_dim = 128,
        latent_dim = 64,
        max_neighbors=20,
        cutoff = 6.,
    ):
        super(EGNNDecoder, self).__init__()
        # self.encoder = encoder
        self.cutoff = cutoff
        self.max_num_neighbors = max_neighbors

        self.egnn = EGNN(
                    in_node_nf=NUM_SPECIES,
                    hidden_nf=32,
                    out_node_nf=1,
                    in_edge_nf=1,
                    ft_basis=10,
                    x_dim= 3,
                )

        # self.fc_atom = nn.Linear(hidden_dim, MAX_ATOMIC_NUM)

    def forward(self, pred_frac_coords, pred_lattices,  pred_atom_type_embs,
                edges, edge_attr, atom_owners):
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

        out = self.egnn(h = pred_atom_type_embs, l = pred_lattices, x = pred_frac_coords, 
                   edges = edges, edge_attr = edge_attr, atom_owners = atom_owners)
        

        return out

        
        # edge_index, to_jimages, num_bonds = radius_graph_pbc(
        #                 pred_frac_coords, lengths, angles, num_atoms, self.cutoff, self.max_num_neighbors,
        #                 device=num_atoms.device)
        # out = get_pbc_distances(
        #                         pred_frac_coords,
        #                         edge_index,
        #                         lengths,
        #                         angles,
        #                         to_jimages,
        #                         num_atoms,
        #                         num_bonds,
        #                         coord_is_cart=True,
        #                         return_offsets=True,
        #                         return_distance_vec=True,
        #                     )

        # pred_cart_coord_diff, pred_atom_types = self.nequip(data)
        # return pred_cart_coord_diff, pred_atom_types
    


if __name__ == "__main__":
    
    ################################################################################
    # Initialize structures.
    ################################################################################
    crys_converter = CrystalGraphConverter(atom_graph_cutoff=5,
                                            bond_graph_cutoff=3)
    chgnet= CHGNet_encoder.load()
    
    # Construct a rock salt MgO structure.
    mgo_structure = Structure(
        lattice=[
            [0, 2.13, 2.13], 
            [2.13, 0, 2.13], 
            [2.13, 2.13, 0]
        ],
        species=["Mg", "O"],
        coords=[
            [0, 0, 0], 
            [0.5, 0.5, 0.5],        
        ],
        coords_are_cartesian=False,     # Default to False.
    )

    
    ################################################################################
    # Basis setting.
    ################################################################################
    x_dim = 3
    
    ################################################################################
    # Crystal graph construction.
    ################################################################################
    # Crystal graph construction for single structure.
    crys_graph = crys_converter(mgo_structure)
    atom_graph = crys_graph.atom_graph
    print('atom graph size:', atom_graph.shape)
    
    # Batch graphs for parallel computing.
    crys_graph_list = [crys_graph, crys_graph]
    
    batched_graph = BatchedGraph.from_graphs(
                                    crys_graph_list,
                                    bond_basis_expansion=chgnet.bond_basis_expansion,   # bond basis expansion function
                                    angle_basis_expansion=chgnet.angle_basis_expansion, # angle basis expansion function
                                    compute_stress= False,
                                )
    batched_atom_graph = batched_graph.batched_atom_graph
    print('batched atom graph:', batched_atom_graph.shape)

    edges = batched_atom_graph.T
    print('edges index size:', edges.shape)

    edge_attr = torch.ones(edges.shape[1], 1)                                           # TODO: remove or change to resonable attr if necessary

    atom_owners = batched_graph.atom_owners                                             # owners of batched atoms for feature indexing
    print('atom owners:', atom_owners)

    ################################################################################
    # Batch Cartesian coordinates, fractional coordinates, lattice, atomic numbers.
    ################################################################################
    # batched Cartesian coordnates with shape (sum(# atoms per structure), dim_x)
    x = batched_graph.atom_positions                                                    # Cartesian coordinates 
    x = torch.cat(x, dim = 0)                                                           # Batched atom positions
    # x = x[atom_owners]                                                                # !!!: Check is it necessary?
    print('x:', x)
    
    # batched fractional coordinates with shape (sum(# atoms per structure), xyz (3))
    f = [g.atom_frac_coord for g in crys_graph_list]  
    f = torch.cat(f, dim = 0)
    # f = f[atom_owners]                                                                # !!!: Check is it necessary?
    print('f:', f)                                                                      # f is the fractional coords in the batched format
    
    # batched lattice with shape (len(edge_index), abc (3), xyz (3))
    l = [g.lattice for g in crys_graph_list]                                            # lattice row corresponds to a lattice vector
    l = torch.stack(l, dim = 0)
    l = l[atom_owners]                                                                  # l is the batched lattice consitent with EGNN
    print('l:', l)
    
    # batched one hot atomic number with shape (sum(# atoms per structure), max(z))
    z = [g.atomic_number for g in crys_graph_list]
    z = torch.cat(z, dim=0) - 1                                                         # Shift atomic number starting from 0
    z_one_hot = F.one_hot(z, num_classes=NUM_SPECIES)
    print('one hot encoded z size:', z_one_hot.shape)

    ################################################################################
    # EGNN.
    ################################################################################
    # Initialize EGNN
    decoder = EGNNDecoder()
    h = z_one_hot.float()
    atom_owners = atom_owners.long()
    
    # Run EGNN
    # l, x = egnn(h, l, f, edges, edge_attr, atom_owners)
    # print(l, x)
    
    from e3nn import o3
    R = o3.rand_matrix(1)[0]
    
    out = egnn(h, l, f, edges, edge_attr, atom_owners)
    print('-'*30, 'BEFORE ROTATION', '-'*30)
    print('lattice:\n', torch.einsum('iax,xb->iab', out[0], R))
    print('fractional coordinates:\n', out[1])
    
    
    l = torch.einsum('iax,xb->iab', l,R)
    out = egnn(h, l, f, edges, edge_attr, atom_owners)
    print('-'*30, 'AFTER ROTATION', '-'*30)
    print('lattice:\n',  out[0])
    print('fractional coordinates:\n', out[1])