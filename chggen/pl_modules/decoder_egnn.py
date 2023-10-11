"""EGNN Decoder for predicting noise for lattice and fractional coordinates."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from chggen.pl_modules.embeddings import MAX_ATOMIC_NUM
from chggen.pl_modules.embeddings.position_embedding import PositionEmbedding



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
        x: torch.Tensor,
        ft_basis: int,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute FT bases expansion of relative fractional distance.
        (f1, f2, f3) -> (cos(2pi*f1), sin(2pi*f1), cos(2pi*f2), sin(2pi*f2), 
        cos(2pi*f3), sin(2pi*f3)), ...
        
        Args:
            x (Tensor): coordinates.
            ft_basis (int): number of FT basis to compute.
            edge_index (Tensor): edge indices.
        """
        row, col = edge_index
        coord_diff = x[row] - x[col]
        ft_feat = torch.arange(2, ft_basis+2, 2, device=coord_diff.device) * torch.pi * coord_diff.unsqueeze(-1)
        ft_feat = torch.cat([torch.sin(ft_feat), torch.cos(ft_feat)], dim = -1) # [batch, x_dim, f_basis]
        ft_feat = ft_feat.reshape(-1, ft_basis * ft_feat.shape[1])
        return ft_feat 
    
    def forward(
        self,
        h: torch.Tensor,
        l: torch.Tensor,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: [torch.Tensor, None] = None,
        node_attr: [torch.Tensor, None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of one E_GCN layer.
        
        Args:
            h (Tensor): hidden node features.
            l (Tensor): lattice vectors with shape like (batched number of nodes, 
                lattice index (3), components (3)).
            x (Tensor): coordinates.
            edge_index (Tensor): edge indices.
            edge_attr (Tensor, optional): edge attribute.
            node_attr (Tensor, optional): node attribute.
        
        Returns:
            h (Tensor): next layer node features.
        """
        row, col = edge_index
        l_feat = self.lattice2feat(l)                                                   # lattice feature
        ft_feat = self.coord2strucfactor(x, ft_basis=self.ft_basis, edge_index=edge_index)    # structure factor feature
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
        atom_owners: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: [torch.Tensor, None] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for EGNN model.
        
        Args:
            h (Tensor): node features.
            l (Tensor): lattice vectors with shape like (batched number of nodes, 
                lattice index (3), components (3)).
            x (Tensor): coordinates.
            atom_owner (Tensor): owners of atoms showing which crystal structures 
                the atoms belongs to.
            edge_index (Tensor): edge indices.
            edge_attr (Tensor, optional): edge features.
        """
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h = self._modules["gcl_%d" % i](h, l, x, edge_index, edge_attr=edge_attr)
        
        agg = unsorted_segment_mean(h, atom_owners, num_segments=atom_owners.max().item()+1)
        l_weight = self.embedding_out_l_weight(agg).reshape(-1, self.x_dim, self.x_dim)
        l = l[get_reduced_id_from_tensor(atom_owners)]          # Reduce l to each structure one lattice.
        l = torch.einsum('iax,ixb->iab', l_weight, l)                        # [# structure, abc (3), xyz (3)]
        x = self.embedding_out_x(h)
        return l, x
    
    

class EGNNDecoder(nn.Module):
    """EGNN Decoder for predicting noise for lattice and fractional coordinates."""

    def __init__(self, num_noise_level: int) -> None:
        """Initialize EGNNDecoder with model parameters."""
        super(EGNNDecoder, self).__init__()
        self.egnn = EGNN(
            in_node_nf=MAX_ATOMIC_NUM + 64,    # h_z and h_p
            hidden_nf=32,
            in_edge_nf=0,
            ft_basis=10,
            x_dim=3,
        )
        self.num_noise_level = num_noise_level

    def forward(
        self, 
        sigma_step: int,
        atomic_numbers,
        noisy_lattices,
        noisy_frac_coords, 
        atom_owners,
        edge_index,
    ) -> torch.Tensor:
        """ Decode with diffusion model using EGNN framework.
        
        Args:
            sigma_step (int): step of sigma for the model. The range is [0, num_noise_level).
            atomic_numbers (Tensor): atomic number with shape like (N_atoms). 
                NOTE: atomic numbers starts from 1. If one hot coding is used, substraction of 1 is needed ahead.
            noisy_lattices (Tensor): noisy lattices before denoising with shape like (N_atoms, 3, 3).
            noisy_frac_coords (Tensor): noisy fractional coordinates with shape like (N_atoms, 3).
            atom_owners (Tensor): owners of atoms showing which crystal structures the atoms belong to, 
                with shape like (N_atoms).
            edge_index (Tensor): edge indices with shape like (2, N_edges). 
        
        Returns:
            lattice_score (Tensor): score of lattice with shape like (N_structure, 3, 3).
            frac_coords_score (Tensor): score of fractional coordiates with shape like (N_atoms, 3)
        
        # TODO: Check updating strategy in the chggen model. Whether output noise or denoised results.
        """
        # Atomic number embedding.
        h_z = F.one_hot(atomic_numbers-1, num_classes = MAX_ATOMIC_NUM).float()
        
        # Position embedding.
        pe = PositionEmbedding(max_position_len = self.num_noise_level, model_dim = 64)
        h_p = pe.to(h_z.device)(sigma_step)
        
        lattice_score, frac_coords_score = self.egnn(
            h = torch.cat((h_z, h_p), dim=-1), 
            l = noisy_lattices, 
            x = noisy_frac_coords, 
            atom_owners = atom_owners,
            edge_index = edge_index, 
        )
        return lattice_score, frac_coords_score