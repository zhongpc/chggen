"""Modules implementing egnn layer and model."""
from __future__ import annotations
from torch import nn
import torch
from typing import Tuple

class E_GCL(nn.Module):
    """E(n) equivariant convolutional layer.
    
    Args:
        input_nf (int): the number of features of input.
        output_nf (int): the number of features of output.
        hidden_nf (int): the number of features of hidden layer.
        edges_in_d (int): the number of features of edge attribute.
        ft_basis (int): the number of Fourier transform basis expansion
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
        self.epsilon = 1e-8             # Small number to avoid division by zero.
        self.ft_basis = ft_basis        # Number of FT basis to compute.
        edge_coords_nf = 3 * ft_basis   # Number of features in the edge coordinates.
        edge_lattice_nf = 9             # Number of features in the edge lattice.
        
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
        
        layer = nn.Linear(hidden_nf, 1, bias=False) # ??? Why no bias?
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001) # ??? Why gain is 0.001?
        
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)
        
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
            l_feat (Tensor): l_feat features 1D.
            ft (Tensor): FT bases expansion of relative fractional distance.
            edge_attr (Tensor): edge attribute.
        
        Returns:
            Tensor: next layer edge attribute.
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
        lattice: torch.Tensor,
    ) -> torch.Tensor:
        """Compute lattice feature L^TL reshaped to 1D from 3D lattice.
        
        Args:
            lattice (Tensor): lattice represented in 3D.
        
        Returns:
            Tensor: lattice feature L^TL reshaped to 1D.
        TODO: Adapt it with Niggas strategy.
        """
        
        l_feat = (l.T * l).reshape(9)
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
    
    def coord2ft(
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
        ft_feat = torch.cat([torch.sin(ft_feat), torch.cos(ft_feat)], dim = -1)
        ft_feat = ft_feat.reshape(-1, ft_basis * 3)
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
            l (Tensor): l represented in 3D.
            coord (Tensor): coordinates.
            edge_attr (Tensor): edge attribute.
            node_attr (Tensor): node attribute.
        
        Returns:
            Tensor: next layer node features.
            Tensor: next layer coordinates.
            Tensor: next layer edge features.
        """
        row, col = edge_index
        l_feat = self.lattice2feat(l)*torch.ones((len(coord)*3, 1))
        ft_feat = self.coord2ft(edge_index, coord, self.ft_basis)
        edge_feat = self.edge_model(h[row], h[col], l_feat, ft_feat, edge_attr)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        
        return h, coord, edge_attr
            
class EGNN(nn.Module):
    """Equivariant GNN implementation.
    
    Args:
        in_node_nf (int): the number of attribute for 'h' at the input.
        hidden_nf (int): the number of hidden features.
        out_node_nf (int): the number of features for 'h' at the output.
        in_edge_nf (int): the number of features for the edge attribute.
        ft_basis (int): the number of Fourier transform basis expansion coordination feature.
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
        self.device = device
        self.n_layers = n_layers
        self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    ft_basis=ft_basis,
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for EGNN model.
        
        Args:
            h (Tensor): node features.
            l (Tensor): l represented in 3D.
            x (Tensor): coordinates.
            edges (Tensor): edge indices.
            edge_attr (Tensor): edge features.
        """
        h = self.embedding_in(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](h, edges, l, x, edge_attr=edge_attr)
        h = self.embedding_out(h)
        return h, x
    
    

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

def get_edges(n_nodes):
    """Get edges for a given fully connected graph.
    
    Args:
        n_nodes (int): number of nodes.
    """
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    edges = [rows, cols]
    return edges

def get_edges_batch(
    n_nodes: int, batch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get edges for a given fully connected graph.
    
    Args:
        n_nodes (int): number of nodes.
        batch_size (int): batch size.
    """
    edges = get_edges(n_nodes)
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)
    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr
    elif batch_size > 1:
        rows, cols = [], []
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
        edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr



if __name__ == "__main__":
    # Dummy parameters
    batch_size = 8
    n_nodes = 4
    n_feat = 1
    x_dim = 3
    
    # Dummy variables h, x and fully connected edges
    h = torch.ones(batch_size * n_nodes, n_feat)
    x = torch.ones(batch_size * n_nodes, x_dim)
    l = torch.ones(3,3)
    edges, edge_attr = get_edges_batch(n_nodes, batch_size)
    
    # Initialize EGNN
    egnn = EGNN(
        in_node_nf=n_feat,
        hidden_nf=32,
        out_node_nf=1,
        in_edge_nf=1,
    )
    
    # Run EGNN
    h, x = egnn(h, l, x, edges, edge_attr)
    print(h.shape, x.shape)