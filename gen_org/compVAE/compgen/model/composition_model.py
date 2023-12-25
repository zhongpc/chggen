"""Class for composition model."""

from __future__ import annotations
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max
from torch_geometric.data import Data

from ..utils import StandardScalerTorch, atom_types2compositions



class CompositionModel(nn.Module):
    """The composition model for composition sampling and average volume prediction."""
    def __init__(
        self,
        elem_emb_len: int,
        elem_fea_len: int = 64,
        n_graph=3,
        elem_heads=3,
        elem_gate=[256],
        elem_msg=[256],
        cry_heads=3,
        cry_gate=[256],
        cry_msg=[256],
        weight_pow = 1,
        activation = nn.SiLU,
        batchnorm = False,
        atom_num_dim: int = 103,                # Accommodate for Matscholar embedding.
        v_scaler: StandardScalerTorch = None,
        **kwargs,
    ) -> None:
        """Initialize the composition model by model hyperparameters."""
        # Store model args for reconstruction.
        self.model_args = {
            k:v for k,v in locals().items() if k not in ["self", "__class__", "kwargs"]
        }
        self.model_args.update(kwargs)
        
        super(CompositionModel, self).__init__()
        desc_dict = {
            "elem_emb_len": elem_emb_len,
            "elem_fea_len": elem_fea_len,
            "n_graph": n_graph,
            "elem_heads": elem_heads,
            "elem_gate": elem_gate,
            "elem_msg": elem_msg,
            "cry_heads": cry_heads,
            "cry_gate": cry_gate,
            "cry_msg": cry_msg,
            "weight_pow": weight_pow,
            "activation": activation,
            "batchnorm": batchnorm,
        }
        self.material_nn = DescriptorNetwork(**desc_dict)
        
        self.encoder_mu = build_mlp(
            input_dim = elem_fea_len,
            hidden_dim = 32,
            output_dim=elem_fea_len,
            activation=activation,
            batchnorm=batchnorm,
        )
        
        self.encoder_logvar = build_mlp(
            input_dim = elem_fea_len,
            hidden_dim = 32,
            output_dim=elem_fea_len,
            activation=activation,
            batchnorm=batchnorm,
        )
        
        self.mlp_comp = build_mlp(
            input_dim = elem_fea_len,
            hidden_dim = 32,
            output_dim=atom_num_dim,
            activation=activation,
            batchnorm=batchnorm,
        )
        
        self.decoder_comp = nn.Sequential(
            self.mlp_comp,
            nn.Softmax(dim=-1),
        )
        
        self.decoder_ave_v = build_mlp(
            input_dim = elem_fea_len,
            hidden_dim = 32,
            output_dim=1,
            activation=activation,
            batchnorm=batchnorm,
        )
        
    def forward(
        self, data: Data,
    ) -> Any:
        """Forward pass of the composition model."""
        cry_fea, _ = self.material_nn(
            elem_weights = data.elem_weights, 
            elem_fea = data.x, 
            self_fea_idx = data.edge_index[0], 
            nbr_fea_idx = data.edge_index[1], 
            cry_elem_idx = data.batch,
        )
        mu = self.encoder_mu(cry_fea)
        logvar = self.encoder_logvar(cry_fea)
        z = self.reparameterize(mu, logvar)
        comp = self.decoder_comp(z)
        ave_v = self.decoder_ave_v(cry_fea)
        
        return mu, logvar, comp, ave_v
    
    def encode(
        self, data: Data,
    ) -> Any:
        """Encode the data with composition model."""
        
        self.eval()
        with torch.no_grad():
            cry_fea, _ = self.material_nn(
                elem_weights = data.elem_weights, 
                elem_fea = data.x, 
                self_fea_idx = data.edge_index[0], 
                nbr_fea_idx = data.edge_index[1], 
                cry_elem_idx = data.batch,
            )
            mu = self.encoder_mu(cry_fea)
            logvar = self.encoder_logvar(cry_fea)
        return mu, logvar
    
    def sample_comp(
        self, num_samples: int, num_atoms: torch.Tensor | int,
    ) -> Any:
        """Sample from the composition model."""
        # Sample letent vectors.
        z = torch.randn(num_samples, self.model_args["elem_fea_len"])
        
        # Decode to composition.
        self.eval()
        with torch.no_grad():
            comp = self.decoder_comp(z)
        if isinstance(num_atoms, int):
            num_atoms = torch.tensor([num_atoms] * num_samples)
        else:
            pass
        
        # Convert normalized composition to atom types.
        atom_types = self.sample_composition(comp, num_atoms)
        
        # Convert atom types to formulas.
        comps = atom_types2compositions(atom_types)
        formulas = [comp.reduced_formula for comp in comps]
        return comp, atom_types, formulas
        
    def predict_ave_v(
        self, data: Data, 
    ) -> Any:
        """Predict the average volume per atom of the material."""
        if self.v_scaler is None:
            raise ValueError("v_scaler is not provided.")
        
        self.eval()
        with torch.no_grad():
            cry_fea, _ = self.material_nn(
                elem_weights = data.elem_weights, 
                elem_fea = data.x, 
                self_fea_idx = data.edge_index[0], 
                nbr_fea_idx = data.edge_index[1], 
                cry_elem_idx = data.batch,
            )
            ave_v = self.decoder_ave_v(cry_fea)
        ave_v = self.v_scaler.inverse_transform(ave_v)
        return ave_v
    
    @staticmethod
    def sample_composition(composition_prob, num_atoms):
        """Sample composition such that it exactly satisfies composition_prob.
        
        Args:
            composition_prob (Tensor): The composition probability. The shape should be
                (num of structures, MAX_ATOMIC_NUM).
            num_atoms (Tensor): The number of atoms for each structure. The shape should
                be (num of structures).
        
        Returns:
            all_sampled_comp (Tensor): The sampled composition. The shape should be
                (num of atoms).
        """
        all_sampled_comp = []

        for comp_prob, num_atom in zip(list(composition_prob), list(num_atoms)):
            comp_num = torch.round(comp_prob * num_atom)
            atom_type = torch.nonzero(comp_num, as_tuple=True)[0] + 1
            atom_num = comp_num[atom_type - 1].long()

            sampled_comp = atom_type.repeat_interleave(atom_num, dim=0)

            # if the rounded composition gives less atoms, sample the rest
            if sampled_comp.size(0) < num_atom:
                left_atom_num = num_atom - sampled_comp.size(0)

                left_comp_prob = comp_prob - comp_num.float() / num_atom

                left_comp_prob[left_comp_prob < 0.] = 0.
                left_comp = torch.multinomial(
                    left_comp_prob, num_samples=left_atom_num, replacement=True)
                # convert to atomic number
                left_comp = left_comp + 1
                sampled_comp = torch.cat([sampled_comp, left_comp], dim=0)

            sampled_comp = sampled_comp[torch.randperm(sampled_comp.size(0))]
            sampled_comp = sampled_comp[:num_atom]
            all_sampled_comp.append(sampled_comp)
            
        # all_sampled_comp = torch.cat(all_sampled_comp, dim=0)
        # assert all_sampled_comp.size(0) == num_atoms.sum()
        return all_sampled_comp
    
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(logvar / 2)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def as_dict(self) -> dict:
        """Return the Unet1D weights and args in a dictionary."""
        return {
            "state_dict": self.state_dict(),
            "model_args": self.model_args,
        }
    
    @classmethod
    def from_dict(cls, dic: dict, **kwargs) -> CompositionModel:
        """Build a CompositionModel from a saved dictionary."""
        composition_model = CompositionModel(**dic["model_args"])
        composition_model.load_state_dict(dic["state_dict"], **kwargs)
        return composition_model
    
    @classmethod
    def from_file(cls, path: str, **kwargs) -> CompositionModel:
        """Build a CompositionModel from a saved file."""
        state = torch.load(path, map_location=torch.device('cpu'))
        return CompositionModel.from_dict(state["model"], **kwargs)
    
    def __repr__(self):
        return self.__class__.__name__

class DescriptorNetwork(nn.Module):
    """The Descriptor Network is the message passing section of the Roost Model.
    
    Args:
        elem_emb_len (int): The dimension of the elemental embedding.
        elem_fea_len (int): The dimension of the elemental feature.
        n_graph (int): The number of message passing layers.
        elem_heads (int): The number of attention heads for the elemental message passing.
        elem_gate (int): The dimension of hidden layer for MLP for gate in message passing.
        elem_msg (int): The dimension of hidden layer for MLP for message in message passing.
        cry_heads (int): The number of attention heads for the crystal global pooling.
        cry_gate (int): The dimension of hidden layer for MLP for gate in crystal global pooling.
        cry_msg (int): The dimension of hidden layer for MLP for message in crystal global pooling.
        weight_pow (int): The power of weights in the attention mechanism.
        activation (nn.Module): The activation function.
        batchnorm (bool): Whether to use batch normalization.
    """
    def __init__(
        self,
        elem_emb_len: int,
        elem_fea_len: int = 64,
        n_graph: int = 3,
        elem_heads: int = 3,
        elem_gate: list = [256],
        elem_msg: list = [256],
        cry_heads: int = 3,
        cry_gate: list = [256],
        cry_msg: list = [256],
        weight_pow: int = 1,
        activation: nn.Module = nn.SiLU,
        batchnorm: bool = False
    ):
        super().__init__()

        self.batchnorm = batchnorm
        self.activation = activation

        # apply linear transform to the input to get a trainable embedding
        # NOTE -1 here so we can add the weights as a node feature
        self.embedding = nn.Linear(elem_emb_len, elem_fea_len - 1)

        # create a list of Message passing layers
        self.graphs = nn.ModuleList(
            [
                MessageLayer(
                    elem_fea_len=elem_fea_len,
                    elem_heads=elem_heads,
                    elem_gate=elem_gate,
                    elem_msg=elem_msg,
                    weight_pow= weight_pow,
                    activation = self.activation,
                    batchnorm = self.batchnorm,
                )
                for i in range(n_graph)
            ]
        )

        # define a global pooling function for materials
        self.cry_pool = nn.ModuleList(
            [
                WeightedAttentionPooling(
                    gate_nn=SimpleNetwork(elem_fea_len, 1, cry_gate, activation= self.activation),
                    message_nn=SimpleNetwork(elem_fea_len, elem_fea_len, cry_msg, activation= self.activation,
                                             batchnorm= self.batchnorm),
                    weight_pow = weight_pow,

                )
                for _ in range(cry_heads)
            ]
        )

    def forward(self, elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N)
            Fractional weight of each Element in its stoichiometry
        elem_fea: Variable(torch.Tensor) shape (N, orig_elem_fea_len)
            Element features of each of the N elems in the batch
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs
        cry_elem_idx: list of torch.LongTensor of length C
            Mapping from the elem idx to crystal idx

        Returns
        -------
        cry_fea: nn.Variable shape (C,)
            Material representation after message passing
        """

        # embed the original features into a trainable embedding space
        elem_fea = self.embedding(elem_fea)

        # add weights as a node feature
        elem_fea = torch.cat([elem_fea, elem_weights], dim=1)

        # apply the message passing functions
        for graph_func in self.graphs:
            elem_fea = graph_func(elem_weights, elem_fea, self_fea_idx, nbr_fea_idx)

        # generate crystal features by pooling the elemental features
        head_fea = []
        for attnhead in self.cry_pool:
            head_fea.append(
                attnhead(elem_fea, index=cry_elem_idx, weights=elem_weights)
            )

        ## return the head-averaged pooling and the elem_fea_matrix
        return torch.mean(torch.stack(head_fea), dim=0), elem_fea

    def __repr__(self):
        return self.__class__.__name__


class MessageLayer(nn.Module):
    """Massage Layers are used to propagate information between nodes in
    the stoichiometry graph."""
    def __init__(self, elem_fea_len, elem_heads, elem_gate, elem_msg, weight_pow,
                 activation = nn.LeakyReLU, batchnorm = False):
        super().__init__()

        self.activation = activation
        self.batchnorm = batchnorm

        # Pooling and Output
        self.pooling = nn.ModuleList(
            [
                WeightedAttentionPooling(
                    gate_nn=SimpleNetwork(2 * elem_fea_len, 1, elem_gate, activation = self.activation),
                    message_nn=SimpleNetwork(2 * elem_fea_len, elem_fea_len, elem_msg, activation= self.activation,
                                             batchnorm= self.batchnorm),
                    weight_pow = weight_pow,
                )
                for _ in range(elem_heads)
            ]
        )

    def forward(self, elem_weights, elem_in_fea, self_fea_idx, nbr_fea_idx):
        """
        Forward pass

        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch

        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N,)
            The fractional weights of elems in their materials
        elem_in_fea: Variable(torch.Tensor) shape (N, elem_fea_len)
            Element hidden features before message passing
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs

        Returns
        -------
        elem_out_fea: nn.Variable shape (N, elem_fea_len)
            Element hidden features after message passing
        """
        # construct the total features for passing
        elem_nbr_weights = elem_weights[nbr_fea_idx, :]
        elem_nbr_fea = elem_in_fea[nbr_fea_idx, :]
        elem_self_fea = elem_in_fea[self_fea_idx, :]
        fea = torch.cat([elem_self_fea, elem_nbr_fea], dim=1)

        # sum selectivity over the neighbours to get elems
        head_fea = []
        for attnhead in self.pooling:
            head_fea.append(
                attnhead(fea, index=self_fea_idx, weights=elem_nbr_weights)
            )

        # average the attention heads
        fea = torch.mean(torch.stack(head_fea), dim=0)

        return fea + elem_in_fea

    def __repr__(self):
        return self.__class__.__name__
    
class WeightedAttentionPooling(nn.Module):
    """Weighted softmax attention layer"""
    def __init__(self, gate_nn, message_nn, weight_pow = 1):
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn

        self.pow = weight_pow # torch.nn.Parameter(torch.randn(1))

    def forward(self, x, index, weights):
        gate = self.gate_nn(x)
        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = (weights ** self.pow) * gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        x = self.message_nn(x)
        out = scatter_add(gate * x, index, dim=0)

        return out

    def __repr__(self):
        return self.__class__.__name__
    

class SimpleNetwork(nn.Module):
    """
    Simple Feed Forward Neural Network
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layer_dims,
        activation=nn.LeakyReLU,
        batchnorm=False,
    ):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)

        """
        super().__init__()

        dims = [input_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList(
                [nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 1)]
            )
        else:
            self.bns = nn.ModuleList([nn.Identity() for i in range(len(dims) - 1)])

        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, bn, act in zip(self.fcs, self.bns, self.acts):
            x = act(bn(fc(x)))

        return self.fc_out(x)

    def __repr__(self):
        return self.__class__.__name__

    def reset_parameters(self):
        for fc in self.fcs:
            fc.reset_parameters()
        self.fc_out.reset_parameters()
        
    

class build_mlp(nn.Module):
    """Use simple network to forward information."""
    def __init__(
        self,
        input_dim = 64,
        hidden_dim = 32,
        output_dim = 32,
        activation=nn.Softplus,
        batchnorm=False,
    ) -> None:
        """Initialize MLP with model size."""
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        if batchnorm:
            self.bn = nn.BatchNorm1d(hidden_dim)
        else:
            self.bn = nn.Identity()

        self.act = activation()

    def forward(self, x):
        out = self.fc(self.act(self.bn( self.hidden(x) )))
        return out

    def __repr__(self):
        return self.__class__.__name__