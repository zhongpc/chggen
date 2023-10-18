from typing import Any, Dict, Tuple, Optional, List, Sequence
from tqdm import tqdm
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch_scatter import scatter
import pytorch_lightning as pl
from torch_geometric.data import Batch

from pymatgen.core import Structure, Lattice, Element

from chgnet.graph import CrystalGraph

from chggen.chgnet.chgnet.model.model import BatchedGraph
from chggen.common.data_utils import (
    EPSILON, 
    mard, 
    lengths_angles_to_volume,
    )
from chggen.common.operations import PModulo
from chggen.pl_modules.embeddings import MAX_ATOMIC_NUM
from chggen.pl_modules.encoder import CHGNet_encoder
from chggen.pl_modules.decoder_egnn import EGNNDecoder
from chggen.pl_modules.condition import Classifier
from chggen.pl_modules.wrapped_normal_distribution import wND
from chggen.common.data_utils import get_exp_sigmas



def build_mlp(
    in_dim: int, 
    hidden_dim: int, 
    fc_num_layers: int, 
    out_dim: int, 
    use_layernorm: bool = False,
    activation: nn.Module = nn.SiLU(),
) -> nn.Module:
    """Build multilayer perceptron (MLP) with activation function."""
    if use_layernorm:
        mods = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), activation]
    else:
        mods = [nn.Linear(in_dim, hidden_dim), activation]

    for i in range(fc_num_layers-1):
        if use_layernorm:
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), activation]
        else:
            mods += [nn.Linear(hidden_dim, hidden_dim), activation]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)



class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
    
    ## TODO: generalize the optimizer 
    def configure_optimizers(self, lr = 1e-3, use_lr_scheduler = True):
        opt = torch.optim.Adam(self.parameters(), lr= lr)
        if use_lr_scheduler:
            return [opt]
        scheduler = ExponentialLR(opt, gamma=0.95)
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}



class CHGGen(BaseModule):
    def __init__(
        self, 
        hparams_dict: dict = {'latent_dim': 256, 'hidden_dim': 128, 'property_dim': 1, 'load_pretrain': True, 'fc_num_layers': 2, 
                        'sigma_F_begin': 1.0, 'sigma_F_end': 0.01, 
                        'sigma_L_begin': 1.0, 'sigma_Lend': 0.01, 
                        'type_sigma_begin': 50.0, 'type_sigma_end': 0.01,
                        'max_atoms': 20, 'predict_property': False, 'num_noise_level': 50, 
                        'lattice_scale_method': 'scale_length', 
                        'cost_natom': 1.0, 'cost_latt': 10.0, 'cost_coord': 10.0, 'cost_type': 1.0, 'cost_lattice': 10.0, 'cost_composition': 1.0, 'cost_edge': 10.0, 'cost_property': 1.0,
                        'beta': 0.01,
                        'teacher_forcing_lattice': True,
                        'teacher_forcing_max_epoch': 1000,
                        'decoder': 'gemnet'},
        lattice_scaler: Any = None,
        frac_score_norms: Any = None,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.save_hyperparameters(hparams_dict)
        
        # Initialize MSE loss for composition.
        self.mse_composition = nn.MSELoss(reduction="none")
        
        # Initialize modulo operation and wrapped normal distribution.
        self.pmodulo = PModulo()
        self.wnd = wND(n=10)
        
        # Initialize encoders and decoder.
        if self.hparams.load_pretrain:              # Pretrained 3dencoder and no training.
            self.encoder3d = CHGNet_encoder().load()
            self.encoder3d.return_crys_feature = True
            for param in self.encoder3d.parameters():
                param.requires_grad = False
        else:                                       # No pretrained 3dencoder and training.
            self.encoder3d = CHGNet_encoder()
            self.encoder3d.return_crys_feature = True
            for param in self.encoder3d.parameters():
                param.requires_grad = True

        # TODO: Add the 1D encoder to encode composition into latents.
        self.encoder1d = None
                
        self.decoder = EGNNDecoder(num_noise_level=self.hparams.num_noise_level)

        # Initialize MLPs for decoding key states.
        self.fc_mu = nn.Linear(
            self.hparams.latent_dim,
            self.hparams.latent_dim,
        )
        self.fc_var = nn.Linear(
            self.hparams.latent_dim,
            self.hparams.latent_dim,
        )
        self.fc_latent_proj = build_mlp(
            64, 
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers, 
            self.hparams.latent_dim, 
            use_layernorm= True,
        )
        self.fc_num_atoms = build_mlp(
            self.hparams.latent_dim, 
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers, 
            self.hparams.max_atoms+1,
        )
        self.fc_lattice = build_mlp(
            self.hparams.latent_dim, 
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers, 
            6,                                      # Niggli reduction returns 3 lengths, 3 angles
        )
        self.fc_composition = build_mlp(
            self.hparams.latent_dim, 
            self.hparams.hidden_dim,
            self.hparams.fc_num_layers, 
            MAX_ATOMIC_NUM,
            activation=nn.ReLU(),                   # Gradient disappears for SiLU.
        )
        if self.hparams.predict_property:
            self.fc_property = build_mlp(
                self.hparams.latent_dim, 
                self.hparams.hidden_dim,
                self.hparams.fc_num_layers, 
                self.hparams.property_dim,
            )
        
        # Initialize the noise scheme sigmas for diffusion.
        self.sigmas_F = get_exp_sigmas(
            sigma_begin = self.hparams.sigma_F_begin,
            sigma_end = self.hparams.sigma_F_end,
            num_noise_level = self.hparams.num_noise_level,
        )
        
        self.sigmas_L = get_exp_sigmas(
            sigma_begin = self.hparams.sigma_L_begin,
            sigma_end = self.hparams.sigma_L_end,
            num_noise_level = self.hparams.num_noise_level,
        )

        self.type_sigmas = get_exp_sigmas(
            sigma_begin = self.hparams.type_sigma_begin,
            sigma_end = self.hparams.type_sigma_end,
            num_noise_level = self.hparams.num_noise_level,
        )

        # Initialize lattice scaler and normalize factor for frac score.
        self.lattice_scaler = lattice_scaler
        self.frac_score_norms = torch.tensor(frac_score_norms(self.sigmas_F.numpy()))

    # Training, validation and testing functions.
    def training_step(
        self, batch: Batch, batch_idx: int,     # Batch_idx is necessary for Pytorch Lightning.
    ) -> torch.Tensor:
        teacher_forcing = (
            self.current_epoch <= self.hparams.teacher_forcing_max_epoch)
        outputs = self(batch, teacher_forcing)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(
        self, batch: Batch, batch_idx: int,
    ) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(
        self, batch: Batch, batch_idx: int,
    ) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
        self.log_dict(
            log_dict,
        )
        return loss
    
    # =================================================================
    # Train the material generator.
    # =================================================================
    def forward(
        self, batch: Batch, teacher_forcing: bool
    ):
        """Forward pass for training and validation.
        
        Args:
            batch (Batch): batched graph in pytorch_geometric.data.
            teacher_forcing (bool): whether to use teacher forcing. If teacher_forcing is True, 
                the ground truth output from the previous time step is used as input to the current 
                time step. If teacher_forcing is False, the predicted output from the previous time 
                step is used as input to the current time step. 
        """
        
        graphs = [g.to(self.device) for g in batch.crys_graph]
        
        mu, log_var, z = self.encode(graphs)
        (pred_num_atoms,                        # Predicted number of atoms
        pred_lengths_and_angles,                # Predicted Niggli reduced and scaled lengths and angles.  
        pred_lengths,                           # Predicted Niggli unreduced and unscaled recovered lengths.
        pred_angles,                            # Same as above.
        pred_composition) = self.decode_stats(
            z, batch.num_atoms, batch.lengths, batch.angles, teacher_forcing,
        )

        # Sample noise levels.
        sigma_step = torch.randint(             # Choose the same sigma step for both lattice and frac_coords denoise.
            0, self.hparams.num_noise_level, (batch.num_atoms.size(0),))
        noise_level_F = self.sigmas_F[sigma_step].to(self.device)
        noise_level_L = self.sigmas_L[sigma_step].to(self.device)
        type_noise_level = self.type_sigmas[sigma_step].to(self.device)
        
        sigma_step_per_atom = sigma_step.to(self.device).repeat_interleave(batch.num_atoms, dim=0)
        sigmas_F_per_atom = noise_level_F.repeat_interleave(batch.num_atoms, dim=0)
        sigmas_L_per_structure = noise_level_L
        type_sigmas_per_atom = type_noise_level.repeat_interleave(batch.num_atoms, dim=0)

        # Add noise to atom types and sample atom types.
        # atom_type_probs = (
        #     F.one_hot(batch.atom_types - 1, num_classes=MAX_ATOMIC_NUM) + 
        #     pred_composition_per_atom * type_sigmas_per_atom[:, None])
        # rand_atom_types = torch.multinomial(input=atom_type_probs, num_samples=1).squeeze(1) + 1
        
        # Add noise to fractional coordinates and Niggli reduced but no scaled lattices.
        # This is for changing the extensive parameters to intensive parameters.
        normal_frac_noises = torch.randn_like(batch.frac_coords)
        normal_latt_noises = torch.randn_like(batch.reduced_lattices)
        frac_noises_per_atom = normal_frac_noises * sigmas_F_per_atom[:, None]
        latt_noises_per_structure = normal_latt_noises * sigmas_L_per_structure[:, None]  # (num_structures, 9)
        
        frac_coords = batch.frac_coords
        lattices = batch.reduced_lattices

        noisy_frac_coords =  self.pmodulo.to(self.device)(frac_coords + frac_noises_per_atom)   # (num_atoms, 3)
        noisy_latt = lattices + latt_noises_per_structure                                       # (num_structures, 9) 
        
        pred_latt_score, pred_frac_coord_score = self.decoder(
            sigma_step = sigma_step_per_atom,
            atomic_numbers = batch.atom_types.to(self.device),
            noisy_lattices = noisy_latt.repeat_interleave(batch.num_atoms, dim=0).reshape(-1,3,3).to(self.device),  # Accormmadate EGNN structure
            noisy_frac_coords = noisy_frac_coords, 
            atom_owners = batch.batch.to(self.device),
            edge_index = batch.edge_index.to(self.device),
        )
        pred_atom_types = None
        # TODO: Implmenet type prediction
        
        # Compute target scores.
        target_latt_score = - normal_latt_noises
        target_frac_coord_score = self.wnd.to(self.device).log_grad(
            noisy_frac_coords, 
            mu=batch.frac_coords, 
            sigma=sigmas_F_per_atom[:,None].repeat_interleave(3, dim=-1),
        )
        # Normalize factor for normalized output.
        print(target_frac_coord_score.shape)
        target_frac_coord_score /= self.frac_score_norms.to(sigma_step_per_atom.device)[sigma_step_per_atom][:,None]
        
        # Compute reconstruction loss and denoising loss.
        num_atom_loss = self.num_atom_loss(pred_num_atoms, batch)
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, batch)
        composition_loss = self.composition_loss(pred_composition, batch)
        latt_loss = self.latt_loss(pred_latt_score.reshape(-1,9), target_latt_score)
        coord_loss = self.coord_loss(pred_frac_coord_score, target_frac_coord_score, batch)
        # type_loss = self.type_loss(pred_atom_types, batch.atom_types,
        #                            used_type_sigmas_per_atom, batch)

        kld_loss = self.kld_loss(mu, log_var)

        if self.hparams.predict_property:
            property_loss = self.property_loss(z, batch)
        else:
            property_loss = 0.

        return {
            'num_atom_loss': num_atom_loss,
            'lattice_loss': lattice_loss,
            'composition_loss': composition_loss,
            'latt_loss': latt_loss,
            'coord_loss': coord_loss,
            # 'type_loss': type_loss,
            'kld_loss': kld_loss,
            'property_loss': property_loss,
            'pred_num_atoms': pred_num_atoms,
            'pred_lengths_and_angles': pred_lengths_and_angles,
            'pred_lengths': pred_lengths,
            'pred_angles': pred_angles,
            'pred_cart_coord_diff': pred_frac_coord_score,
            'pred_atom_types': pred_atom_types,
            'pred_composition': pred_composition,
            'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,
            'rand_frac_coords': noisy_frac_coords,
            # 'rand_atom_types': rand_atom_types,
            'z': z,
        }
    
    def encode(self, graphs: Sequence[CrystalGraph]) -> Tuple[torch.Tensor]:
        """Encode crystal structures to latents by pre-trained CHGNet.
        
        Args:
            graphs (sequence): batched graph in pytorch_geometric.data.
        
        Returns:
            mu (Tensor): mean of the latent Gaussian with shape (batch_size, latent_dim).
            log_var (Tensor): standard deviation of the latent Gaussian with shape (batch_size,
                latent_dim).
            z (Tensor): reparameterized latent Gaussian with shape (batch_size, latent_dim).
        """
        prediction = self.encoder3d(graphs) # prediction returned from CHGNet
        crystal_fea = prediction['crystal_fea']
        latent_fea = self.fc_latent_proj(crystal_fea)

        mu = self.fc_mu(latent_fea)
        log_var = self.fc_var(latent_fea)

        z = self.reparameterize(mu, log_var)
        return mu, log_var, z
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick to sample from N(mu, var) from N(0,1).
        
        Args:
            mu (Tensor): Mean of the latent Gaussian with shape (batch_size, latent_dim).
            logvar (Tensor): Standard deviation of the latent Gaussian with shape (batch_size, 
                latent_dim).
        
        Returns:
            out (Tensor): Reparameterized latent Gaussian with shape (batch_size, latent_dim).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    # def generate_rand_init(self, pred_composition, pred_lengths,
    #                        pred_angles, num_atoms, batch):
    #     rand_frac_coords = torch.rand(num_atoms.sum(), 3,
    #                                   device=num_atoms.device)
    #     pred_composition = F.softmax(pred_composition,
    #                                           dim=-1)
    #     rand_atom_types = self.sample_composition(
    #         pred_composition, num_atoms)
    #     return rand_frac_coords, rand_atom_types

    # =================================================================
    # Sample new materials.
    # =================================================================
    def sample(self, num_samples, ld_kwargs):
        z = torch.randn(num_samples, self.hparams.hidden_dim,
                        device=self.device)
        samples = self.langevin_dynamics_guidance(z, ld_kwargs)
        return samples
    
    def langevin_dynamics_guidance(
        self, 
        z: torch.Tensor, 
        prop_guidance: float, 
        ld_kwargs: SimpleNamespace, 
        gt_num_atoms: Optional[torch.Tensor] = None, 
        gt_atom_types: Optional[torch.Tensor] = None,
    ) -> dict:
        """Decode crystral structure from latent embeddings with property guidance.
        
        Args:
            z (Tensor): latent embeddings. The shape should be (batch_size, latent_dim).
            prop_guidance (float): the target property value. It should be a scalar.
            ld_kwargs (Simplenamespace): args for doing annealed langevin dynamics sampling:
                n_step_each (int): number of steps for each sigma level.
                step_lr (float): epsilon parameter for diffusion.
                min_sigma (float): minimum sigma to use in annealed langevin dynamics.
                save_traj (bool): save the entire LD trajectory.
                disable_bar (bool): disable the progress bar of langevin dynamics.
                compute_force (bool): compute force during langevin dynamics.
                beta_c (float): weight for property.        
                beta_f (float): weight for atomic force.   
            gt_num_atoms (Tensor, optional): if not None, use the ground truth number of atoms.     
            gt_atom_types (Tensor, optional): if not None, use the ground truth atom types.         
        
        Mathematics:
            x_t = x_{t-1} + epsilon/2 * grad_x log p(x_{t-1}) + sqrt(epsilon) * N(0, I) (No peroperty guidance)
            In annealed Langevin Dynamics, we have:
                epsilon = step_lr * (sigma_i / sigma_L)^2, sigma_i is exponentially decayed to sigma_L.
            
            The sorce function term 'grad_x log p(x_{t-1})' can be approximated by score network estimation.
            grad_x log p(x_t) ~ s_t(R) + beta_c * grad_R log q(c|R) - beta_f * grad_R E(R)

        """
        if ld_kwargs.save_traj:
            all_frac_coords = []
            all_pred_cart_coord_diff = []
            all_noise_cart = []
            all_atom_types = []

        # Initialize classfier.
        # classifier = Classifier(prop_given= prop_guidance)

        # Obtain key states.
        num_atoms, _, lengths, angles, composition = self.decode_stats(
            z, gt_num_atoms, 
        )

        # Obtain atom types.
        if gt_atom_types is None:
            cur_atom_types = self.sample_composition(composition, num_atoms)
        else:
            cur_atom_types = gt_atom_types

        # Initialize coordinates and lattices.
        cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=z.device, requires_grad = False)
        cur_lattices = self.get_lattices(lengths, angles, num_atoms)
        cur_lattices /= ((num_atoms)**(1/3))[:, None, None]   # Niggli reduction.
        
        # Annealed langevin dynamics.
        with torch.no_grad():
            for sigma_step in tqdm(range(self.hparams.num_noise_level), total=self.hparams.num_noise_level, disable=ld_kwargs.disable_bar):
                sigma_F = self.sigmas_F[sigma_step]
                sigma_L = self.sigmas_L[sigma_step]
                if sigma_F < ld_kwargs.min_sigma:           # Stop if sigma is too small.
                    break
                step_size_F = ld_kwargs.step_lr * (sigma_F / self.sigmas_F[-1]) ** 2
                step_size_L = ld_kwargs.step_lr * (sigma_L / self.sigmas_L[-1]) ** 2

                for step in range(ld_kwargs.n_step_each):
                    noise_frac = torch.randn_like(cur_frac_coords) * torch.sqrt(step_size_F * 2)    
                    noise_latt = torch.randn_like(cur_lattices) * torch.sqrt(step_size_L * 2)      
                    
                    structure_list = self.get_pymatgen_structure_from_lattice(
                        cur_lattices, num_atoms, cur_frac_coords, cur_atom_types,
                    )
                    
                    # for s_gen in structure_list:
                    #     print(s_gen.composition.reduced_formula)

                    Z, l, f, edge_index, atom_owners = self.batched_structure_stat(structure_list, device=z.device)
                    sigma_step_per_atom = torch.tensor([sigma_step], device=z.device).repeat_interleave(len(Z), dim=0)
                    
                    # Forward to decoder for denoising, output is the score of lattices and frac_coords
                    pred_latt_score, pred_frac_coord_diff = self.decoder(
                        sigma_step = sigma_step_per_atom,
                        atomic_numbers = Z,
                        noisy_lattices = l,
                        noisy_frac_coords = f,   
                        atom_owners = atom_owners,
                        edge_index = edge_index,
                    )
                
                    pred_lattices_diff = pred_latt_score / sigma_L          
                    pred_frac_coords_diff = pred_frac_coord_diff * self.frac_score_norms.to(sigma_step)(sigma_step)

                    cur_frac_coords = self.pmodulo.to(self.device)(cur_frac_coords + step_size_F * pred_frac_coords_diff + noise_frac)
                    cur_lattices = cur_lattices + step_size_L * pred_lattices_diff + noise_latt

                # #### need to update the lattice and frac ####
                # batch_c_grad, batch_forces = self.compute_grad(
                #     batched_graph=batched_graph,
                #     classifier=classifier, 
                #     ld_kwargs=ld_kwargs,
                # )

                # ## TODO: double check the scale and sign !

                # prop_disp = ld_kwargs.beta_c * batch_c_grad
                # force_disp = ld_kwargs.beta_f * batch_forces
                
                # # Clamp to the maximum displacement 0.5
                # prop_disp = torch.clamp(prop_disp, min = -0.5, max = 0.5)
                # force_disp = torch.clamp(force_disp, min = -0.5, max = 0.5)
        
        # Recover Niggli reduction.
        cur_lattices *= ((num_atoms)**(1/3))[:, None, None] 
        
        output_dict = {'num_atoms': num_atoms, 
                       'frac_coords': cur_frac_coords, 
                       'lattices': cur_lattices,
                       'atom_types': cur_atom_types,
                       'is_traj': False}

        if ld_kwargs.save_traj:
            output_dict.update(dict(
                all_frac_coords=torch.stack(all_frac_coords, dim=0),
                all_atom_types=torch.stack(all_atom_types, dim=0),
                all_pred_cart_coord_diff=torch.stack(
                    all_pred_cart_coord_diff, dim=0),
                all_noise_cart=torch.stack(all_noise_cart, dim=0),
                is_traj=True))

        return output_dict
    
    def decode_stats(
        self, 
        z: torch.Tensor, 
        gt_num_atoms = None, 
        gt_lengths = None, 
        gt_angles = None,
        teacher_forcing: bool = False,
    ) -> Tuple[torch.Tensor]:
        """Decode key states from latent embeddings.
        
        Args:
            z (Tensor): latent embeddings. The shape should be (batch_size, latent_dim).
            gt_num_atoms (Tensor): if not None, use the ground truth number of atoms.
            gt_lengths (Tensor): if not None, use the ground truth lengths.
            gt_angles (Tensor): if not None, use the ground truth angles.
            teacher_forcing (bool): whether to use teacher forcing.
        
        Returns:
            num_atoms (Tensor): number of atoms for each structures. The shape should 
                be (num of atoms).
            pred_lengths_and_angles (Tensor): predicted lengths and angles. It is Niggli 
                reduced and standardized (sacled). The shape should be (num of atoms, 6).
            lengths (Tensor): lattice lengths. It is Niggli unreduced and unscaled. The 
                shape should be (num of atoms, 3).
            angles (Tensor): lattice angles. Information is the same as above.
            composition (Tensor): composition of each structure. The shape should be
                (num of structures, MAX_ATOMIC_NUM).
        """
        
        if gt_num_atoms is not None:
            num_atoms = self.predict_num_atoms(z)
            pred_lengths_and_angles, lengths, angles = self.predict_lattice(z, gt_num_atoms)
            composition = self.predict_composition(z)
            if self.hparams.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
        else:
            num_atoms = self.predict_num_atoms(z).argmax(dim=-1)
            pred_lengths_and_angles, lengths, angles = self.predict_lattice(z, num_atoms)
            composition = self.predict_composition(z)
        return num_atoms, pred_lengths_and_angles, lengths, angles, composition

    def sample_composition(self, composition_prob, num_atoms):
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

        all_sampled_comp = torch.cat(all_sampled_comp, dim=0)
        assert all_sampled_comp.size(0) == num_atoms.sum()
        return all_sampled_comp

    def get_crystal_graph(self, pymatgen_structure_list):
        crystal_graph_list = []
        for s in pymatgen_structure_list:
            crystal_graph = self.encoder3d.graph_converter(s)
            crystal_graph_list.append(crystal_graph.to(self.device))

        return crystal_graph_list
    
    def batched_graph_stat(self, batched_graph, crys_graph_list):
        """Batched graph states."""
        batched_atom_graph = batched_graph.batched_atom_graph
        
        edges_index = batched_atom_graph.T
        atom_owners = batched_graph.atom_owners.long()                                      # owners of batched atoms for feature indexing

        x = batched_graph.atom_positions                                                    # Cartesian coordinates 
        x = torch.cat(x, dim = 0)                                                           # Batched atom positions

        f = [g.atom_frac_coord for g in crys_graph_list]  
        f = torch.cat(f, dim = 0)

        l = [g.lattice for g in crys_graph_list]                                            # lattice row corresponds to a lattice vector
        l = torch.stack(l, dim = 0)
        l = l[atom_owners]                                                                  # l is the batched lattice consitent with EGNN

        Z = [g.atomic_number for g in crys_graph_list]
        Z = torch.cat(Z, dim=0)                                                             # Shift atomic number starting from 0
        return Z, l, f, edges_index, atom_owners
    
    def batched_structure_stat(self, structure_list: List[Structure], device: torch.device):
        """Batched structure states read out from pymatgen structure list.
        
        Args:
            structure_list (list): list of predicted oymatgen structure.
        
        Returns:
            Z (Tensor): atomic numbers. The shape should be (num of atoms).
            l (Tensor): lattice vectors. The shape should be (num of atoms, 3, 3).
            f (Tensor): fractional coordinates. The shape should be (num of atoms, 3).
            edge_index (Tensor): edge index for fully connected graph. The shape should 
                be (2, num of edges).
            atom_owners (Tensor): atom owners. The shape should be (num of atoms).
        """
        # Read info from structure list.
        Z = [torch.tensor(s.atomic_numbers) for s in structure_list]
        l = [torch.tensor(s.lattice.matrix) for s in structure_list]
        f = [torch.tensor(s.frac_coords) for s in structure_list]
        atom_owners = [
            torch.ones(len(s), dtype=torch.long) * i for i, s in enumerate(structure_list)
        ]
        cum_num_atoms = torch.tensor([0] + [len(s) for s in structure_list]).long()
        cum_num_atoms = cum_num_atoms.cumsum(dim=0)
        edge_index = [
            self.get_fc_edge_index(torch.arange(len(s))) + cum_num_atoms[i] 
            for i, s in enumerate(structure_list)
        ]
        
        # Convert to tensors.
        Z = torch.cat(Z, dim=0).to(device).long()
        l = torch.stack(l, dim=0).to(device).float()
        f = torch.cat(f, dim=0).to(device).float()
        atom_owners = torch.cat(atom_owners, dim=0).to(device).long()
        edge_index = torch.cat(edge_index, dim=1).to(device).long()
        l = l[atom_owners]
        
        return Z, l, f, edge_index, atom_owners
        
    def get_fc_edge_index(self, node_index):
        """Get edge index for fully connected graph with given node index.
        
        Args:
            node_index (Tensor): node index.
        
        Returns:
            edge_index (Tensor): edge index for the fully connected graph.
            The shape should be (2, num of edges).
        """
        edge_index = [(i,j) for i in node_index for j in node_index if i != j]
        edge_index = torch.Tensor(edge_index).long().T
        return edge_index
    
    def compute_grad(self, batched_graph, classifier, ld_kwargs):
        """Compute gradients for property and atomic forces.
        
        Args:
            batched_graph (BatchedGraph): batched graph.
            classifier (Classifier): classifier for property prediction.
            ld_kwargs (SimpleNamespace): args for doing annealed langevin dynamics sampling.
        
        Returns:
            batch_c_grad (Tensor): gradients for property. The shape should be
                (batch_size, 3).
            batch_forces (Tensor): gradients for atomic forces. The shape should be
                (batch_size, 3).
        """
        prediction = self.encoder3d._compute( batched_graph,
                                            compute_force = False,
                                            return_crystal_feas= True,
                                            )
        crystal_fea_all = prediction['crystal_fea']
        energy = prediction['e']        # TODO: whether to sum the energy or not.
        
        if ld_kwargs.compute_force:
            f_grad = torch.autograd.grad(energy.sum(), 
                                         batched_graph.atom_positions, 
                                        #  grad_outputs=torch.ones(log_p_c.shape, device = log_p_c.device), 
                                         create_graph=True, retain_graph=True)
            
            forces = [-1 * force_dim for force_dim in f_grad] # in cart_coords
            batch_forces = torch.cat(forces, dim = 0).detach()

            del f_grad
        else:
            batch_forces = None
        
        c = self.predict_property(crystal_fea_all)    # c is the property from a regressor, i.e. c = (c|R)
        # change the regression problem into a classfication probability
        
        p_c = classifier(c)              
        log_p_c = torch.log(p_c)
        c_grad = torch.autograd.grad(log_p_c, batched_graph.atom_positions, 
                                        grad_outputs=torch.ones(log_p_c.shape, device = log_p_c.device),        # TODO: Check the correctness of grad_outputs.
                                        create_graph=True, retain_graph= True)
        batch_c_grad = torch.cat(c_grad, dim = 0).detach()
        del c_grad
        return batch_c_grad, batch_forces
    
    # Training information.
    def compute_stats(self, batch, outputs, prefix):
        num_atom_loss = outputs['num_atom_loss']
        lattice_loss = outputs['lattice_loss']
        latt_loss = outputs['latt_loss']
        coord_loss = outputs['coord_loss']
        # type_loss = outputs['type_loss']
        kld_loss = outputs['kld_loss']
        composition_loss = outputs['composition_loss']
        property_loss = outputs['property_loss']

        loss = (
            self.hparams.cost_natom * num_atom_loss +
            self.hparams.cost_lattice * lattice_loss +
            self.hparams.cost_latt * latt_loss +
            self.hparams.cost_coord * coord_loss +
            # self.hparams.cost_type * type_loss +
            self.hparams.beta * kld_loss +
            self.hparams.cost_composition * composition_loss +
            self.hparams.cost_property * property_loss)

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_natom_loss': num_atom_loss,
            f'{prefix}_lattice_loss': lattice_loss,
            f'{prefix}_latt_loss': latt_loss,
            f'{prefix}_coord_loss': coord_loss,
            # f'{prefix}_type_loss': type_loss,
            f'{prefix}_kld_loss': kld_loss,
            f'{prefix}_composition_loss': composition_loss,
        }

        if prefix != 'train':
            # validation/test loss only has latt, coord and type
            loss = (
                self.hparams.cost_latt * latt_loss +
                self.hparams.cost_coord * coord_loss
                # self.hparams.cost_type * type_loss)
            )

            # evaluate num_atom prediction.
            pred_num_atoms = outputs['pred_num_atoms'].argmax(dim=-1)
            num_atom_accuracy = (
                pred_num_atoms == batch.num_atoms).sum() / batch.num_graphs

            # evalute lattice prediction.
            pred_lengths_and_angles = outputs['pred_lengths_and_angles']
            scaled_preds = self.lattice_scaler.inverse_transform(
                pred_lengths_and_angles)
            pred_lengths = scaled_preds[:, :3]
            pred_angles = scaled_preds[:, 3:]

            if self.hparams.lattice_scale_method == 'scale_length':
                pred_lengths = pred_lengths * batch.num_atoms.view(-1, 1).float()**(1/3)
            lengths_mard = mard(batch.lengths, pred_lengths)
            angles_mae = torch.mean(torch.abs(pred_angles - batch.angles))

            pred_volumes = lengths_angles_to_volume(pred_lengths, pred_angles)
            true_volumes = lengths_angles_to_volume(
                batch.lengths, batch.angles)
            volumes_mard = mard(true_volumes, pred_volumes)

            # evaluate atom type prediction.
            # pred_atom_types = outputs['pred_atom_types']
            # target_atom_types = outputs['target_atom_types']
            # type_accuracy = pred_atom_types.argmax(dim=-1) == (target_atom_types-1)
            # type_accuracy = scatter(type_accuracy.float(), batch.batch, dim=0, reduce='mean').mean()

            log_dict.update({
                f'{prefix}_loss': loss,
                f'{prefix}_property_loss': property_loss,
                f'{prefix}_natom_accuracy': num_atom_accuracy,
                f'{prefix}_lengths_mard': lengths_mard,
                f'{prefix}_angles_mae': angles_mae,
                f'{prefix}_volumes_mard': volumes_mard,
                # f'{prefix}_type_accuracy': type_accuracy,
            })

        else:
            self.log(f'{prefix}_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log(f'{prefix}_natom_loss', num_atom_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log(f'{prefix}_lattice_loss', lattice_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log(f'{prefix}_latt_loss', latt_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log(f'{prefix}_coord_loss', coord_loss, on_step=True, on_epoch=True, prog_bar=True)
            # self.log(f'{prefix}_type_loss', type_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log(f'{prefix}_kld_loss', kld_loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log(f'{prefix}_composition_loss', composition_loss, on_step=True, on_epoch=True, prog_bar=True)
           
        return log_dict, loss

    # Screening.
    def on_train_epoch_end(self):
        # read from self.log_dict
        metrics = self.trainer.logged_metrics

        ## TODO: change the loss_step to average loss of the epoch
        ## TODO: get loss_avg
        print("*"*100)
        print(f"Epoch {self.current_epoch} - loss: {metrics.get('train_loss_epoch', 0):.4f}")
        print(f"Epoch {self.current_epoch} - num_atom_loss: {metrics.get('train_natom_loss_epoch', 0):.4f}")
        print(f"Epoch {self.current_epoch} - lattice_loss: {metrics.get('train_lattice_loss_epoch', 0):.4f}")
        print(f"Epoch {self.current_epoch} - latt_loss: {metrics.get('train_latt_loss_epoch', 0):.4f}")
        print(f"Epoch {self.current_epoch} - coord_loss: {metrics.get('train_coord_loss_epoch', 0):.4f}")
        print(f"Epoch {self.current_epoch} - type_loss: {metrics.get('train_type_loss_epoch', 0):.4f}")
        print(f"Epoch {self.current_epoch} - kld_loss: {metrics.get('train_kld_loss_epoch', 0):.4f}")
        print(f"Epoch {self.current_epoch} - composition_loss: {metrics.get('train_composition_loss_epoch', 0):.4f}")
    
    # Functions for lattice convertion, and structure building up.
    def get_lattices(self, lengths, angles, num_atoms):
        """Reconstruct lattices in 3D from inner coordinates: lengths and angles.
        
        Args:
            lengths (Tensor): lattice lengths. The shape of lengths should be 
                (num of structure, 3).
            angles (Tensor): lattice angles. The shape of angles should be
                (num of structure, 3).
            num_atoms (Tensor): number of atoms for each structure. The shape of
                num_atoms should be (num of structure).
        
        Returns:
            lattices (Tensor): reconstructed lattices. The shape of lattices should
                be (num of structure, 3, 3).
        """
        structure_ids = torch.arange(len(num_atoms), device = num_atoms.device)
        structure_ids = structure_ids.repeat_interleave(num_atoms)
        lattice_list = []
        for structure_id in range(len(num_atoms)):
            indices = torch.where(structure_ids == structure_id)[0] 
            if len(indices) == 0: # remove the structure with zero atoms
                continue
            a, b, c = lengths[structure_id].cpu()
            alpha, beta, gamma = angles[structure_id].cpu()
            Latt = Lattice.from_parameters(
                a = a, b = b, c = c, alpha = alpha, beta = beta, gamma = gamma,
            )
            lattice_list.append(Latt.matrix)
        
        if len(lattice_list) == 1:
            return torch.tensor(lattice_list[0], device= num_atoms.device, dtype= torch.float32)[None, :]
        else:
            return torch.tensor(lattice_list, device = num_atoms.device, dtype= torch.float32)
    
    def get_pymatgen_structure(self, lengths, angles, num_atoms, frac_coords, atom_types):
        structure_ids = torch.arange(len(num_atoms), device = num_atoms.device)
        structure_ids = structure_ids.repeat_interleave(num_atoms)
        s_list = []
        lattice_list = []
        for structure_id in range(len(num_atoms)):
            indices = torch.where(structure_ids == structure_id)[0] 
            if len(indices) == 0: # remove the structure with zero atoms
                continue
            
            a, b, c = lengths[structure_id].cpu()
            alpha, beta, gamma = angles[structure_id].cpu()
            Latt = Lattice.from_parameters(
                a = a, b = b, c = c, alpha = alpha, beta = beta, gamma = gamma,
            )
            frac_ = frac_coords[indices]
            type_ = atom_types[indices]
            species_ = [Element.from_Z(ele_Z) for ele_Z in type_]

            s_gen = Structure(lattice= Latt , species= species_, coords= frac_.cpu().detach().numpy(),
                            to_unit_cell=False,coords_are_cartesian=False)

            s_list.append(s_gen)
            lattice_list.append(Latt.matrix)
        return s_list, lattice_list
    
    def get_pymatgen_structure_from_lattice(self, lattices, num_atoms, frac_coords, atom_types):
        """Get pymatgen structure from lattices, frac_coords and atom_types."""
        structure_ids = torch.arange(len(num_atoms), device = num_atoms.device)
        structure_ids = structure_ids.repeat_interleave(num_atoms)
        s_list = []
        for structure_id in range(len(num_atoms)):
            indices = torch.where(structure_ids == structure_id)[0] 
            if len(indices) == 0: # remove the structure with zero atoms
                continue
            Latt = Lattice(lattices[structure_id].cpu().detach().numpy())
            
            frac_ = frac_coords[indices]
            type_ = atom_types[indices]
            species_ = [Element.from_Z(ele_Z) for ele_Z in type_]

            s_gen = Structure(
                lattice = Latt , 
                species = species_, 
                coords = frac_.cpu().detach().numpy(),
                to_unit_cell = False,
                coords_are_cartesian = False,
            )

            s_list.append(s_gen)
        return s_list
    
    # Predictor.
    def predict_num_atoms(self, z: torch.Tensor):
        """Predict number of atoms from latent embeddings."""
        return self.fc_num_atoms(z)

    def predict_property(self, z: torch.Tensor):
        """Predict property from latent embeddings."""
        return self.fc_property(z)

    def predict_lattice(self, z: torch.Tensor, num_atoms: torch.Tensor):
        """Predict lattice from latent embeddings and number of atoms. 
        
        Args:
            z (Tensor): latent embeddings. The shape should be (batch_size, latent_dim).
            num_atoms (Tensor): number of atoms for each structures. The shape should
                be (batch_size). It is used to scale the predicted lattice to the correct 
                size. See more details in Niggli Reduction.
                
        Returns:
            pred_lengths_and_angles (Tensor): predicted lattice lengths and angles. It is 
                Niggli reduced and standardized (sacled). The shape should be (batch_size, 
                6).
            pred_lengths (Tensor): lattice lengths. It is recovered back to Niggli unreduced
                and unscaled space. The shape should be (batch_size, 3).
            pred_angles (Tensor): lattice angles. It is recovered back to Niggli unreduced
                and unscaled space. The shape should be (batch_size, 3).        
        """
        self.lattice_scaler.match_device(z)
        pred_lengths_and_angles = self.fc_lattice(z)  # (N, 6)
        # Recover standarization and Niggli reduction.
        scaled_preds = self.lattice_scaler.inverse_transform(pred_lengths_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        if self.hparams.lattice_scale_method == 'scale_length':
            pred_lengths = pred_lengths * num_atoms.view(-1, 1).float()**(1/3)
        return pred_lengths_and_angles, pred_lengths, pred_angles

    def predict_composition(self, z):
        """Predict composition from latent embeddings and number of atoms.
        
        Args:
            z (Tensor): The latent embeddings of crystal structures. The shape 
                should be (num of structures, latent_dim).
        
        Returns:
            pred_composition (Tensor): predicted composition. The shape should 
            be (num of structures, MAX_ATOMIC_NUM).
        """        
        pred_composition = self.fc_composition(z)
        pred_composition = F.softmax(pred_composition, dim=-1)
        return pred_composition

    # Loss functions.
    def num_atom_loss(self, pred_num_atoms: torch.Tensor, batch: Batch) -> torch.Tensor:
        """Compute loss for number of atoms."""
        return F.cross_entropy(pred_num_atoms, batch.num_atoms)

    def property_loss(self, z: torch.Tensor, batch: Batch) -> torch.Tensor:
        """Compute loss for property."""
        predict_propeties = self.fc_property(z)
        return F.mse_loss(predict_propeties, batch.properties)

    def lattice_loss(self, pred_lengths_and_angles: torch.Tensor, batch: Batch) -> torch.Tensor:
        """Compute loss for lattice reconstruction. 
        
        Args:
            pred_lengths_and_angles (Tensor): Predicted Niggli reduced and scaled 
                lengths and angles. The shape should be (num_structures, 6).
            batch (Batch): batched data.
        
        Returns:
            loss (Tensor): Mean squared error for lattices in scaled (standarized) and Niggli 
                reduced space.
        """
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        # Do standarization and Niggli reduction on target.
        if self.hparams.lattice_scale_method == 'scale_length':
            target_lengths = batch.lengths / batch.num_atoms.view(-1, 1).float()**(1/3)
        target_lengths_and_angles = torch.cat([target_lengths, batch.angles], dim=-1)
        target_lengths_and_angles = self.lattice_scaler.transform(target_lengths_and_angles)
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def composition_loss(
        self, 
        pred_composition: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        """Compute loss for composition.
        
        Args:
            pred_composition (Tensor): predicted composition. The shape should
                be (num of structures, MAX_ATOMIC_NUM).
            batch (Batch): batched data.
            NOTE: The atomic number index for pred_composition start from 0.
        Returns:
            loss (Tensor): Cross entropy loss for composition.
        """
        # Compute target composition from batch.
        atom_types_one_hot = F.one_hot(batch.atom_types-1, num_classes=MAX_ATOMIC_NUM)
        target_composition = scatter(
            atom_types_one_hot, batch.batch, dim=0, reduce='sum')/batch.num_atoms[:,None]
        # Compute cross entropy loss for composition.
        loss = self.mse_composition(pred_composition, target_composition)
        return loss.mean()

    def latt_loss(
        self,
        pred_latt_score: torch.Tensor,
        target_latt_score: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss for lattice denoising.
        
        Mathematics:
            L_l = 1/2* || sigma * s_theta (pred, sigma) + N(0, 1) ||_2^2
            where the neural network models sigma*s_theta.
        
        Args:
            pred_latt_score (Tensor): the predicted Niggli reduced but unscaled 
                lattice difference. The shape should be (num_structures, 9).
            target_latt_score (Tensor): target lattice difference to match prediction
                for adding noise to Niggli reduced by unscaled lattice. The shape 
                should be (num_structures, 9).
        
        Returns:
            loss (Tensor): Mean squared error for lattices in Niggli reduced 
                but not scaled space.
        """
        loss_per_atom = torch.sum((pred_latt_score - target_latt_score)**2, dim=1)
        loss_per_atom = 0.5 * loss_per_atom
        return loss_per_atom.mean()

    def coord_loss(
        self, 
        pred_frac_coord_score: torch.Tensor, 
        target_frac_coord_score: torch.Tensor,
        batch: Batch,
    ) -> torch.Tensor:
        """Compute loss for coordinates.
        
        Mathematics:
            L_c = 1/2 * || grad(log q(Ft|F0))/N - s_theta(pred, sigma)/N ||_2^2
            where the neural network models s_theta/N.
        
        Args:
            pred_frac_coord_score (Tensor): the predicted fractional coordinate 
                difference. The shape should be (num_atoms, 3).
            noisy_frac_coord_score (Tensor): the noisy fractional coordinates. The
                shape should be (num_atoms, 3).
            batch (Batch): batched data.
        """
        
        loss_per_atom = torch.sum((pred_frac_coord_score - target_frac_coord_score)**2, dim=1)
        loss_per_atom = 0.5 * loss_per_atom
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()

    def type_loss(
        self, 
        pred_atom_types: torch.Tensor, 
        target_atom_types: torch.Tensor,
        used_type_sigmas_per_atom: torch.Tensor, 
        batch: Batch,
    ) -> torch.Tensor:
        """Compute loss for atom types."""
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(
            pred_atom_types, target_atom_types, reduction='none')
        # rescale loss according to noise
        loss = loss / used_type_sigmas_per_atom
        return scatter(loss, batch.batch, reduce='mean').mean()

    def kld_loss(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence loss."""
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        return kld_loss