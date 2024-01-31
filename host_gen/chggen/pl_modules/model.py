"""CHGGEN model containing trainer and sampler."""
from typing import Any
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch_scatter import scatter
import pytorch_lightning as pl

from chggen.common.data_utils import ( 
    cart_to_frac_coords, 
    frac_to_cart_coords, 
    min_distance_sqr_pbc,
)
from chggen.pl_modules.decoder import NequipTableDecoder



def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim, use_layernorm = False):
    if use_layernorm:
        mods = [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
    else:
        mods = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]

    for i in range(fc_num_layers-1):
        if use_layernorm:
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()]
        else:
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)

class BaseModule(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        # populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters()
        
    # def configure_optimizers(self, use_lr_scheduler = True):
    #     opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    #     if not use_lr_scheduler:
    #         return [opt]
    #     scheduler = ExponentialLR(opt, gamma=0.95)
    #     return {
    #         "optimizer": opt, 
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "monitor": "val_loss",
    #         }
    #     }
    
    def configure_optimizers(self, use_lr_scheduler = True):
        opt = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        if not use_lr_scheduler:
            return [opt]
        
        num_epoch = self.trainer.max_epochs
        num_batch = self.hparams.num_batch
        total_step = num_epoch * num_batch  # Total number of training steps
        warmup_step = 0.1 * total_step      # Number of steps for the warm-up (10% of total)
        
        eta_max = self.hparams.lr           # Maximum learning rate
        eta_min = 1e-2 * self.hparams.lr    # Minimum learning rate (1% of max lr)

        # Lambda function for linear warmup
        warmup_lambda = lambda step: step / warmup_step
        # Lambda function for cosine annealing after warmup
        cosine_lambda = lambda step: (eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * (step - warmup_step) / (total_step - warmup_step)))) / self.hparams.lr
        # Combined lambda function
        combined_lambda = lambda step: cosine_lambda(step) if step >= warmup_step else warmup_lambda(step)
        # Create the scheduler
        scheduler = LambdaLR(opt, lr_lambda=combined_lambda)
        return {
            "optimizer": opt, 
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # or "epoch" depending on your preference
                "frequency": 1,
            }
        }

class CHGGen(BaseModule):
    def __init__(
            self, 
            hparams_dict = {'latent_dim': 256, 'hidden_dim': 128, 'property_dim': 1, 'fc_num_layers': 2, 
                            'sigma_begin': 10.0, 'sigma_end': 0.01,
                            'predict_property': False, 'num_noise_level': 50, 
                            'cost_coord': 10.0, 'cost_property': 1.0,
                            'beta': 0.01,
                            'teacher_forcing_lattice': True,
                            'teacher_forcing_max_epoch': 1000,
                            'decoder': 'nequip',
                            'lr': 1e-3,},
            lattice_scaler = None,
            **kwargs,
        ) -> None:
        super().__init__()
        
        self.save_hyperparameters(hparams_dict)

        # Initialize decoder.
        if self.hparams.decoder == 'nequip':
            self.decoder = NequipTableDecoder(model_version = 'nequip')
        elif self.hparams.decoder == 'nequip_v2':
            self.decoder = NequipTableDecoder(model_version = 'nequip_v2')
        else:
            raise NotImplementedError

        # For property prediction.
        if self.hparams.predict_property:
            self.fc_property = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                         self.hparams.fc_num_layers, self.hparams.property_dim)
        # Noise levels.
        sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.sigma_begin),
            np.log(self.hparams.sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)
        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

        # Obtain from datamodule.
        self.lattice_scaler = lattice_scaler
        self.scaler = None

    def langevin_dynamics(
        self, 
        lengths: torch.Tensor,
        angles: torch.Tensor,
        composition: torch.Tensor,
        num_atoms: torch.Tensor,
        ld_kwargs, 
    ):
        """Decode crystral structure given lattice and num_atoms.
            
        Args:
            lengths (tensor): lattice lengths of shape (batch_size, 3).
            angles (tensor): lattice angles of shape (batch_size, 3).
            composition(tensor): composition of shape (batch_size, num_atom_per_structure).
            ld_kwargs (SimpleNameSpace): arguments for annealed langevin dynamics sampling:
                n_step_each (int): number of steps for each sigma level.
                min_sigma (float): minimum sigma to use in annealed langevin dynamics.
                save_traj (bool): if <True>, save the entire LD trajectory.
                disable_bar (bool): disable the progress bar of langevin dynamics.
        """

        if ld_kwargs.save_traj:
            all_frac_coords = []
            all_pred_cart_coord_diff = []
            all_noise_cart = []
            all_atom_types = []

        # Initialize fractional coordinates.
        cur_frac_coords = torch.rand((num_atoms.sum(), 3), device=self.device, requires_grad = False)
        
        # Rearange composition.
        cur_atom_types = composition.view(-1)

        # Loop over noise levels.
        for (step_size, sigma) in tqdm(zip(self.sigmas[:-1]**2-self.sigmas[1:]**2, self.sigmas[:-1]), total=self.sigmas.size(0)-1, disable=ld_kwargs.disable_bar):
            
            # Predictor
            cur_cart_coords = frac_to_cart_coords(
                frac_coords=cur_frac_coords, 
                lengths=lengths, 
                angles=angles, 
                num_atoms=num_atoms, 
            )

            # Compute score term
            with torch.no_grad():   
                pred_cart_coord_diff = self.decoder(
                    pred_cart_coords=cur_cart_coords, 
                    pred_atom_types=cur_atom_types, 
                    num_atoms=num_atoms, 
                    lengths=lengths, 
                    angles=angles,
                )
            pred_cart_coord_diff = pred_cart_coord_diff / sigma
            
            # Compute noise term
            noise_cart = torch.randn_like(cur_cart_coords) * torch.sqrt(step_size)
            
            # Update
            cur_cart_coords = cur_cart_coords + step_size * pred_cart_coord_diff + noise_cart/3.74
            cur_frac_coords = cart_to_frac_coords(
                cart_coords=cur_cart_coords, 
                lengths=lengths, 
                angles=angles, 
                num_atoms=num_atoms,
            )
            
            # Corrector
            for step in range(ld_kwargs.n_step_each):
                
                cur_cart_coords = frac_to_cart_coords(
                    frac_coords=cur_frac_coords, 
                    lengths=lengths, 
                    angles=angles, 
                    num_atoms=num_atoms, 
                )
                
                # Compute score term
                with torch.no_grad():
                    pred_cart_coord_diff = self.decoder(
                        pred_cart_coords=cur_cart_coords, 
                        pred_atom_types=cur_atom_types, 
                        num_atoms=num_atoms, 
                        lengths=lengths, 
                        angles=angles,
                    )
                pred_cart_coord_diff = pred_cart_coord_diff / sigma
                
                # Compute step term
                eps = 2 * (
                    ld_kwargs.signal_to_noise_ratio * torch.sqrt(torch.tensor(3)) / torch.norm(pred_cart_coord_diff, dim=1, keepdim=True)    
                )**2
                
                # Compute noise term
                noise_cart = torch.randn_like(cur_cart_coords) * torch.sqrt(2*eps)

                # Update
                cur_cart_coords = cur_cart_coords + eps * pred_cart_coord_diff + noise_cart
                cur_frac_coords = cart_to_frac_coords(
                    cart_coords=cur_cart_coords, 
                    lengths=lengths, 
                    angles=angles, 
                    num_atoms=num_atoms,
                )

            if ld_kwargs.save_traj:
                all_frac_coords.append(cur_frac_coords)
                all_pred_cart_coord_diff.append(
                    step_size * pred_cart_coord_diff)
                all_noise_cart.append(noise_cart)
                all_atom_types.append(cur_atom_types)
        
        output_dict = {'num_atoms': num_atoms, 'lengths': lengths, 'angles': angles,
                       'frac_coords': cur_frac_coords, 'atom_types': cur_atom_types,
                       'is_traj': False}

        if ld_kwargs.save_traj:
            output_dict.update(dict(
                all_frac_coords=torch.cat(all_frac_coords, dim=0),
                all_atom_types=torch.cat(all_atom_types, dim=0),
                all_pred_cart_coord_diff=torch.cat(all_pred_cart_coord_diff, dim=0),
                all_noise_cart=torch.cat(all_noise_cart, dim=0),
                is_traj=True))

        return output_dict

    def conditional_langevin_dynamics(
        self,
        lengths: torch.Tensor,
        angles: torch.Tensor,
        composition: torch.Tensor,
        num_atoms: torch.Tensor,
        ori_frac_coords: torch.Tensor,
        mask: torch.Tensor,
        ld_kwargs,
    ) -> dict:
        """Decode crystral structure given lattice and num_atoms with 
        fixed framework.
        
        Args:
            lengths (tensor): lattice lengths of shape (batch_size, 3).
            angles (tensor): lattice angles of shape (batch_size, 3).
            composition(tensor): composition of shape (batch_size, num_atom_per_structure).
            num_atoms (tensor): number of atoms of shape (batch_size,).
            ori_frac_coords (tensor): fractional coordinates for framework of shape (num_atom, 3).
            mask (tensor): mask of inpainting atoms on ori_frac_coords(num_atom, 1).
            ld_kwargs (SimpleNameSpace): arguments for annealed langevin dynamics sampling:
                n_step_each (int): number of steps for each sigma level.
                min_sigma (float): minimum sigma to use in annealed langevin dynamics.
                save_traj (bool): if <True>, save the entire LD trajectory.
                disable_bar (bool): disable the progress bar of langevin dynamics.
        """
        if ld_kwargs.save_traj:
            all_frac_coords = []
            all_atom_types = []
        
        times = self.get_scheduler(t_T=self.sigmas.size(0))[:-1]  # -1 is the end term.
        
        # Revert sigmas and step_size to accomodate the scheduler.
        step_sizes = self.sigmas[:-1]**2 - self.sigmas[1:]**2
        step_sizes = torch.flip(step_sizes, dims=(0,))          # Increasing step size.
        sigmas = torch.flip(self.sigmas[:-1], dims=(0,))        # Increasing noise level.
        
        # Langevin dynamics.
        
        # Initialize fractional coordinates for masked atoms.
        cur_frac_coords = ori_frac_coords.clone()
        cur_frac_coords[mask] = torch.rand((mask.sum().item(), 3), device=self.device, requires_grad = False)
        ori_cart_coords = frac_to_cart_coords(
            frac_coords=ori_frac_coords,
            lengths=lengths,
            angles=angles,
            num_atoms=num_atoms,
        )
        
        # Rearange composition.
        cur_atom_types = composition.view(-1)
        
        for t_last, t_cur in tqdm(zip(times[:-1], times[1:]), total=len(times)-1, disable=ld_kwargs.disable_bar):
            if t_cur < t_last:  # Backward denoise
                
                # Obtain x_unknown at t_cur
                cur_cart_coords = frac_to_cart_coords(
                    frac_coords=cur_frac_coords, 
                    lengths=lengths, 
                    angles=angles, 
                    num_atoms=num_atoms, 
                )
                
                # Compute score term
                with torch.no_grad():   
                    pred_cart_coord_diff = self.decoder(
                        pred_cart_coords=cur_cart_coords, 
                        pred_atom_types=cur_atom_types, 
                        num_atoms=num_atoms, 
                        lengths=lengths, 
                        angles=angles,
                    )
                pred_cart_coord_diff = pred_cart_coord_diff / sigmas[t_cur]
                
                # Compute noise term
                
                noise_cart = torch.randn_like(cur_cart_coords) * torch.sqrt(step_sizes[t_cur])   # Backward cart coords noise
    
                # Update
                cur_cart_coords = cur_cart_coords + step_sizes[t_cur] * pred_cart_coord_diff + noise_cart/3.74
                cur_frac_coords = cart_to_frac_coords(
                    cart_coords=cur_cart_coords, 
                    lengths=lengths, 
                    angles=angles, 
                    num_atoms=num_atoms,
                )
                
                # Corrector
                for step in range(ld_kwargs.n_step_each):
                    
                    cur_cart_coords = frac_to_cart_coords(
                        frac_coords=cur_frac_coords, 
                        lengths=lengths, 
                        angles=angles, 
                        num_atoms=num_atoms, 
                    )
                    
                    # Compute score term
                    with torch.no_grad():
                        pred_cart_coord_diff = self.decoder(
                            pred_cart_coords=cur_cart_coords, 
                            pred_atom_types=cur_atom_types, 
                            num_atoms=num_atoms, 
                            lengths=lengths, 
                            angles=angles,
                        )
                    pred_cart_coord_diff = pred_cart_coord_diff / sigmas[t_cur]
                    
                    # Compute step term
                    eps = 2 * (
                        ld_kwargs.signal_to_noise_ratio * torch.sqrt(torch.tensor(3)) / torch.norm(pred_cart_coord_diff, dim=1, keepdim=True)    
                    )**2
                    
                    # Compute noise term
                    noise_cart = torch.randn_like(cur_cart_coords) * torch.sqrt(2*eps)

                    # Update
                    cur_cart_coords = cur_cart_coords + eps * pred_cart_coord_diff + noise_cart
                    cur_frac_coords = cart_to_frac_coords(
                        cart_coords=cur_cart_coords, 
                        lengths=lengths, 
                        angles=angles, 
                        num_atoms=num_atoms,
                    )
                
                # Obtain x_known at t_cur
                cur_cart_coords = frac_to_cart_coords(
                    frac_coords=cur_frac_coords,
                    lengths=lengths,
                    angles=angles,
                    num_atoms=num_atoms,
                )
                noise_cart = torch.randn_like(ori_cart_coords) * sigmas[t_cur]   # Forward cart coords noise

                # Update
                known_cur_cart_coords = ori_cart_coords + noise_cart
                
                # Add x_known and x_unknown
                cur_cart_coords[~mask] = known_cur_cart_coords[~mask]
                
                # Perioric boundary condition
                cur_frac_coords = cart_to_frac_coords(
                    cart_coords=cur_cart_coords, 
                    lengths=lengths, 
                    angles=angles, 
                    num_atoms=num_atoms,
                )
                
                if ld_kwargs.save_traj:
                    all_frac_coords.append(cur_frac_coords)
                    all_atom_types.append(cur_atom_types)
                
            else:   # Forward diffusion
                cur_cart_coords = frac_to_cart_coords(
                    frac_coords=cur_frac_coords, 
                    lengths=lengths, 
                    angles=angles, 
                    num_atoms=num_atoms, 
                )
                
                # Compute noise term
                noise_cart = torch.randn_like(cur_cart_coords) * step_sizes[t_cur]   # Forward cart coords noise
                
                # Update
                cur_cart_coords = cur_cart_coords + noise_cart
                cur_frac_coords = cart_to_frac_coords(
                    cart_coords=cur_cart_coords, 
                    lengths=lengths, 
                    angles=angles, 
                    num_atoms=num_atoms,
                )
        
        output_dict = {'num_atoms': num_atoms, 'lengths': lengths, 'angles': angles,
                       'frac_coords': cur_frac_coords, 'atom_types': cur_atom_types,
                       'is_traj': False}
        
        if ld_kwargs.save_traj:
            output_dict.update(dict(
                all_frac_coords=torch.cat(all_frac_coords, dim=0),
                all_atom_types=torch.cat(all_atom_types, dim=0),
                is_traj=True))
            
        return output_dict
    
    @staticmethod
    def get_scheduler(
        t_T = 200,
        jump_len = 10,
        jump_n_sample = 3,
    ):
        """Obtain a sigma scheduler for the given parameters."""
        jumps = {} 
        for j in range(0, t_T - jump_len, jump_len): 
            jumps[j] = jump_n_sample - 1 

        t = t_T 
        ts = [] 
        while t >= 1: 
            t = t-1 
            ts.append(t)
            if jumps.get(t, 0) > 0: 
                jumps[t] = jumps[t] - 1 
                for _ in range(jump_len): 
                    t=t+1 
                    ts.append(t) 
        ts.append(-1)
        return ts
        
    def sample(
        self, 
        lattice_size: int,
        num_samples: int, 
        ld_kwargs,
    ):
        """Sample new structures for langevin_deynamics"""
        # TODO: Add Predictor for average number density.
        ave_volume = 12.8
        ave_num_atoms = round(lattice_size ** 3 / ave_volume)
        # TODO: Add Composition Sampler for atom types sampling. 
        atom_types = torch.randint(1, 104, (ave_num_atoms*num_samples,), device = self.device)
        num_atoms = torch.tensor([ave_num_atoms], device=self.device).repeat(num_samples)
        lengths = torch.ones((num_samples, 3), device=self.device) * lattice_size
        angles = torch.ones((num_samples, 3), device=self.device) * 90
        
        results = self.langevin_dynamics(
            lengths=lengths,
            angles=angles,
            composition=atom_types,
            num_atoms=num_atoms,
            ld_kwargs=ld_kwargs,
        )
        return results

    def forward(
        self, 
        batch,  
        training,           # Necessary parameter for pytorch_lightning.
    ):  
        
        # Sample noise levels.
        noise_level = torch.randint(
            low=0, 
            high=self.sigmas.size(0), 
            size=(batch.num_atoms.size(0),),
            device=self.device,
        )
        used_sigmas_per_atom = self.sigmas[noise_level]\
            .repeat_interleave(batch.num_atoms, dim=0)
            
        # Add noise to the cartesian coordinates.
        cart_noises_per_atom = torch.randn_like(batch.frac_coords) \
            * used_sigmas_per_atom[:, None]
        cart_coords = frac_to_cart_coords(
            frac_coords=batch.frac_coords, 
            lengths=batch.lengths, 
            angles=batch.angles, 
            num_atoms=batch.num_atoms,
        )
        cart_coords = cart_coords + cart_noises_per_atom

        pred_cart_coord_diff  = self.decoder(
            pred_cart_coords=cart_coords, 
            pred_atom_types=batch.atom_types, 
            num_atoms=batch.num_atoms, 
            lengths=batch.lengths, 
            angles=batch.angles,
        )
        
        # Compute loss.
        coord_loss = self.coord_loss(
            pred_cart_coord_diff, 
            cart_coords, 
            used_sigmas_per_atom, 
            batch,
        )

        return {
            'coord_loss': coord_loss,
            'pred_cart_coord_diff': pred_cart_coord_diff,
            'target_frac_coords': batch.frac_coords,
            'target_atom_types': batch.atom_types,
        }

    def sample_composition(self, composition_prob, num_atoms):
        """Sample composition such that it exactly satisfies composition_prob"""
        batch = torch.arange(
            len(num_atoms), device=num_atoms.device).repeat_interleave(num_atoms)
        assert composition_prob.size(0) == num_atoms.sum() == batch.size(0)
        composition_prob = scatter(
            composition_prob, index=batch, dim=0, reduce='mean')

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

    def predict_composition(self, z, num_atoms):
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_composition_per_atom,
                               target_atom_types, reduction='none')
        
        return scatter(loss, batch.batch, reduce='mean').mean()

    def coord_loss(
        self, 
        pred_cart_coord_diff, 
        noisy_cart_coords,
        used_sigmas_per_atom, 
        batch,
    ) -> torch.Tensor:
        """Compute the coordinate loss."""
        target_cart_coords = frac_to_cart_coords(
            batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        _, target_cart_coord_diff = min_distance_sqr_pbc(
            target_cart_coords, noisy_cart_coords, batch.lengths, batch.angles,
            batch.num_atoms, self.device, return_vector=True)

        target_cart_coord_diff = target_cart_coord_diff / \
            used_sigmas_per_atom[:, None]**2
        pred_cart_coord_diff = pred_cart_coord_diff / \
            used_sigmas_per_atom[:, None]

        loss_per_atom = torch.sum(
            (target_cart_coord_diff - pred_cart_coord_diff)**2, dim=1)

        loss_per_atom = 0.5 * loss_per_atom * used_sigmas_per_atom**2
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        teacher_forcing = (
            self.current_epoch <= self.hparams.teacher_forcing_max_epoch)
        outputs = self(batch, training=True)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=16,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=16,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
        self.log_dict(
            log_dict,
            batch_size=16,
        )
        return loss

    def compute_stats(
        self, 
        batch, 
        outputs, 
        prefix,
    ) -> Any:
        coord_loss = outputs['coord_loss']
        loss = self.hparams.cost_coord * coord_loss
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        
        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_coord_loss': coord_loss,
        }

        if prefix != 'train':
            # validation/test loss only has coord and type
            loss = self.hparams.cost_coord * coord_loss
            log_dict.update({
                f'{prefix}_loss': loss,
            })

        else:
            self.log(f'{prefix}_lr', current_lr, on_step=True, on_epoch=False, prog_bar=True, batch_size=16)
            self.log(f'{prefix}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=16)
            self.log(f'{prefix}_coord_loss', coord_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=16)
        
        return log_dict, loss

    def on_train_epoch_end(self):
        # read from self.log_dict
        metrics = self.trainer.logged_metrics

        print("*"*100)
        print(f"Epoch {self.current_epoch} - loss: {metrics.get('train_loss_epoch', 0):.4f}")
        print(f"Epoch {self.current_epoch} - coord_loss: {metrics.get('train_coord_loss_epoch', 0):.4f}")