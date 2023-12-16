"""CHGGEN model containing trainer and sampler."""
from typing import Any
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch_scatter import scatter
import pytorch_lightning as pl

from chggen.common.data_utils import ( 
    cart_to_frac_coords, 
    frac_to_cart_coords, 
    min_distance_sqr_pbc,
)
from chggen.pl_modules.encoder import CHGNet_encoder
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
    
    ## TODO: generalize the optimizer 
    def configure_optimizers(self, lr = 1e-3, use_lr_scheduler = True):
        opt = torch.optim.Adam(self.parameters(), lr= lr)
        if use_lr_scheduler:
            return [opt]
        scheduler = ExponentialLR(opt, gamma=0.95)
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}

class CHGGen(BaseModule):
    def __init__(self, 
                 hparams_dict = {'latent_dim': 256, 'hidden_dim': 128, 'property_dim': 1, 'load_pretrain': True, 'fc_num_layers': 2, 
                                 'sigma_begin': 10.0, 'sigma_end': 0.01, 'type_sigma_begin': 5.0, 'type_sigma_end': 0.01,
                                 'max_atoms': 20, 'predict_property': False, 'num_noise_level': 50, 
                                 'lattice_scale_method': 'scale_length', 
                                 'cost_natom': 1.0, 'cost_coord': 10.0, 'cost_type': 1.0, 'cost_lattice': 10.0, 'cost_composition': 1.0, 'cost_edge': 10.0, 'cost_property': 1.0,
                                 'beta': 0.01,
                                 'teacher_forcing_lattice': True,
                                 'teacher_forcing_max_epoch': 1000,
                                 'decoder': 'nequip_table'},
                 lattice_scaler = None,
                 **kwargs) -> None:
        super().__init__()
        
        self.save_hyperparameters(hparams_dict)
        self.lattice_scaler = lattice_scaler

        self.encoder = CHGNet_encoder(return_crystal_feas = True)


        if self.hparams.decoder == 'nequip_table':
            self.decoder = NequipTableDecoder()
        else:
            raise NotImplementedError

        # for property prediction.
        if self.hparams.predict_property:
            self.fc_property = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                         self.hparams.fc_num_layers, self.hparams.property_dim)

        sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.sigma_begin),
            np.log(self.hparams.sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

        # obtain from datamodule.
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
                step_lr (float): step size param.
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
        for sigma in tqdm(self.sigmas, total=self.sigmas.size(0), disable=ld_kwargs.disable_bar):
            if sigma < ld_kwargs.min_sigma:
                break
            step_size = ld_kwargs.step_lr * (sigma / self.sigmas[-1]) ** 2

            # Loop over steps for each noise level.
            for step in range(ld_kwargs.n_step_each):
                noise_cart = torch.randn_like(cur_frac_coords) * torch.sqrt(step_size * 2)
                cur_cart_coords = frac_to_cart_coords(
                    frac_coords=cur_frac_coords, 
                    lengths=lengths, 
                    angles=angles, 
                    num_atoms=num_atoms, 
                )
                with torch.no_grad():
                    pred_cart_coord_diff = self.decoder(
                        pred_cart_coords=cur_cart_coords, 
                        pred_atom_types=cur_atom_types, 
                        num_atoms=num_atoms, 
                        lengths=lengths, 
                        angles=angles,
                    )
                pred_cart_coord_diff = pred_cart_coord_diff / sigma

                cur_cart_coords = cur_cart_coords + step_size * pred_cart_coord_diff + noise_cart
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
                all_frac_coords=torch.stack(all_frac_coords, dim=0),
                all_atom_types=torch.stack(all_atom_types, dim=0),
                all_pred_cart_coord_diff=torch.stack(
                    all_pred_cart_coord_diff, dim=0),
                all_noise_cart=torch.stack(all_noise_cart, dim=0),
                is_traj=True))

        return output_dict

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


    def generate_rand_init(self, pred_composition_per_atom, pred_lengths,
                           pred_angles, num_atoms, batch):
        rand_frac_coords = torch.rand(num_atoms.sum(), 3,
                                      device=num_atoms.device)
        pred_composition_per_atom = F.softmax(pred_composition_per_atom,
                                              dim=-1)
        rand_atom_types = self.sample_composition(
            pred_composition_per_atom, num_atoms)
        return rand_frac_coords, rand_atom_types

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
            self.log(f'{prefix}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=16)
            self.log(f'{prefix}_coord_loss', coord_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=16)
           
        return log_dict, loss


    def on_train_epoch_end(self):
        # read from self.log_dict
        metrics = self.trainer.logged_metrics

        ## TODO: change the loss_step to average loss of the epoch
        ## TODO: get loss_avg
        print("*"*100)
        print(f"Epoch {self.current_epoch} - loss: {metrics.get('train_loss_step', 0):.4f}")
        print(f"Epoch {self.current_epoch} - num_atom_loss: {metrics.get('train_natom_loss_step', 0):.4f}")
        print(f"Epoch {self.current_epoch} - lattice_loss: {metrics.get('train_lattice_loss_step', 0):.4f}")
        print(f"Epoch {self.current_epoch} - coord_loss: {metrics.get('train_coord_loss_step', 0):.4f}")
        print(f"Epoch {self.current_epoch} - type_loss: {metrics.get('train_type_loss_step', 0):.4f}")
        print(f"Epoch {self.current_epoch} - kld_loss: {metrics.get('train_kld_loss_step', 0):.4f}")
        print(f"Epoch {self.current_epoch} - composition_loss: {metrics.get('train_composition_loss_step', 0):.4f}")
        
