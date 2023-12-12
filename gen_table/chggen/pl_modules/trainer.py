import datetime
import inspect
import os
import random
import shutil
import time
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    MultiStepLR,
)
from torch.utils.data import DataLoader
from torch_geometric.data import Batch


class Trainer:
    """A trainer to train CHGgen with multiple loss"""

    def __init__(
        self,
        model: nn.Module,
        # targets,
        optimizer: str = "Adam",
        scheduler: str = "CosLR",
        epochs: int = 10,
        starting_epoch: int = 0,
        learning_rate: float = 1e-3,
        print_freq: int = 100,
        torch_seed: int = 114514,
        data_seed: int = 114514,
        **kwargs,
    ) -> None:
        
        self.trainer_args = {
            k: v
            for k, v in locals().items()
            if k not in ["self", "__class__", "model", "kwargs"]
        }
        self.trainer_args.update(kwargs)

        self.model = model
        # self.targets = targets
        self.device = self.model.device

        if torch_seed is not None:
            torch.manual_seed(torch_seed)
        if data_seed:
            random.seed(data_seed)

        # Define optimizer
        if optimizer == "SGD":
            momentum = kwargs.pop("momentum", 0.9)
            weight_decay = kwargs.pop("weight_decay", 0)
            self.optimizer = torch.optim.SGD(
                model.parameters(),
                learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif optimizer == "Adam":
            weight_decay = kwargs.pop("weight_decay", 0)
            self.optimizer = torch.optim.Adam(
                model.parameters(), learning_rate, weight_decay=weight_decay
            )
        elif optimizer == "AdamW":
            weight_decay = kwargs.pop("weight_decay", 1e-2)
            self.optimizer = torch.optim.AdamW(
                model.parameters(), learning_rate, weight_decay=weight_decay
            )
        elif optimizer == "RAdam":
            weight_decay = kwargs.pop("weight_decay", 0)
            self.optimizer = torch.optim.RAdam(
                model.parameters(), learning_rate, weight_decay=weight_decay
            )

        # Define learning rate scheduler
        if scheduler in ["MultiStepLR", "multistep"]:
            scheduler_params = kwargs.pop(
                "scheduler_params",
                {
                    "milestones": [4 * epochs, 6 * epochs, 8 * epochs, 9 * epochs],
                    "gamma": 0.3,
                },
            )
            self.scheduler = MultiStepLR(self.optimizer, **scheduler_params)
            self.scheduler_type = "multistep"
        elif scheduler in ["ExponentialLR", "Exp", "Exponential"]:
            scheduler_params = kwargs.pop("scheduler_params", {"gamma": 0.98})
            self.scheduler = ExponentialLR(self.optimizer, **scheduler_params)
            self.scheduler_type = "exp"
        elif scheduler in ["CosineAnnealingLR", "CosLR", "Cos", "cos"]:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=10 * epochs,  # Maximum number of iterations.
                eta_min=1e-2 * learning_rate,
            )
            self.scheduler_type = "cos"
        elif scheduler in ["CosRestartLR"]:
            scheduler_params = kwargs.pop("scheduler_params", {"T_0": 10, "T_mult": 2})
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, eta_min=1e-2 * learning_rate, **scheduler_params
            )
            self.scheduler_type = "cosrestart"
        else:
            raise NotImplementedError


    def train(self, train_loader: DataLoader, val_loader: DataLoader,):
        self.model.train()

        for ii, batch in enumerate(train_loader):
            
            batch
            print(batch.frac_coords.shape)

            # compute output
            predictions = self.model(batch, 
                                     teacher_forcing = True, training = True)

            total_loss = loss_1 + loss_2 + loss_3  + loss_regu

            losses.update(total_loss.data.cpu().item(), len(targets))


            targets = torch.stack(targets, dim = 1)

            Q0_mae_error = mae(output_Q0.data.cpu(), targets[:, 0, :].reshape([-1, 1]))
            Q0_mae_errors.update(Q0_mae_error, len(targets))

            Q_mae_error = mae(output_Q.data.cpu(), targets[:, 1, :].reshape([-1, 1]))
            Q_mae_errors.update(Q_mae_error, targets.size(0))

            dQ_mae_error = mae(output_dQ.data.cpu(), targets[:, 2, :].reshape([-1, 1]))
            dQ_mae_errors.update(dQ_mae_error, targets.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (ii % print_freq) == 0:

                print('Epoch: [{0}][{1}/{2}]\t'
                            'Loss: {loss.val:.4f}\t'
                            'Q0-MAE {Q0_mae_errors.val:.3f} ({Q0_mae_errors.avg:.3f})\t'
                            'Q-MAE {Q_mae_errors.val:.3f} ({Q_mae_errors.avg:.3f})\t'
                            'dQ-MAE {dQ_mae_errors.val:.3f} ({dQ_mae_errors.avg:.3f})\t'
                            'regu-loss {regu_loss:.3f}'.format(
                            epoch, ii+1, len(train_loader),
                            loss=losses, Q0_mae_errors=Q0_mae_errors,  Q_mae_errors = Q_mae_errors, dQ_mae_errors = dQ_mae_errors,
                            regu_loss = loss_regu.detach().cpu().numpy(),
                            )
                        )