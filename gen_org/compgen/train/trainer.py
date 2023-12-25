"""Auxiliary functions for denoising project."""

from __future__ import annotations

import numpy as np
import os
import inspect
import shutil

import time
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    MultiStepLR,
)
from typing import Tuple, List, Dict, Literal

from .loss import (
    KLDivergence,
    CompositionLoss,
    VolumeLoss,
)
from ..utils import (
    AverageMeter,
)
from ..model.composition_model import CompositionModel



class Trainer():
    """Trainer for training the model."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: str = "Adam",
        scheduler: str = "CosLR",
        epochs: int = 50,
        starting_epoch: int = 0,
        learning_rate: float = 1e-3,
        print_freq: int = 100,
        torch_seed: int | None = None,
        data_seed: int | None = None,
        device: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize all functions and hyperparameters for trainer.
        
        Args:
            model (nn.Module): model to be trained.
            optimizer (str): optimizer to update the model paremeters. Can be "Adam", "SGD", "AdamW", 
            "RAdam".
            scheduler (str): scheduler to adjust the learning rate. Can be "CosLR", "ExponentialLR", 
                "StepLR".
            criterion (tuple(str, str)): loss function criterion between output and groundtruth. Can be "MSE", "BCE", 
                "wBCE", "Focal loss", "IoU". 
            epochs (int): number of epochs for training.
                Default = 50
            starting_epoch (int): The epoch number to start training at.
                Default = 0
            learning_rate (float): initial learning rate.
                Default = 1e-3
            print_freq (int): frequency to print training output.
                Default = 100
            torch_seed (int, optional): random seed for torch.
            data_seed (int, optional): random seed for numpy.
            device (str, optional): use CPU or GPU to train the model.
            *kwargs (dict): additional hyperparameters for optimizer, scheduler, etc. 
        """
        # Store training args for reproducibility
        self.trainer_args = {
            k: v for k, v in locals().items() if k not in ["self", "__class__", "kwargs", "model"]
        }
        self.trainer_args.update(kwargs)
        
        self.model = model
        if torch_seed is not None:
            torch.manual_seed(torch_seed)
        if data_seed is not None:
            np.random.seed(data_seed)
        
        # Define optimizer
        if optimizer == "SGD":
            momentum = kwargs.pop("momentum", 0.9)
            weight_decay = kwargs.pop("weight_decay", 0.0)
            self.optimizer = torch.optim.SGD(
                params=model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif optimizer == "Adam":
            weight_decay = kwargs.pop("weight_decay", 0.0)
            self.optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer == "AdamW":
            weight_decay = kwargs.pop("weight_decay", 1e-2)
            self.optimizer = torch.optim.AdamW(
                params=model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer == "RAdam":
            weight_decay = kwargs.pop("weight_decay", 0.0)
            self.optimizer = torch.optim.RAdam(
                params=model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        
        # Define learning rate scheduler
        if scheduler in ["MultiStepLR", "multistep"]:
            scheduler_params = kwargs.pop(
                "scheduler_params",
                {
                    "milestones": [4*epochs, 6*epochs, 8*epochs, 10*epochs],
                    "gamma": 0.3,
                }
            )
            self.scheduler = MultiStepLR(self.optimizer, **scheduler_params)
            self.scheduler_type = "multistep"
        elif scheduler in ["ExponeitialLR", "Exp", "Exponential"]:
            scheduler_params = kwargs.pop("shceduler_params", {"gamma": 0.98})
            self.scheduler = ExponentialLR(self.optimizer, **scheduler_params)
            self.scheduler_type = "exp"
        elif scheduler in ["CosineAnnealingLR", "CosLR", "Cos", 'cos']:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=10*epochs,                # Maximum number of iteration
                eta_min = 1e-2*learning_rate    # The lowest learning rate
            )
            self.scheduler_type = 'cos'
        elif scheduler in ["CosRestartLR"]:
            scheduler_params = kwargs.pop("scheduler_params", {"T_0" :10, "T_mult": 2})
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, eta_min=1e-2*learning_rate, **scheduler_params
            )
            self.scheduler_type = "cosrestart"
        else:
            raise NotImplementedError
        
        # Define loss criterion
        self.kl_loss = KLDivergence()
        self.comp_loss = CompositionLoss()
        self.ave_v_loss = VolumeLoss()
        
        self.epochs = epochs
        self.starting_epoch = starting_epoch
        
        # Determine the device to use.
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self.print_freq = print_freq
        self.training_history: Dict[Literal["train", "val", "test"], List[float]] = {
            "train": [], "val": [], "test": []
        }
        self.best_model = None
        
    def train(
        self,
        train_loader: DataLoader,
        save_dir: str | None = None,
    ) -> None:
        """Train the model using torch dataloader.
        
        Args:
            train_loader (DataLoader): dataloader to update model parameters.
            save_dir (str, optional): the directory to save the trained parameters or model.
        """
        
        if self.model is None:
            raise ValueError("Model is not initialized")
        if save_dir is None:
            save_dir = f"{datetime.now():%m-%d-%Y}"
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Begin Training: using {self.device} device")
        self.model.to(self.device)
        
        for epoch in range(self.starting_epoch, self.epochs):
            # train
            train_loss = self._train(train_loader, epoch)
            self.training_history["train"].append(train_loss)
            
            self.save_checkpoint(epoch, train_loss, save_dir=save_dir)
            
    def _train(
        self, 
        train_loader: DataLoader, 
        current_epoch: int
    ) -> float:
        """Train all data for one epoch.
        
        Args:
            train_loader (DataLoader): train loader to update model parameters.
            current_epoch (int): used for resume unfinished training.
            
        Returns:
            training errors.
        """
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        kl_losses = AverageMeter()
        comp_losses = AverageMeter()
        ave_v_losses = AverageMeter()
        
        # switch to train mode
        self.model.train()
        
        start = time.perf_counter()         # start timer.
        for idx, batch in enumerate(train_loader):
            # measure data loading time.
            data_time.update(time.perf_counter()-start)
            
            # Forward pass output.
            mu, logvar, pred_comp, pred_ave_v = self.model(batch.to(self.device))
            
            # Compute loss.
            kl_loss = self.kl_loss(mu, logvar)
            comp_loss = self.comp_loss(pred_comp, batch)
            ave_v_loss = self.ave_v_loss(pred_ave_v, batch)
            loss = kl_loss + comp_loss + ave_v_loss
            
            # Update loss.
            kl_losses.update(kl_loss.item(), n = len(batch.ave_v))
            comp_losses.update(comp_loss.item(), n = len(batch.ave_v))
            ave_v_losses.update(ave_v_loss.item(), n = len(batch.ave_v))
            losses.update(loss.item(), n = len(batch.ave_v))

            # Backward pass.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # adjust learning rate every 1/10 of the epoch.
            if idx + 1 in np.arange(1,11) * len(train_loader) // 10:
                self.scheduler.step()
            
            # measure elapsed time
            batch_time.update(time.perf_counter() - start)
            start = time.perf_counter()
            
            # Print loss.
            if idx == 0 or (idx + 1) % self.print_freq == 0:
                message = (
                    f"Epoch: [{current_epoch+1}][{idx+1}/{len(train_loader)}]\t"
                    f"Time ({batch_time.ave:.3f}) Data ({data_time.ave:.3f}) "
                    f"Loss {losses.val:.4f} ({losses.ave:.4f}) "
                    f"KL Loss {kl_losses.val:.4f} ({kl_losses.ave:.4f}) " 
                    f"Comp Loss {comp_losses.val:.4f} ({comp_losses.ave:.4f}) " 
                    f"Ave V Loss {ave_v_losses.val:.4f} ({ave_v_losses.ave:.4f})"
                )            
                print(message)
        return round(losses.ave, 6)
    
    def get_best_model(self) -> nn.Module:
        """Get best model recorded in the trainer."""
        if  self.best_model is None:
            raise RuntimeError("the model must be trained first")
        MAE = min(self.training_history["val"])
        print(f"Best model val: {MAE=:.4}")
        return self.best_model
    
    @property
    def _init_keys(self):
        return [
            key for key in list(inspect.signature(Trainer.__init__).parameters)
            if key not in (["self", "model", "kwargs"])
        ]
        
    def save(
        self, filename: str = "training_result.pth.tar"
    ) -> None:
        """Save the model, optimizer, etc."""
        state = {
            "model": self.model.as_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "training_history": self.training_history,
            "trainer_args": self.trainer_args,
        }
        torch.save(state, filename)
    
    def save_checkpoint(
        self, 
        epoch: int, 
        mae_error: float, 
        save_dir: str | None = None,
    ) -> None:
        """Save model trained parameters after each epoch.
        
        Args:
            epoch (int): the epoch number.
            mae_error (dict): MAE error.
            save_dir (str): the directory to save trained parameters.
        """
        for fname in os.listdir(save_dir):
            if fname.startswith('epoch'):
                os.remove(os.path.join(save_dir, fname))
        rounded_mae = round(mae_error * 1000)  
        filename = os.path.join(
            save_dir,
            f"epoch{epoch}_mae_{rounded_mae}.pth.tar",
        )
        self.save(filename=filename)
        
        # save the model if it has minimal val loss.
        if mae_error == min(self.training_history["train"]):
            self.best_model = self.model
            for fname in os.listdir(save_dir):
                if fname.startswith("best"):
                    os.remove(os.path.join(save_dir, fname))
            shutil.copyfile(
                filename,
                os.path.join(
                    save_dir,
                    f"best_epoch{epoch}_mae{rounded_mae}.pth.tar",
                )
            )
    
    @classmethod
    def load(cls, path: str) -> Trainer:
        """Load trainer state_dict."""
        state = torch.load(path, map_location=torch.device('cpu'))
        model = CompositionModel.from_dict(state["model"])
        print(f"Loaded model params = {sum([p.numel() for p in model.parameters()]):}")
        state["trainer_args"].pop("model", None)        # drop model from trainer_args if present
        trainer = Trainer(model=model, **state["trainer_args"])
        trainer.model.to(trainer.device)
        trainer.optimizer.load_state_dict(state["optimizer"])
        trainer.scheduler.load_state_dict(state["scheduler"])
        trainer.training_history = state["training_history"]
        trainer.starting_epoch = len(trainer.training_history["train"])
        return trainer