"Modules for auxillary loss functions."

from __future__ import annotations

from torch import nn
import torch
import numpy as np
from typing import Dict, Tuple, Sequence


class KLDivergence(nn.Module):
    """Class for KL divergence."""
    def __init__(
        self, reduction: str = 'mean'
    ) -> None:
        """Initialize a KLLoss module."""
        super(KLDivergence, self).__init__()
        self.reduction = reduction
    
    def forward(
        self, mu, logvar,
    ) -> torch.Tensor:
        """Compute KL divergence."""
        if self.reduction == 'mean':
            return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        elif self.reduction == 'sum':
            return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        elif self.reduction == 'none':
            return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        else:
            raise NotImplementedError

class CompositionLoss(nn.Module):
    """Class for composition loss."""
    def __init__(
        self, reduction: str = 'mean'
    ) -> None:
        """Initialize a CompositionLoss module."""
        super(CompositionLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.reduction = reduction
    
    def forward(
        self, pred_comp, batch,
    ) -> torch.Tensor:
        """Compute composition loss using cross entrpy."""
        target_comp = batch.comp
        
        # Cross entropy loss.
        # loss = self.ce_loss(pred_comp, target_comp)
        loss = self.mse_loss(pred_comp, target_comp)
        return loss

class VolumeLoss(nn.Module):
    """Class for volume loss."""
    def __init__(
        self, reduction: str = 'mean'
    ) -> None:
        """Initialize a VolumeLoss module."""
        super(VolumeLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.reduction = reduction
    
    def forward(
        self, pred_v, batch,
    ) -> torch.Tensor:
        """Compute volume loss using mean squared error."""
        target_v = batch.ave_v.view(-1,1)
        # L2 loss.
        loss = self.mse_loss(pred_v, target_v)
        return loss