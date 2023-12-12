"""Modules for normalizing flow model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal

from typing import Tuple

class SimpleAffine(nn.Module):
    """Implement a simple affine with log_scale and shift.
    
    Args:
        dim (int): feature dimention of the input data.
    """
    def __init__(
        self, dim: int = 2,
    ) -> None:
        """Initialize SimpleAffine with log_scale and shift."""
        super(SimpleAffine, self).__init__()
        self.dim = dim
        # Initialze the training parameter in one layer of normalizing flow model.
        self.a = nn.Parameter(torch.zeros(self.dim))    # scale
        self.b = nn.Parameter(torch.zeros(self.dim))    # shift
    
    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the affine transformation and log determinant of the Jacobian.
        Forward means from Gaussian space to data space.
        """
        y = torch.exp(self.a) * x + self.b
        det_jac = torch.exp(self.a.sum())
        log_det_jac = torch.ones(y.shape[0]) * torch.log(det_jac)           # torch.ones for matching batch size.
        return y, log_det_jac
    
    def inverse(
        self, y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the inverse affine transformation and log determinant of 
        the inverse Jacobian. Inverse means from data space to Gaussian space."""
        x = (y - self.b) / torch.exp(self.a)
        det_jac = 1 / torch.exp(self.a.sum())
        inv_log_det_jac = torch.ones(y.shape[0]) * torch.log(det_jac)       # torch.ones for matching batch size.
        return x, inv_log_det_jac

class StackSimpleAffine(nn.Module):
    """Implement the stack of simple affine transformations."""
    def __init__(
        self, transform: nn.ModuleList, dim: int = 2,
    ) -> None:
        """Initialize StackSimpleAffine with transform and dimensions."""
        super(StackSimpleAffine, self).__init__()
        self.dim = dim
        self.transform = nn.ModuleList(transform)
        self.distribution = MultivariateNormal(
            torch.zeros(self.dim), torch.eye(self.dim)
        )
    
    def log_probability(
        self, x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the log probability.
        
        Args:
            x (Tensor): sample from data space.
        """
        log_prob = torch.zeros(x.shape[0])
        for transform in reversed(self.transform):
            x, inv_log_det_jac = transform.inverse(x)
            log_prob += inv_log_det_jac
        log_prob += self.distribution.log_prob(x)
        return log_prob
    
    def resample(
        self, num_samples: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from the Guassian space in trained normalizing flow.
        
        Args:
            num_samples (int): number of samples. 
        """
        x = self.distribution.sample((num_samples,))
        log_prob = self.distribution.log_prob(x)
        for transform in self.transform:
            x, log_det_jac = transform.forward(x)
            log_prob += log_det_jac
        return x, log_prob
    
class RealNVPNode(nn.Module):
    """Implement RealNVP Node.
    
    Args:
        mask (Tensor): mask for the affine transformation.
        hidden_size (int): hidden size of the affine transformation.
    
    Mathematics:
        Given a D-dimensional input x, and d < D,
            y[1:d] = x[1:d]
            y[d+1:D] = x[d+1:D] dot exp(s(x[1:d])) + t(x[1:d])
        This specially designed transform has benefits:
        1) simple calculation of determinant of Jacobian
        2) s and t neural network can be flexble
    """
    def __init__(
        self, mask: torch.Tensor, hidden_size: int,
    ) -> None:
        """Initialize the real NVP Node with mask and hidden_size."""
        super(RealNVPNode, self).__init__()
        self.dim = len(mask)
        self.mask = nn.Parameter(mask, requires_grad=False)
        self.s_func = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=hidden_size), nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=self.dim)
        )
        self.scale = nn.Parameter(torch.Tensor(self.dim))
        self.t_func = nn.Sequential(
            nn.Linear(in_features=self.dim, out_features=hidden_size), nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.LeakyReLU(),
            nn.Linear(in_features=hidden_size, out_features=self.dim)
        )
        
    def forward(
        self, x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the transformation and log determinant of the Jacobian.
        Forward means from Gaussian space to data space."""
        x_mask = x*self.mask
        s = self.s_func(x_mask) * self.scale        # TODO: check whether need scale here.
        t = self.t_func(x_mask)
        
        y = x_mask + (1 - self.mask) * (x*torch.exp(s) + t)
        log_det_jac = ((1 - self.mask) * s).sum(-1)
        return y, log_det_jac
    
    def inverse(
        self, y: torch.Tensor,
    ) -> Tuple(torch.Tensor, torch.Tensor):
        y_mask = y * self.mask
        s = self.s_func(y_mask) * self.scale
        t = self.t_func(y_mask)
        
        x = y_mask + (1 - self.mask) * (y - t) * torch.exp(-s)
        inv_log_det_jac = ((1 - self.mask) * -s).sum(-1)
        return x, inv_log_det_jac
    
class RealNVP(nn.Module):
    """Implement Real NVP."""
    def __init__(
        self, masks: torch.Tensor, hidden_size: int,
    ) -> None:
        """Initialize RealNVP with masks and hidden_size."""
        super(RealNVP, self).__init__()
        self.dim = len(masks[0])
        self.hidden_size = hidden_size
        self.masks = nn.ParameterList([nn.Parameter(torch.Tensor(mask), requires_grad=False) for mask in masks])
        self.layers = nn.ModuleList([RealNVPNode(mask, self.hidden_size) for mask in self.masks])
        self.distribution = MultivariateNormal(torch.zeros(self.dim), torch.eye(self.dim))
        
    def log_probability(
        self, x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the log probability.
        
        Args:
            x (Tensor): sample from data space.
        """
        log_prob = torch.zeros(x.shape[0])
        
        for layer in reversed(self.layers):
            x, inv_log_det_jac = layer.inverse(x)
            log_prob += inv_log_det_jac
        log_prob += self.distribution.log_prob(x)
        return log_prob
    
    def resample(
        self, num_samples: int,
    ) -> torch.Tensor:
        """Sample from the Guassian space in trained normalizing flow.
        
        Args:
            num_samples (int): number of samples. """
        x = self.distribution.sample((num_samples,))
        log_prob = self.distribution.log_prob(x)
        for layer in self.layers:
            x, log_det_jac = layer.forward(x)
            log_prob += log_det_jac
        return x, log_prob
    
    def sample_each_step(self, num_samples):
        samples = []

        x = self.distribution.sample((num_samples,))
        samples.append(x.detach().numpy())

        for layer in self.layers:
            x, _ = layer.forward(x)
            samples.append(x.detach().numpy())

        return samples