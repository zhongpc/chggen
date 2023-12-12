
"""Modules for generating dataset and plotting."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import inflect



class FlowDataset(Dataset):
    """Dataset for loading artificial dataset (multi-gaussian) to train normalizing flow.
    
    Args:
        num_samples (int): Number of samples to draw.
        seed (int): Random seed.
    """
    def __init__(
        self, 
        num_gaussian: int = 3,
        num_samples: int = 1000, 
        seed: int = 0,
    ) -> None:
        np.random.seed(seed)
        self.data = make_multi_gaussian(num_gaussian, num_samples)
        self.num_samples = len(self.data)
        return None
    
    def  __len__(self) -> int:
        """Get the number of samples."""
        return self.num_samples
    
    def __getitem__(self, index) -> torch.Tensor:
        """Get a sample."""
        return torch.from_numpy(self.data[index]).type(torch.FloatTensor)    
        
def make_multi_gaussian(
    num_gaussian: int = 3,
    num_samples: int = 1000,
) -> None:
    """Make artifial multiple gaussian for as data."""
    radius = 2
    angles = np.linspace(0, 2 * np.pi, num_gaussian, endpoint=False)
    cov = np.array([[0.1,0], [0,0.1]])
    results = []
    
    for angle in angles:
        results.append(
            np.random.multivariate_normal(
                radius * np.array([np.cos(angle), np.sin(angle)]), 
                cov,
                int(num_samples/num_gaussian),
            )
        )
    return np.random.permutation(np.concatenate(results, axis=0))



def plot_density(model, true_dist=None, num_samples=100, mesh_size=4.):
    x_mesh, y_mesh = np.meshgrid(np.linspace(- mesh_size, mesh_size, num=num_samples),
                                 np.linspace(- mesh_size, mesh_size, num=num_samples))

    cords = np.stack((x_mesh, y_mesh), axis=2)
    cords_reshape = cords.reshape([-1, 2])
    log_prob = np.zeros((num_samples ** 2))

    for i in range(0, num_samples ** 2, num_samples):
        data = torch.from_numpy(cords_reshape[i:i + num_samples, :]).float()
        log_prob[i:i + num_samples] = model.log_probability(data).cpu().detach().numpy()

    plt.scatter(cords_reshape[:, 0], cords_reshape[:, 1], c=np.exp(log_prob))
    plt.colorbar()
    if true_dist is not None:
        plt.scatter(true_dist[:, 0], true_dist[:, 1], c='orange', alpha=.05)
    plt.show()
    
def plot_each_step(model, num_samples=200):
    data = model.sample_each_step(num_samples)
    len_data = len(data)

    fig, axis = plt.subplots(2, int((len_data+1)/2), figsize=(15, 10),
                             sharex=True, sharey=True)
    p = inflect.engine()

    num_plot = 0
    for i in range(len_data):
        if i == round((len_data+1)/2):
            axis.flatten()[num_plot].axis('off')
            num_plot += 1

        d = data[i]
        ax = axis.flatten()[num_plot]
        if i == 0:
            title = 'Original data'
        else:
            title = p.ordinal(i) + ' layer'

        ax.scatter(d[:, 0], d[:, 1], alpha=.2)
        ax.set_title(title)
        num_plot += 1