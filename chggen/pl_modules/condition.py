"""Modules for property-guided generation"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import Softmin
from typing import Any, List
import numpy as np

# import M3GNet

class Predictor(nn.Module):
    """Predictor for property from structure."""
    
    def __init__(self) -> None:
        super(Predictor, self).__init__()
        return None
    
    def forward(
        self, 
        structure: Any,
    ) -> float:
        # TODO: Process in batch.
        """Predict property from strcture.
        
        Args:
            structure (): input structure to predict property. 
            
        Returns:
            prop (float): predicted property for the structure.
        """
        # TODO: Add a pre-trained property predictor here.
        return structure.mean()
        
class Classifier(nn.Module):
    """Convert the regressor for property to classifier to obtain probability.
    
    Args:
        prop_given (float): conditional property given as hyper-parameter.
        bins (int): number of discretized property sub-intervals to set.
        interval (list): overall coverage interval of the input property. 
        
    Mathematics:
        q(c|R) = exp(-|predictor(R) - disc_c_i|) / sum_i(exp(-|predictor(R) - disc_c_i|))
    """
    
    def __init__(
        self,
        prop_given: torch.Tensor, # a scalar
        bins: int = 200,                                 # Discritize 20 properties.
        interval: List[float, float] = [-0.1, 0.1],      # From -0.1 eV to 0.1 eV.
    ) -> None:                                           # TODO: Rescale E_hull for interval here?
        """Initialize Classifier with bins and interval."""
        super(Classifier, self).__init__()

        disc_props_bins = torch.linspace(interval[0], interval[1], bins, device= prop_given.device)
        self.disc_props = (disc_props_bins[1:] + disc_props_bins[:-1])/2

        # Coarse grain the given property to nearest discretized property.
        self.prop_given_id = torch.argmin(torch.abs(prop_given - self.disc_props))
        self.prop_given = self.disc_props[self.prop_given_id]        
        return None
    
    def forward(
        self,
        prop_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate the probability of obtaining the given property with the structure given.
        
        NOTE: prop_pred has the shape like (B,1), where B is the number of structure in the batch.
        """
        self.disc_props.to(device= prop_pred.device)
        probs = Softmin(dim=-1)(torch.abs(prop_pred - self.disc_props))
        prob = probs[:,self.prop_given_id].unsqueeze(dim=1)
        return prob

if __name__ == "__main__":

    predictor = Predictor()
    classifier = Classifier(prop_given = 0.0)
    
    # print('-'*10, 'Testing', '-'*10)
    # print(classifier(prop_pred = -0.1))
    # print(classifier(prop_pred = 0.04))
    # print(classifier(prop_pred = 0.05))
    # print(classifier(prop_pred = 0.06))
    # print(classifier(prop_pred = 0.1))
    
    print('-'*10, 'Using', '-'*10)
    structure = torch.randn(size=(10,3), requires_grad=True)
    prop_pred = predictor(structure)
    prop_pred = torch.Tensor([[-0.1],[0.0],[0.1]])
    print(prop_pred)
    print(classifier(prop_pred))
    
# NOTE: 1. whether need whole coverage of E_hull? Can we only map, say, -100meV to -0.1 and 100meV to 0.1 and leave outliers linearly scale accordingly.
#       2. Unsymmetric probability to prop_given here, like prop_pred = 0.04 and prop_pred = 0.06 above. How about Gaussian-like function broadening? (Symmetric probability to prop_given)