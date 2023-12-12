"""Main file for running normalizing flow."""

import numpy as np
import matplotlib.pyplot as plt
import torch
import tqdm
from flow import (
    RealNVPNode, 
    RealNVP,
)

# Training loop
def train(model, data, epochs = 100, batch_size = 64):
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters())
    
    losses = []
    with tqdm.tqdm(range(epochs), unit=' Epoch') as tepoch:
        epoch_loss = 0
        for epoch in tepoch:
            for batch_index, training_sample in enumerate(train_loader):
                log_prob = model.log_probability(training_sample)
                loss = - log_prob.mean(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss
            epoch_loss /= len(train_loader)
            losses.append(np.copy(epoch_loss.detach().numpy()))
            tepoch.set_postfix(loss=epoch_loss.detach().numpy())

    return model, losses

# Setting.
torch.manual_seed(2)
np.random.seed(0)

num_layers= 4       # Number of layers in flow model.
hidden_size = 32    # Number of hidden units in each MLP layer.
dim = 64            # Feature dimension of data.

# Data.
data = torch.randn(100,dim)   # 100 samples with dimension dim.

# Generate masks for reversible affine transformation.
masks = torch.zeros(num_layers, dim, dtype=torch.int)
for num_layer in range(num_layers):
    masks[num_layer, :dim//2] = 1 if num_layer % 2 == 0 else 0
    masks[num_layer, dim//2:] = 1   if num_layer % 2 == 1 else 0
print(masks)

# Initialize model.
NVP_model = RealNVP(masks, hidden_size)

if __name__ == '__main__':
    # Training.
    NVP_model, loss = train(NVP_model, data, epochs= 1000)
    
    # Resample.
    resample = NVP_model.resample(1000)[0].detach().numpy()
    print(resample.shape)