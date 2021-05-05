import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PPO.PPOutils import Storage, orthogonal_init

class PPOFlatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class PPOEncoder(nn.Module):
  def __init__(self, in_channels, input_dims, feature_dim):
    super().__init__()
    self.convlayers = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU(),
        )
    
    conv_dims = self.calculate_conv_output_dims(input_dims)
    self.linear = nn.Sequential(
        PPOFlatten(),
        nn.Linear(in_features=conv_dims, out_features=feature_dim),
        nn.ReLU()
        )
        
    self.apply(orthogonal_init)

  def calculate_conv_output_dims(self, input_dims):
      state = torch.zeros(1, *input_dims)
      dims = self.convlayers(state)
      return int(np.prod(dims.size()))

  def forward(self, x):
    x = self.convlayers(x)
    x = self.linear(x)
    return x

""" Declaration of policy and value functions of actor-critic method """

class PPOPolicy(nn.Module):
  def __init__(self, encoder, feature_dim, num_actions):
    super().__init__()
    self.encoder = encoder
    self.policy = orthogonal_init(nn.Linear(feature_dim, num_actions), gain=.01)
    self.value = orthogonal_init(nn.Linear(feature_dim, 1), gain=1.)

  def act(self, x):
    with torch.no_grad():
      #x = x.cuda().contiguous()
      dist, value = self.forward(x)
      action = dist.sample()
      log_prob = dist.log_prob(action)
    
    return action.cpu(), log_prob.cpu(), value.cpu()

  def forward(self, x):
    #print("input shape: ", x.shape)
    x = self.encoder(x)
    #print("afterencoder shape: ", x.shape)
    logits = self.policy(x)
    value = self.value(x).squeeze(1)
    dist = torch.distributions.Categorical(logits=logits)

    return dist, value
