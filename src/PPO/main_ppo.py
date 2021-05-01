import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PPO.PPOutils make_env, Storage, orthogonal_init
from PPO.ppo_learning import *
""" Hyperparameters """
# Hyperparameters
total_steps = 8e6
num_envs = 32
num_levels = 10
num_steps = 256
num_epochs = 3
batch_size = 512
eps = .2
grad_eps = .5
value_coef = .5
entropy_coef = .01

""" Define network and optimizer """

# Define network
in_channels,_,_ = env.observation_space.shape
feature_dim = 256
num_actions = env.action_space.n
encoder = Encoder(in_channels=in_channels,feature_dim=feature_dim)
policy = Policy(encoder=encoder,feature_dim=feature_dim,num_actions=num_actions)
policy.cuda()

# Define optimizer
optimizer = torch.optim.Adam(policy.parameters(), lr=5e-4, eps=1e-5)

# Define temporary storage
# we use this to collect transitions during each iteration
storage = Storage(
    env.observation_space.shape,
    num_steps,
    num_envs
)

""" Run training """
obs = env.reset()
reward_storage = 0
std_storage = 0
step_storage = 0

step = 0
while step < total_steps:

  # Use policy to collect data for num_steps steps
  policy.eval()
  for _ in range(num_steps):
    # Use policy
    action, log_prob, value = policy.act(obs)
    
    # Take step in environment
    next_obs, reward, done, info = env.step(action)

    # Store data
    storage.store(obs, action, reward, done, info, log_prob, value)
    
    # Update current observation
    obs = next_obs

  # Add the last observation to collected data
  _, _, value = policy.act(obs)
  storage.store_last(obs, value)

  # Compute return and advantage
  storage.compute_return_advantage()

  # Optimize policy
  policy.train()
  for epoch in range(num_epochs):

    # Iterate over batches of transitions
    generator = storage.get_generator(batch_size)
    for batch in generator:
      b_obs, b_action, b_log_prob, b_value, b_returns, b_advantage = batch

      # Get current policy outputs
      new_dist, new_value = policy(b_obs)
      new_log_prob = new_dist.log_prob(b_action)

      # Clipped policy objective      
      ratio = torch.exp(new_log_prob - b_log_prob)
      surr1 = ratio * b_advantage
      surr2 = torch.clamp(ratio, 1.0 - eps, 1.0 + eps) * b_advantage
      pi_loss = -torch.min(surr1, surr2).mean()

      # Clipped value function objective
      clipped_value = b_value + (new_value-b_value).clamp(min=-eps, max=eps)
      vf_loss = torch.max((new_value-b_returns).pow(2), (clipped_value - b_returns).pow(2))
      value_loss = vf_loss.mean()

      # Entropy loss
      entropy_loss = new_dist.entropy().mean()

      # Backpropagate losses
      loss = pi_loss + value_coef*value_loss - entropy_coef*entropy_loss
      loss.backward()

      # Clip gradients
      torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

      # Update policy
      optimizer.step()
      optimizer.zero_grad()

