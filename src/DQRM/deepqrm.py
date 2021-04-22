import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, n_rm_states,chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.n_actions = n_actions
        self.n_rm_states = n_rm_states

        # each self.rm_network index will correspond to a separate neural network
        # for learing an specific "n" rm state policy
        self.rm_network = nn.Sequential(
                    nn.Linear(input_dims, 512),nn.ReLU(),
                    nn.Linear(512, 512), nn.ReLU(),
                    nn.Linear(512, n_actions)
                    )
                    
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        # device nn.Module function already defined in Agent
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        actions = self.rm_network(state)        
        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
