import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, n_rm_states, chkpt_dir, n_neurons=512):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.n_actions = n_actions
        self.n_rm_states = n_rm_states
        self.n_neurons = n_neurons

        self.encoder = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU()
                )


        # each self.rm_network index will correspond to a separate neural network
        # for learing an specific "n" rm state policy
        conv_input_dims = self.calculate_conv_output_dims(input_dims)
        self.connecting_layer = nn.Sequential(
                nn.Linear(conv_input_dims, 256),
                nn.ReLU()
                )
        
        self.rm_network = nn.Sequential(
                nn.Linear(256+self.n_rm_states, n_neurons), nn.ReLU(),
                nn.Linear(n_neurons, n_neurons), nn.ReLU(),
                nn.Linear(n_neurons, n_actions)
                )
        
        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        
        # device nn.Module function already defined in Agent
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        conv_state = state.view(state.size()[0], state.size()[3], state.size()[1], state.size()[2])
        dims = self.encoder(conv_state)
        return int(np.prod(dims.size()))

    def forward(self, state, rm_state):
        # reshape: [BS, height, width, channels] --> [BS, channels, height, width]
        conv_state = state.view(state.size()[0], state.size()[3], state.size()[1], state.size()[2])
        conv_state = self.encoder(conv_state)
        # Flatten the observation [BS, x, y, z] --> [BS, x*y*z] + extra dim for rm_state one hot encode
        flat_state = conv_state.view(conv_state.size()[0], -1)
        # Feed forward to connecting layer
        flat_state = self.connecting_layer(flat_state)
        # Append rm_state one hot encoding
        flat_state = T.cat((flat_state,rm_state), dim=1)
        actions = self.rm_network(flat_state)
        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
