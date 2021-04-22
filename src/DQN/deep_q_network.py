import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        
        self.encoder = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4), nn.ReLU(),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), nn.ReLU(),
                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1), nn.ReLU()
                )

        conv_input_dims = self.calculate_conv_output_dims(input_dims)
        self.layers = nn.Sequential(
                    nn.Linear(conv_input_dims, 1024), nn.ReLU(),
                    nn.Linear(1024, 1024), nn.ReLU(),
                    nn.Linear(1024, n_actions)
                )

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        conv_state = state.view(state.size()[0], state.size()[3], state.size()[1], state.size()[2])
        dims = self.encoder(conv_state)
        return int(np.prod(dims.size()))

    def forward(self, state):
        # Reshape the image and feed forward it through encoder
        conv_state = state.view(state.size()[0], state.size()[3], state.size()[1], state.size()[2])
        conv_state = self.encoder(conv_state)
        # Flatten the observation
        flat_state = conv_state.view(conv_state.size()[0], -1)
        
        actions = self.layers(flat_state)
        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
