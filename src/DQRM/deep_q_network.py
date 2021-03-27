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
        self.rm_network = []
        for i in range(n_rm_states):
            self.rm_network.append(
                    nn.Sequential(
                        nn.Linear(np.prod(input_dims), 64),nn.ReLU(),
                        nn.Linear(64, 64),nn.ReLU(),
                        nn.Linear(64, 64),nn.ReLU(),
                        nn.Linear(64, 64),nn.ReLU(),
                        nn.Linear(64, 64),nn.ReLU(),
                        nn.Linear(64, n_actions)
                        )
                    )
        # nn.ModuleList makes Pytorch read the created Python list as a
        # nn.Module list of objects
        self.rm_network = nn.ModuleList(self.rm_network)
        self.optimizer = optim.Adam(self.rm_network.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        
        # device nn.Module function already defined in Agent
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #self.to(self.device)

    def forward(self, state, rm_state):
        # Flatten the observation [BS, 7, 7, 3] --> [BS, 147]
        flat_state = state.view(state.size()[0], -1)
        # Initialize the actions tensor [BS, n_actions=7] 
        actions = T.zeros(state.size()[0], self.n_actions)
        for i in range(self.n_rm_states):
            # Loop over the rm_states and feed forward each observation to its
            # correspondant rmstate neural network
            try:
                rm_index = T.where(rm_state == i)[0]
                actions[rm_index] = self.rm_network[i](flat_state[rm_index])
            except:
                pass
        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))
