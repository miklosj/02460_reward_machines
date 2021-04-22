import numpy as np
import torch as T
import torch.nn as nn
from DQRM.encoder import Encoder
from DQRM.deepqrm import DeepQNetwork
from DQRM.replay_memory import ReplayBuffer
from reward_machine import RewardMachine
from utils import *

class DQRMAgent(nn.Module):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        super(DQRMAgent, self).__init__()

        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
    
        temp = self.env_name.split("-")[1]
        self.name2indx_dict = {"DoorKey":1, "Unlock":2, "Empty":3}
        self.env_indx = self.name2indx_dict[temp]

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        self.rm = RewardMachine("minigrid_reward_machines.json", self.env_indx) # load Reward Machine
        self.n_rm_states = self.rm.n_rm_states

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions, self.n_rm_states)

        self.encoder = Encoder(self.lr, input_dims=self.input_dims,
                name = self.env_name+"_"+self.algo+"encoder", chkpt_dir = self.chkpt_dir)

        self.q_eval = []
        self.q_next = []
        self.flat_dims = self.calculate_conv_output_dims(input_dims)
        for i in range(self.n_rm_states):
            self.q_eval.append(DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.flat_dims, n_rm_states=self.n_rm_states,
                                    name=self.env_name+'_'+self.algo+'_q_eval'+str(i),
                                    chkpt_dir=self.chkpt_dir))

            self.q_next.append(DeepQNetwork(self.lr, self.n_actions,
                                    input_dims=self.flat_dims, n_rm_states=self.n_rm_states,
                                    name=self.env_name+'_'+self.algo+'_q_next'+str(i),
                                    chkpt_dir=self.chkpt_dir))
    
        # Loss function
        self.loss = nn.MSELoss()

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.encoder.forward(state)
        return int(np.prod(dims.size()))

    def choose_action(self, observation, rm_state):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.device)
            state = self.encoder.forward(state)
            actions = self.q_eval[rm_state].forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, u1, action, reward, state_, u2, done):
        self.memory.store_transition(state, u1, action, reward, state_, u2, done)

    def sample_memory(self):
        state, u1, action, reward, new_state, u2, done = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.device)
        u1s = T.tensor(u1).to(self.device)
        dones = T.tensor(done).to(self.device)
        rewards = T.tensor(reward).to(self.device)
        actions = T.tensor(action).to(self.device)
        states_ = T.tensor(new_state).to(self.device)
        u2s = T.tensor(u2).to(self.device)

        return states, u1s, actions, rewards, states_, u2s, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            for i in range(self.n_rm_states):
                self.q_next[i].load_state_dict(self.q_eval[i].state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        for i in range(self.n_rm_states):
            self.q_eval[i].save_checkpoint()
            self.q_next[i].save_checkpoint()

    def load_models(self):
        for i in range(self.n_rm_states):
            self.q_eval[i].load_checkpoint()
            self.q_next[i].load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        for i in range(self.n_rm_states):
            self.q_eval[i].optimizer.zero_grad()

        self.replace_target_network()

        states, u1s, actions, rewards, states_, u2s, dones = self.sample_memory()
        indices = np.arange(self.batch_size)

        flat_states = self.encoder(states).to(self.device)
        flat_states_ = self.encoder(states_).to(self.device)

        for i in range(self.n_rm_states):
            # Loop over n rm states to backpropagate through each neural network
            try:
                q_pred = self.q_eval[i].forward(flat_states)[indices, actions]
                q_next = self.q_next[i].forward(flat_states_).max(dim=1)[0]
 
                q_next[dones] = 0.0
                q_target = rewards[indices][i] + self.gamma*q_next

                loss = self.loss(q_target, q_pred).to(self.device)
                loss.backward()

                self.encoder.optimizer.step()

                self.q_eval[i].optimizer.step()
            except:
                pass

        self.learn_step_counter += 1
        self.decrement_epsilon()
