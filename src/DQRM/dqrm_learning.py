
import numpy as np
import torch as T
from DQRM.deep_q_network import DeepQNetwork
from DQRM.replay_memory import ReplayBuffer
import torch.nn as nn
import torch.optim as optim
from reward_machine import RewardMachine

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

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

        temp = self.env_name.split("-")[1]
        self.name2indx_dict = {"DoorKey":1, "Unlock":2, "Empty":3}
        self.env_indx = self.name2indx_dict[temp]

        self.rm = RewardMachine("minigrid_reward_machines.json", self.env_indx) # load Reward Machine
        self.n_rm_states = self.rm.n_rm_states
        
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,input_dims=self.input_dims,
                name=self.env_name+'_'+self.algo+'_q_eval', n_rm_states = self.n_rm_states,
                chkpt_dir=self.chkpt_dir)

        self.q_next = DeepQNetwork(self.lr, self.n_actions,input_dims=self.input_dims,
                name=self.env_name+'_'+self.algo+'_q_next', n_rm_states = self.n_rm_states,
                chkpt_dir=self.chkpt_dir)

        #self.optimizer = optim.RMSprop(list(self.q_eval.parameters()), lr=lr)

    def choose_action(self, observation, rmstate):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation],dtype=T.float).to(self.device)
            actions = self.q_eval.forward(state, rmstate)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def store_transition(self, state, rmstate, action, reward, state_, rmstate_, done):
        self.memory.store_transition(state, rmstate, action, reward, state_, rmstate_, done)

    def sample_memory(self):
        state, rmstate, action, reward, new_state, new_rmstate, done = \
                                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.device)
        rmstates = T.tensor(rmstate).to(self.device)
        rewards = T.tensor(reward).to(self.device)
        dones = T.tensor(done).to(self.device)
        actions = T.tensor(action).to(self.device)
        states_ = T.tensor(new_state).to(self.device)
        rmstates_ = T.tensor(new_rmstate).to(self.device)

        return states, rmstates, actions, rewards, states_, rmstates_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        for i in range(self.n_rm_states):
            self.q_eval.save_checkpoint()
            self.q_next.save_checkpoint()

    def load_models(self):
        for i in range(self.n_rm_states):
            self.q_eval.load_checkpoint()
            self.q_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return


        states, rmstates, actions, rewards, states_, rmstates_, dones = self.sample_memory()
        
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states, rmstates)[indices, actions].to(self.device)
        q_next = self.q_next.forward(states_, rmstates_).max(dim=1)[0].to(self.device)

        #rm_index = T.where(rmstates==(self.n_rm_states-1))[0]
        rm_index_ = T.where(rmstates_==(self.n_rm_states-1))[0]
        
        #q_pred[rm_index] = 1.0 
        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        
        self.decrement_epsilon()
