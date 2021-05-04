import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, n_rm_states):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)
        self.u1_memory = np.zeros((self.mem_size, n_rm_states), dtype=np.float32)
        self.u2_memory = np.zeros((self.mem_size, n_rm_states), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, u1, action, reward, state_, u2, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.u1_memory[index] = u1
        self.u2_memory[index] = u2
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        u1s = self.u1_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        u2s = self.u2_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, u1s, actions, rewards, states_, u2s, terminal
