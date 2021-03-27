import numpy as np

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        print(input_shape)
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),
                                         dtype=np.float32)
        self.rmstate_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.new_rmstate_memory = np.zeros(self.mem_size, dtype=np.int64)

        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, rmstate, action, reward, state_, rmstate_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.rmstate_memory[index] = rmstate
        self.new_rmstate_memory[index] = rmstate_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        rmstates = self.rmstate_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        rmstates_ = self.new_rmstate_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, rmstates, actions, rewards, states_, rmstates_, terminal
