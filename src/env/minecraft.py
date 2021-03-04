import numpy as np
import matplotlib.pyplot as plt

""" Simple Minecraft environment """

" The environment state is a two dimensional array "
" shaped [grid position, task]                     "
    
" In this simple implementation there are only two "
" possible tasks:                                  "
" 1st: Get wood                                    "
" 2nd: go to crafttable                            "

" The terminal state is set when the agent reaches " 
" the crafttable only after picking the wood.      "


class Minecraft(object):
    def __init__(self, m, n, resources):
        # Define gridsize m*n
        self.grid = np.zeros((m,n))
        self.m = m 
        self.n = n
        
        # State space: position, wood
        self.state_space = [self.m*self.n, 1]
        # 0: move down, 1:move up, 2:move left, 3:move right
        self.action_space = {0:-self.m, 1:self.m, 2:-1, 3:1}
        self.actions = [0,1,2,3]
        self.add_resources(resources)
        self.state = [0, 0]
        
    def add_resources(self, resources):
        # Places the resources on the grid
        # resources = [wood, crafttable]
        self.wood = [resources[0]]
        self.crafttable = [resources[1]]
        
    def isTerminalState(self):
        # flags = [wood flag, crafttable flag]
        if self.flags[0] & self.flags[1]:
            return True
        else:
            return False
        
    def getCoordinates(self, position):
        # Convert 1D position array to x-y coordinates
        # (only used in render function)
        x = position // self.m
        y = position % self.n
        return [x, y]
    
    def offGridcheck(self, state_):
        # Check if the new state is out of the grid, if
        # it is return to the previous state
        if (state_ < 0) or (state_) >= self.m*self.n:
            state_ = self.state[0]
        return state_
    
    def step(self, action):
        # Returns [new state, reward, terminal flag, None]
        # after performing step in environment given an action
        reward = 0
        state_ = self.state[0] + self.action_space[action]
        state_ = self.offGridcheck(state_)
        if (state_ == self.wood[0] and self.state[1] == 0):
            self.state[1] = 1
            self.flags[0] = True
        elif (state_ == self.crafttable[0] and self.state[1] == 1):
            reward += 1
            self.flags[1] = True
        else:
            reward += 0
        self.state[0] = state_            
        return self.state, reward, self.isTerminalState(), None
    
    def actionSpaceSample(self):
        # Sample random action for epsilon greedy policy learn
        return np.random.choice(self.actions)
            
    def reset(self):
        # Reset environment to start episode
        self.state = [0, 0]
        self.grid = np.zeros((self.m, self.n))
        self.flags = [False, False]
        return self.state
    
    def render(self):
        # Render simple ASCII representation of the current env state
        woodcoords = self.getCoordinates(self.wood[0])
        craftcoords = self.getCoordinates(self.crafttable[0])
        playercoords = self.getCoordinates(self.state[0])
        print(woodcoords, craftcoords, playercoords)
        
        for row in range(self.m):
            for col in range(self.n):
                coords = [row, col]
                if coords == playercoords:
                    print("P",end="\t")
                if coords == craftcoords:
                    print("C", end="\t")
                if coords == woodcoords:
                    print("W", end="\t")
                else:
                    print("-", end="\t")
            print("\n")
        
