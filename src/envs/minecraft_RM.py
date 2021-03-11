# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 21:40:45 2021

@author: Carlos Marcos Torrejón
"""

import numpy as np
import matplotlib.pyplot as plt

""" Simple Minecraft environment """

" The environment state is a one dimensional array "
" which unique lement represents the player        "
" position                                         "
    
" In this simple implementation there are only two "
" possible tasks:                                  "
" 1st: Get wood                                    "
" 2nd: go to crafttable                            "

" The terminal state is set when the agent reaches " 
" the crafttable only after picking the wood.      "



class Minecraft(object):
    def __init__(self, args):
        self.m = args[0]
        self.n = args[1]
        self.resources = args[2]
        self.grid = np.zeros((self.m,self.n))
        
        # State space: position
        self.state_space = [self.m* self.n]
        self.rmstate_space = 2
        # 0: move down, 1:move up, 2:move left, 3:move right
        self.action_space = {0:-self.m, 1:self.m, 2:-1, 3:1}
        self.actions = [0,1,2,3]
        self.add_resources(self.resources)
        self.state = 0
        self.rmstate = 0
        self.reward = 0
        self.flags = [False, False] # Terminal flags
        self.done = False
        
    def add_resources(self, resources):
        # Places the resources on the grid
        # resources = [wood, crafttable]
        self.wood = [resources[0]]
        self.crafttable = [resources[1]]
        
    
    def isTerminalState(self, state_):
        # flags = [wood flag, crafttable flag]
        self.reward = -1
        if (state_ == self.resources[0]) and (self.rmstate == 0):
            self.flags[0] = True
            self.rmstate = 1
            self.reward = 0
        elif (state_ == self.resources[1]) and (self.rmstate == 1):
            self.flags[1] = True
            self.reward = 0
        if self.flags[0] and self.flags[1]:
            return True
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
        if (state_ < 0) or (state_ >= self.m*self.n):
            state_ = self.state
        return state_
    
    def step(self, action):
        # Returns [new state, terminal flag, None]
        # after performing step in environment given an action
        state_ = self.state + self.action_space[action]
        state_ = self.offGridcheck(state_) 
        self.done = self.isTerminalState(state_)
        self.state = state_
        return self.state, self.rmstate, self.reward, self.done, None
    
    def actionSpaceSample(self):
        # Sample random action for epsilon greedy policy learn
        return np.random.choice(self.actions)
            
    def reset(self):
        # Reset environment to start episode
        self.state = 0
        self.rmstate = 0
        self.grid = np.zeros((self.m, self.n))
        self.flags = [False, False]
        self.done = False
        return self.state, self.rmstate
    
    def render(self):
        # Render simple ASCII representation of the current env state
        woodcoords = self.getCoordinates(self.wood[0])
        craftcoords = self.getCoordinates(self.crafttable[0])
        playercoords = self.getCoordinates(self.state)
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
    

    def give_rmstate(self, rmstate, state_):
        # Define RM state transition function
        if (state_ == self.resources[0]) and (rmstate == 0):
            rmstate = 1
        elif (state_ == self.resources[1]) and (rmstate == 1):
            rmstate = 2 # Terminal RM state
        return rmstate

    def give_reward(self, rmstate, rmstate_):
        # Define reward transition function
        # Only give reward not equal to 0 if there is RM transition
        if rmstate != rmstate_:
            return 0
        return -1

