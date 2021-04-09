import gym
from gym_minigrid.wrappers import *
import numpy as np

class logical_symbols:
    def __init__(self):
        self.goal_pos = [-1,-1]
        self.agent_pos = [-2,-2]

    def get_special_symbols(self, obs):
        special_symbols = []
        xdim, ydim, _ = obs.shape
        self.agent_pos = [-1,-1]
        for i in range(xdim):
            for j in range(ydim):
                obj_id, color_id, state_id = obs[i,j,:]
                if obj_id in [4,5,10,8]:
                    if obj_id == 10:
                        self.agent_pos = [i, j]
                    if self.agent_pos == self.goal_pos:
                        return True
                    if obj_id == 8:
                        self.goal_pos = [i, j]
                    special_symbols.append((obj_id, state_id))
        return set(special_symbols)

    def return_symbol(self, symbols_past, symbols_new, state_label):
        if (symbols_new==True) or (symbols_past==True) or ("g" in state_label):
            state_label += "g"
            return state_label

        dict_symbols = {4:"d", 5:"k", 8:"g", 10:"a"}

        lost_symbols = symbols_past.difference(symbols_new)
        if len(lost_symbols) != 0:
            for symbol in lost_symbols:
                obj_id, state_id = list(symbol)

                #get type of object
                if dict_symbols[obj_id] == "k":
                    state_label += "k" #key is picked up
                if (dict_symbols[obj_id] == "d") and (state_id in [1,2]):
                    state_label += "ko" #open the door
                if (dict_symbols[obj_id] == "d") and (state_id == 0):
                    state_label = state_label.replace("o", "")

        gain_symbols = symbols_new.difference(symbols_past)
        if len(gain_symbols) != 0:
            for symbol in gain_symbols:
                obj_id, state_id = list(symbol)

                if dict_symbols[obj_id] == "k":
                    state_label = state_label.replace("k","") #key is dropped

        return state_label

def get_special_symbols(obs):
    """
    returns list of special symbols in the map
    4 - door, 5 - key, 8 - goal

    Parameters:
    --------------
    obs: 3d array of dimension x_dim, y_dim, 3 (tuple)
        array of objects defined by tuple (object_idx, color_idx, state_idx)

    Return:
    -------------
    special_symbols: set of tuples (int, int)
        special symbols (object_idx, state_idx)
    """
    special_symbols = []
    xdim, ydim, _ = obs.shape
    for i in range(xdim):
        for j in range(ydim):
            obj_id, color_id, state_id = obs[i,j,:]
            if obj_id in [4,5,8]:
                special_symbols.append((obj_id, state_id))
    return set(special_symbols) # [(a,a), (b,b), ...]

def return_symbol(symbols_past, symbols_new, state_label):
    """
    Given two sets of special symbols compares symbols present in the past no longer present (lost symbols)
    and symbols present only now and not the past (gained_symbols/new_symbols), then appends/removes the char symbols
    to the state_label.

    Parameters:
    --------------
    symbols_past: set of tuples (int, int)
        represent list of special objects/symbols present in the past
    symbols_new: set of tuples (int, int)
        represent list of special objects/symbols present now
    state_label: string
        sequence of char symbols, true_propositions in the game
        e.g if agent has picked key then state_label: ""-> "k"
        e.g if agent has dropped key then state_label: "k"-> "", as k is no longer true
    """
    print(f"symbols before: {state_label}")
    dict_symbols = {4:"o", 5:"k", 8:"g"}

    # lost symbols in the new iteration of the agent
    lost_symbols = symbols_past.difference(symbols_new)
    print(f"lost symbols: {lost_symbols}")
    if len(lost_symbols) != 0:
        for symbol in lost_symbols: # (a,a)
            obj_id, state_id = symbol

            # get type of object
            if dict_symbols[obj_id] == "k":
                state_label+="k" # key was picked in new iteration

            if (dict_symbols[obj_id] == "o") & (state_id in [1,2]): # door was locked/closed
                state_label += "o"

            if (dict_symbols[obj_id] == "o") & (state_id == [0]): # door was open
                state_label = state_label.replace("o", "") # door was closed in new iteration

            if dict_symbols[obj_id] == "g": # if goal disappear then agent is on top
                state_label+="g"

    new_symbols = symbols_new.difference(symbols_past)
    print(f"new symbols: {new_symbols}")
    if len(new_symbols) != 0:
        for symbol in new_symbols: # (a,a)
            obj_id, state_id = symbol
             
            # get type of object
            if dict_symbols[obj_id] == "k":
               state_label = state_label.replace(dict_symbols[obj_id],"") # key was dropped (it appeared in the map)

            if (dict_symbols[obj_id] == "o") & (state_id in [1,2]): # door is locked/closed
                state_label = state_label.replace(dict_symbols[obj_id],"")

            if (dict_symbols[obj_id] == "o") & (state_id == [0]): # door is open now
                state_label += "o"
    print(f"symbols after: {state_label}")
    return state_label
