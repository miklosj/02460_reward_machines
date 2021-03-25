import gym
from gym_minigrid.wrappers import *
import numpy as np

"""
def Get_door_state(obs):
    
    OBJECT_TO_IDX = {
    'unseen'        : 0,
    'empty'         : 1,
    'wall'          : 2,
    'floor'         : 3,
    'door'          : 4,
    'key'           : 5,
    'ball'          : 6,
    'box'           : 7,
    'goal'          : 8,
    'lava'          : 9,
    'agent'         : 10,
    }
    
    ## Find objects position
    special_tuples = []
    for i in range(x_dim):
        for j in range(y_dim):
    return special_tuples
"""

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

    #print(f"State label input {state_label}")
    #print(f"in past: {symbols_past}")
    #print(f"in new: {symbols_new}")
    dict_symbols = {4:"o", 5:"k", 8:"g"}

    # lost symbols in the new iteration of the agent
    lost_symbols = symbols_past.difference(symbols_new)
    #print(f"lost symbols: {lost_symbols}")
    if len(lost_symbols) != 0:
        for symbol in lost_symbols: # (a,a,a)
            #print(f"innerloop_past {symbol}")
            obj_id, state_id = symbol

            # get type of object
            if dict_symbols[obj_id] == "k":
                state_label+="k" # key was picked in new iteration

            if (dict_symbols[obj_id] == "o") & (state_id in [1,2]): # door was locked/closed
                state_label += "o"

            if (dict_symbols[obj_id] == "o") & (state_id == [0]): # door was open
                state_label = state_label.replace("o", "") # door was closed in new iteration

    new_symbols = symbols_new.difference(symbols_past)
    #print(f"gained symbols: {new_symbols}")
    if len(new_symbols) != 0:
        for symbol in new_symbols: # (a,a,a)
            #print(f"innerloop_new {symbol}")
            obj_id, state_id = symbol

            # append its associated symbol, as it is a lost
            state_label = state_label.replace(dict_symbols[obj_id],"")
    #print(f"State label output {state_label}")
    return state_label