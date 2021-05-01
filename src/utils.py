import gym
from gym_minigrid.wrappers import *
import numpy as np

""" Class that will handle the reward machine states """
class logical_symbols:
    def __init__(self):
        self.goal_pos = [-1,-1]
        self.agent_pos = [-2,-2]

    def get_special_symbols(self, obs):
        special_symbols = []
        xdim, ydim, _ = obs.shape
        self.agent_pos = [-2,-2]
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

""" Necessary function for reshaping the images """
def reshape_obs(obs):
    "[h,w,n_chan] --> [n_chan,h,w]"
    dims = obs.shape
    obs = obs.reshape(dims[2],dims[0],dims[1])
    return obs

""" One hot encode the Reward Machine state """
def rm_state_onehot(rm_state, n_rm_states):
    rm_state_oh = np.zeros(n_rm_states)
    rm_state_oh[rm_state] = 1.0
    return rm_state_oh
