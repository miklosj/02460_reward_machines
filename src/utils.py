import gym
from gym_minigrid.wrappers import *
import numpy as np


def Get_door_state(obs)
    """
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
    """
    ## Find objects position
    special_tuples = []
    for i in range(x_dim):
        for j in range(y_dim):
    return special_tuples
