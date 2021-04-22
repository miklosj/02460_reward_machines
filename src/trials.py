import gym
from gym_minigrid.wrappers import *
from copy import deepcopy

env = gym.make("MiniGrid-DoorKey-5x5-v0")
envListener = deepcopy(env)

envListener = FullyObsWrapper(envListener)
envListener = ImgObsWrapper(envListener)
env = FullyObsWrapper(env)
env = ImgObsWrapper(env)

envListener.seed(0)
env.seed(0)

obsListener = envListener.reset()
obs = env.reset()
