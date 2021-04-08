import numpy as np
from reward_machine import RewardMachine
from rm_utils import evaluate_dnf
from collections import defaultdict
from copy import deepcopy

class CRMParams:
	def __init__(self, gamma, eps_start, eps_dec, eps_end, n_actions, lr, env_name):
		self.gamma = gamma
		self.epsilon = eps_start
		self.eps_dec = eps_dec
		self.eps_end = eps_end
		self.n_actions = n_actions
		self.lr = lr

		# maps names of environents to index with which to look the RM in the json
		# NOTE: It would be better to just parse json using env_names, but we can do later
		env_name2index = {"DoorKey":1, "Unlock":2, "Empty":3}

		# Handles exception of reward machine not defined in .json for environement env_name
		try:
			self.env_idx = env_name2index[env_name.split("-")[1]]
		except:
			raise NotImplementedError("There is no reward machine defined in the .json for this environment. To add it, define it in the .json and include its index in the dictionary <env_name2_index>.")
			exit(1)

class CRMAgent(object):
	def __init__(self, params):
		self.gamma = params.gamma
		self.epsilon = params.epsilon
		self.eps_dec = params.eps_dec
		self.eps_end = params.eps_end
		self.n_actions = params.n_actions
		self.lr = params.lr

		self.rm = RewardMachine("minigrid_reward_machines.json", params.env_idx) # load Reward Machine

		self.Q = defaultdict(int) # Q network
		self.encoding = []
		self.label = ""

	def __map_encode(self, observation):
		"""
		Maps observations (3d array) to unique int "idx"
		so they can be used in the Q network as:
		Q[(idx, u1, action)] = q-value
		"""
		for idx, obs in enumerate(self.encoding):
			if (observation==obs).all():
				return idx
		self.encoding.append(observation)
		return len(self.encoding)

	def learn(self, observation, u1, action, reward, observation_, u2):
		# encode observation states
		observation = self.__map_encode(observation)
		observation_ = self.__map_encode(observation_)		

		# check if transition has been seen before, else assign q-value 0
		q_values = np.array([self.Q[(observation_, u1, a)] for a in range(self.n_actions)])
		a_max = np.argmax(q_values)
		self.Q[(observation, u1, action)] += self.lr * (reward + self.gamma*self.Q[(observation_, u2, a_max)] - self.Q[(observation, u1, action)])

		# decrement epsilon of greedy exploratory policy
		if (self.epsilon - self.eps_dec < self.eps_end):
			self.epsilon = self.eps_end
		else:
			self.epsilon = self.epsilon - self.eps_dec

	def choose_action(self, observation, u1):
		"""
		Uses eps-greedy exploratory policy
		"""
		observation = self.__map_encode(observation)
		if np.random.random() < self.epsilon:
			action = np.random.choice([i for i in range(self.n_actions)])

		else:
			q_values = np.array([self.Q[(observation, u1, a)] for a in range(self.n_actions)])
			action = np.argmax(q_values)
		return action