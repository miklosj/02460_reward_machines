import numpy as np
from reward_machine import RewardMachine
from rm_utils import evaluate_dnf
from collections import defaultdict
from copy import deepcopy

class QRMParams:
	def __init__(self, gamma, eps_start, eps_dec, eps_end, n_actions, lr, env_idx=1):
		self.gamma = gamma
		self.epsilon = eps_start
		self.eps_dec = eps_dec
		self.eps_end = eps_end
		self.n_actions = n_actions
		self.lr = lr
		self.env_idx = env_idx

class QRMAgent(object):
	def __init__(self, params):
		self.gamma = params.gamma
		self.epsilon = params.epsilon
		self.eps_dec = params.eps_dec
		self.eps_end = params.eps_end
		self.n_actions = params.n_actions
		self.lr = params.lr

		self.Q = defaultdict(int) # Q newtork
		self.rm = RewardMachine("minigrid_reward_machines.json", params.env_idx) # load Reward Machine
		self.encoding = []
		self.label = ""

	def __map_encode(self, observation):
		"""
		Maps observations (3d array) to unique int "idx"
		so they can be used in the Q network as:
		Q[(idx, action)] = q-value
		"""
		for idx, obs in enumerate(self.encoding):
			if (observation==obs).all():
				return idx
		self.encoding.append(observation)
		return len(self.encoding)

	def learn(self, observation, action, reward, observation_, done):
		# encode observation states
		observation = self.__map_encode(observation)
		observation_ = self.__map_encode(observation_)		

		# check if transition has been seen before, else assign q-value 0
		q_values = np.array([self.Q[(observation_,a)] for a in range(self.n_actions)])
		a_max = np.argmax(q_values)

		self.Q[(observation, action)] += self.lr * (reward + self.gamma*self.Q[(observation_,a_max)] - self.Q[(observation, action)])
		# decrement epsilon of greedy exploratory policy
		if (self.epsilon - self.eps_dec < self.eps_end):
			self.epsilon = self.eps_end
		else:
			self.epsilon = self.epsilon - self.eps_dec

	def choose_action(self, observation):
		"""
		Uses eps-greedy exploratory policy
		"""
		observation = self.__map_encode(observation)
		if np.random.random() < self.epsilon:
			action = np.random.choice([i for i in range(self.n_actions)])

		else:
			q_values = np.array([self.Q[(observation,a)] for a in range(self.n_actions)])
			action = np.argmax(q_values)
		return action