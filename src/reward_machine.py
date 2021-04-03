from rm_utils import evaluate_dnf, parse_json_reward_machine
import time

"""
Class that assigns constant reward for any to pair of states and
action.

Note: This could be generalized so that it inherits from a common
blueprint class RewardFunction sot that we can then extend for different
Reward functions. But since we are only using constant reward its ok for now.
"""
class ConstantRewardFunction:
	def __init__(self, c):
		self.c = c
		self.type = "constant"

	def get_type(self):
		return self.type

	def get_reward(self):
		return self.c

"""
Main RewardMachine Class, constructor loads everything for the environment
selected in <file_rm> and for the env specified by the index <idx_rm>

Available methods:
- is_terminal(u1): returns boolean
- get_next_state(u1, true_props): returns next state u2
- get_reward(u1, u2, s1, action, s2): returns constant reward as given by delta_r[u1][u2]
"""
class RewardMachine:
	# set U of <u1, u2, delta_u, delta_r>
	def __init__(self, file_rm, idx_env):
		self.U = []			# list of machine reward states
		self.u0 = None		# initial reward machine state
		self.delta_u = {}	# transition function for states
		self.delta_r = {}	# transition function for rewards

		self.unique_states = [] # list to store unique rm_states
		self.file_rm = file_rm # file path to environment.json rm definitions
		self.idx_env= idx_env  # index of task RM to parse

		self.T = set()				 # set of terminal states
		self.__load_reward_machine() # load reward machine info

	def __is_terminal(self, u1):
		"""
		Private function to be used to fill in the terminal states self.T when called in
		__load_reward_machines

		state is terminal if any policy is optimal for that node, two cases are considered
			1. No transition is defined for u1 (len(self.delta_u[u1] == 0))
			2. There is only one "True" self-pointing loop and reward from u1 into itself is constant
		"""
		# first case (not transitions defined)
		if (len(self.delta_u[u1]) == 0):
			return True

		"""
		second case (self-pointing true constant)
		NOTE: 3 conditions are checked here:
			1. check only one transition possible (len(self.delta_u[u1]) == 1)
			2. check if transtition is always True
		"""
		if (len(self.delta_u[u1]) == 1):
			u2 = list(self.delta_u[u1].keys())[0]
			if (self.delta_u[u1][u2] == "True"):
				return True

		# if none of the previous conditions where triggered then state is not terminal
		return False

	def __add_transition(self, u1, u2, state_transition, reward_transition):
		# append the new rm states to unique_states list
		if u1 not in self.unique_states:
			self.unique_states.append(u1)
		if u2 not in self.unique_states:
			self.unique_states.append(u2)

		# add <u1, u2, state_transition, reward_transition>
		# add state if not in state space already
		if (u1, u2) not in self.U:
			self.U.append((u1, u2))

		## add state-transition to delta_u dict map
		# if u1 key not yet defined then initialize it empty
		if u1 not in self.delta_u:
			self.delta_u[u1] = {}
		self.delta_u[u1][u2] = state_transition

		## add reward-transition to delta_r dict map
		# if u1 key not yet defined then initilaize it empty
		if u1 not in self.delta_r:
			self.delta_r[u1] = {}
		self.delta_r[u1][u2] = reward_transition

	def __load_reward_machine(self):
		"""
		dictionary with parsed content
		- reward_machine_dict["env_id"]: index of task
		- reward_machine_dict["initial_state"]: u0, initial state of RM
		- reward_machine_dict["rm_states"]: list of <u1, u2, delta_u, delta_r>
		"""
		# load the data for the reward machine
		reward_machine_dict = parse_json_reward_machine(self.file_rm, self.idx_env)

		## Setting the Deterministic finite automata (DFA)
		# adding initial state
		self.u0 = reward_machine_dict["initial_state"]
        	# adding total number of reward machine states
		self.n_rm_states = reward_machine_dict["n_rm_states"]
       
        # adding transitions
		# NOTE: *eval() its so that it behaves as a tuple argument
		for rm_state in reward_machine_dict["rm_states"]:
			# {"U":0, "u0":0, "delta_u":"!c&!*", "delta_r": "ConstantRewardFunction(0)"}
			# __add_transition(self, u1, u2, state_transition, reward_transition)
			self.__add_transition(rm_state["u1"], rm_state["u2"], rm_state["delta_u"], eval(rm_state["delta_r"]))

		# adding terminal states
		for u1 in self.delta_u:
			if self.__is_terminal(u1):
				self.T.add(u1)
		"""
		for cases where there is no defined transition a self pointing dummy transition
		is defined to add to terminal states
		- example: see its use in get_next_states()
		"""
		self.u_broken = len(self.U)
		self.__add_transition(self.u_broken, self.u_broken, "True", ConstantRewardFunction(0))

	"""
	-------------------------------------------------------------------------------------------
	this functions are meant to be called once all reward machines are loaded
	and provide faster checks
	"""
	def is_terminal(self, u1):
		"""
		Instead of using __is_terminal() now all terminal states are stored in the array self.T
		so we can check if u1 is terminal by checking if its in self.T, which might be faster
		"""
		return (u1 in self.T) # returns True if in array, else False

	def get_next_state(self, u1, true_props):
		"""
		checks which of the next states already stored in self.delta_u[u1]
		validates evaluate_dnf(), if cant find any return dummy broken state
		"""
		# check not one of the broken
		if (u1 < self.u_broken):
			# for every next possible state
			for u2 in self.delta_u[u1]:
				# if validates the formula (see in reward machine utils evaluate_dnf(formula, true_props))
				# then its the next state for the agent (note this is not pruning the next possible states)
				if evaluate_dnf(self.delta_u[u1][u2], true_props):
					return u2
		# if u1 is broken or none of the next states validates evaluate_dnf() then return broken
			#print(u1,u2,self.delta_u[u1][u2],true_props)
			return self.u_broken

	def get_reward(self, u1, u2):
		"""
		general case would be get_reward(self, u1, u2, s1, action, s2)
		but here its sufficient

		returns reward associated with transition as given
		by delta_r[u1][u2] which is a ConstantRewardFunction object
		"""
		reward = 0
		if (u1 in self.delta_r) and (u2 in self.delta_r[u1]):
			# in general get_reward(s1, action, s2)
			# here its sufficient with get_reward()
			reward += self.delta_r[u1][u2].get_reward()
		return reward

	def append_state_label(symbol):
		"""
		c -- Close door, o -- Open door, u -- Unlock door,
		d -- Drop key, k -- Key picked,
		"""
		if symbol == "c":
			self.state_label = self.state_label.replace("o", "")
		if symbol == "d":
			self.state_label = self.state_label.replace("k", "")
		else:
			self.state_label += symbol

# for debugging purposes
if __name__ == "__main__":
	import time

	start_t = time.time()
	print("---- Debugging -----")
	print("RewardMachine Class, to construct needs <json environment file>, <idx of task>")
	file_rm = "minigrid_reward_machines.json"
	idx_rm = int(input("index of env (eg. 1 for MiniGridEmpty): "))
	reward_machine = RewardMachine(file_rm, idx_rm)

	print("reward_machine.is_terminal: ")
	print(reward_machine.is_terminal(0))
	print("reward_machine.get_next_state: ")
	print(reward_machine.get_next_state(0, "!D"))
	print("reward_machine.get_reward: ")
	print(reward_machine.get_reward(0,1))

	print(f"---- Finished {time.time()-start_t} s ----")
