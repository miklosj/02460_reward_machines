import time, gym, argparse
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Wrappers to define Observartion output
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, OneHotPartialObsWrapper, ImgObsWrapper

# Agents
from q_learning  import QParams, QAgent
from qrm_learning  import QRMParams, QRMAgent
from dqn_agent import DQNParams, DQNAgent 

# Algorithms
def random_baseline(args):
	"""
	Runs an agent that chooses actions within the action space
	randomly, so as to use as baseline

	Parameters:
	------------
	args: dict
		command line arguments (args.num_games, ...)

	Returns:
	------------
	avg_reward: list
		list of average rewards every AVG_FREQ num episodes
	"""

	# make environement and define observation format with Wrappers
	env = gym.make(args.env_name)
	env = RGBImgPartialObsWrapper(env) # Get pixel observations
	env = ImgObsWrapper(env) # Get rid of the 'mission' field
	env.seed(0)		  # sets the seed
	obs = env.reset() # This now produces an RGB tensor only

	# Vector for storing intermediate results
	reward_history, avg_reward = [], []

	print("Episode num_steps avg_reward")
	for i in range(args.num_games):
		num_step , accum_reward = 0, 0
		done = False
		env.reset()
		while not done:

			if (num_step > MAX_NUM_STEPS): # To avoid it running forever
				raise CustomError(f"Maximum number of Steps reached ({MAX_NUM_STEPS})")
				exit(1)

			obs, reward, done, info = env.step(env.action_space.sample()) # take random action
			accum_reward += reward
			num_step += 1

		reward_history.append(accum_reward/num_step)
		avg_reward.append(np.mean(reward_history[-AVG_FREQ:]))

		if (i % PRINT_FREQ == 0): # Print training every PRINT_FREQ episodes
			print('%i \t %s \t %.3f' % (i, env.step_count, avg_reward[-1]))
	env.close()
	return avg_reward
def q_learning_baseline(args):
	"""
	Runs an agent with Q-learning algorihtm, so as to use as baseline

	Parameters:
	------------
	args: dict
		command line arguments (args.num_games, ...)

	Returns:
	------------
	avg_reward: list
		list of average rewards every AVG_FREQ num episodes
	"""
	# make environement and define observation format with Wrappers
	env = gym.make(args.env_name)
	# env = RGBImgPartialObsWrapper(env) # Get pixel observations
	env = OneHotPartialObsWrapper(env)
	# env = ImgObsWrapper(env) # Get rid of the 'mission' field
	env.seed(0)		  # sets the seed
	obs = env.reset() # This now produces an RGB tensor only

	# Vector for storing intermediate results
	reward_history, avg_reward = [], []

	params = QParams(gamma=0.99, eps_start=1.0, eps_dec=5e-6, eps_end=0.01, n_actions=7, lr=5e-3)
	agent = QAgent(params)

	print("Episode num_steps avg_Reward")
	for i in range(args.num_games):
		num_step , accum_reward = 0, 0
		done = False
		obs = env.reset()
		while not done:

			if (num_step > MAX_NUM_STEPS): # To avoid it running forever
				raise CustomError(f"Maximum number of Steps reached ({MAX_NUM_STEPS})")
				exit(1)

			action = agent.choose_action(obs)
			obs_, reward, done, info = env.step(action)
			#print(obs.size)
			#print(obs.ndim)
			#print(obs.shape)
			agent.learn(obs, action, reward, obs_, done)
			obs = deepcopy(obs_)
			accum_reward += reward
			num_step += 1

		reward_history.append(accum_reward/num_step)
		avg_reward.append(np.mean(reward_history[-AVG_FREQ:]))

		if (i % PRINT_FREQ == 0): # Print training every PRINT_FREQ episodes
			print('%i \t %s \t %.3f' % (i, env.step_count, avg_reward[-1]))
	env.close()
	return avg_reward
def dqn_learning_baseline(args):
	"""
	Runs an agent with DQN-learning algorihtm, so as to use as baseline

	Parameters:
	------------
	args: dict
		command line arguments (args.num_games, ...)

	Returns:
	------------
	avg_reward: list
		list of average rewards every AVG_FREQ num episodes
	"""
	# make environement and define observation format with Wrappers
	env = gym.make(args.env_name)
	env = RGBImgPartialObsWrapper(env) # Get pixel observations
	env = ImgObsWrapper(env) # Get rid of the 'mission' field
	env.seed(0)		  # sets the seed
	obs = env.reset() # This now produces an RGB tensor only

	# Vector for storing intermediate results
	reward_history, avg_reward = [], []

	params = DQNParams(gamma=0.99, epsilon=1.0, eps_dec=1e-5, eps_end=0.01, n_actions=7, input_dims=env.observation_space.shape, mem_size=50000, replace=1000, batch_size=10, lr=5e-3)
	agent = DQNAgent(params)

	print("Episode num_steps avg_Reward")
	for i in range(args.num_games):
		num_step , accum_reward = 0, 0
		done = False
		obs = env.reset()
		while not done:

			if (num_step > MAX_NUM_STEPS): # To avoid it running forever
				raise CustomError(f"Maximum number of Steps reached ({MAX_NUM_STEPS})")
				exit(1)

			action = agent.choose_action(obs)
			obs_, reward, done, info = env.step(action)
			agent.store_transition(obs, action, reward, obs_, done)
			agent.learn()
			obs = deepcopy(obs_)
			accum_reward += reward
			num_step += 1

		reward_history.append(accum_reward/num_step)
		avg_reward.append(np.mean(reward_history[-AVG_FREQ:]))

		if (i % PRINT_FREQ == 0): # Print training every PRINT_FREQ episodes
			print('%i \t %s \t %.3f' % (i, env.step_count, avg_reward[-1]))
	env.close()
	return avg_reward
def qrm_learning(args):
	"""
	Runs an agent with QRM-learning algorihtm,
	Q learning for reward Machines

	Parameters:
	------------
	args: dict
		command line arguments (args.num_games, ...)

	Returns:
	------------
	avg_reward: list
		list of average rewards every AVG_FREQ num episodes
	"""
	# make environement and define observation format with Wrappers
	env = gym.make(args.env_name)
	# env = RGBImgPartialObsWrapper(env) # Get pixel observations
	env = OneHotPartialObsWrapper(env)
	env = ImgObsWrapper(env) # Get rid of the 'mission' field
	env.seed(0)		  # sets the seed
	obs = env.reset() # This now produces an RGB tensor only

	# Vector for storing intermediate results
	reward_history, avg_reward = [], []

	params = QRMParams(gamma=0.99, eps_start=1.0, eps_dec=5e-6, eps_end=0.01, n_actions=7, lr=5e-3)
	agent = QRMAgent(params)

	print("Episode num_steps avg_Reward")
	for i in range(args.num_games):
		num_step , accum_reward = 0, 0
		done = False
		s1 = env.reset()
		u1 = self.rm.u0 # initial state from reward machine
		while not done:

			if (num_step > MAX_NUM_STEPS): # To avoid it running forever
				raise CustomError(f"Maximum number of Steps reached ({MAX_NUM_STEPS})")
				exit(1)

			action = agent.choose_action(s1)
			s2, _, done, info = env.step(action) # we dont want the action to come from the env but the rm

			# dirty workaround to get detect a transition in rm watching reward given by game
			# since only when tasks are done in the correct order env_reward is >0 we can use this to detect a transition in rm
			state_label  = ""
			u2 = rm.get_next_state(u1, state_label)
			reward = rm.delta_r[u1][u2]
			
			agent.learn(s1, action, reward, s2, done)
			s1 = deepcopy(s2)
			accum_reward += reward
			num_step += 1

		reward_history.append(accum_reward/num_step)
		avg_reward.append(np.mean(reward_history[-AVG_FREQ:]))

		if (i % PRINT_FREQ == 0): # Print training every PRINT_FREQ episodes
			print('%i \t %s \t %.3f' % (i, env.step_count, avg_reward[-1]))
	env.close()
	return avg_reward

# Define some constants
MAX_NUM_STEPS = 10000 # max number of steps within episode/games
AVG_FREQ = 10		 # compute reward avg of last AVG_FREQ num of episodes/games
PRINT_FREQ = 1  	 # every PRINT_FREQ number of episodes print train information

# Custom error class
class CustomError(Exception):
	pass

# parsing user input
# example: python main.py --env_name=MiniGrid-Empty-8x8-v0 --algo=dqn_learning --num_games=20000
parser = argparse.ArgumentParser()
parser.add_argument("--env_name", dest="env_name", help="gym environment to load", default='MiniGrid-Empty-8x8-v0', type=str)
parser.add_argument("--algo", help="Algorithm to use", default="random_baseline")
parser.add_argument("--num_games", help="Number of games to play.", default=100, type=int)
args = parser.parse_args()

# action space is discrete(7)
t0 = time.time()
if args.algo == "random_baseline":
	avg_reward = random_baseline(args=args)

elif args.algo == "q_learning":
	avg_reward = q_learning_baseline(args=args)

elif args.algo == "dqn_learning":
	avg_reward = dqn_learning_baseline(args=args)

else:
	raise NotImplementedError("To be implemented")
	exit(1)

t1 = time.time()
dt = t1 - t0
print("--- finished %s ---" % round(dt, 3))

# Plot results
plt.plot(avg_reward)
plt.xlabel("Episode")
plt.ylabel("Avg reward")
plt.title(f"Training Agent: {args.algo}, env: {args.env_name}")
fig_name = str(args.algo) + "_" + str(args.env_name) + "_" + str(args.num_games) + ".png"
plt.savefig(fig_name,  format="png")

exit(0)