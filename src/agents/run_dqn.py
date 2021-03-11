from dqn_algorithm import DQNAgent
from env.office import Office, OfficeParams
import numpy as np
import matplotlib.pyplot as plt

class DQNParams:
	def __init__(self, gamma, epsilon, batch_size, n_actions, eps_end, input_dims, lr):
		self.gamma = gamma
		self.epsilon = epsilon
		self.batch_size = batch_size
		self.n_actions = n_actions
		self.eps_end = eps_end
		self.input_dims = input_dims
		self.lr = lr

def run_dqn(environment_name, num_episodes):
	"""
	"""
	if environment_name == "office":
		params = OfficeParams() # office params
		env = Office(params) # office environment

		# network parameters for the environment
		dqn_params = DQNParams(gamma=0.99, epsilon=1.0, batch_size=1, n_actions=4, eps_end=0.01, input_dims=params.m*params.n, lr=0.001)

	else:
		raise NotImplementedError("Only office is implemented.")
		exit(1)

	# load agent with parameters from DQNParams
	agent = Agent(gamma=dqn_params.gamma, epsilon=dqn_params.epsilon, batch_size=dqn_params.batch_size, n_actions=dqn_params.n_actions,
					eps_end=dqn_algorithm.eps_end, input_dims=dqn_algorithm.input_dims, lr=dqn_algorithm.lr)

	# lists to keep track of progress
	scores, eps_history = [], []

	# print experiment information
	print(f"------ {environment_name} ------")
	print("Algorithm: DQN")
	print(f"Total episodes: {num_episodes}")
	print(f"Gamma: {dqn_params.gamma}")
	print(f"lr: {dqn_params.lr}")
	print()

	print("episode \t score \t average score \t epsilon")

	for episode in num_episodes:
		score = 0
		done = False
		observation = env.reset()
		while not done:
			action = agent.choose_action(observation)
			observation_, reward, done, info = env.step(action)
			score += reward
			agent.store_transition(observation, action, reward, observation_, done)
			agent.learn()
			observation = observation_
		scores.append(score)
		eps_history.append(agent.epsilon)

		avg_score = np.mean(scores[-10:])

		# print info of run
		print(episode + "\t" + round(score,2) + "\t" + round(avg_score,2) + "\t" + agent.epsilon)

	# plotting
	x =[i+1 for i in range(n_games)]
	plt.plot(x, eps_history)
	plt.title("Epsilon history")
	plt.xlabel("Game episode")
	plt.ylabel("Epsilon")
	plot.show()

	plot(x, scores)
	plt.title("Score")
	plt.xlabel("Game episode")
	plt.ylabel("Score")
	plot.show()


if __name__ == "__main__":
	import time
	t_start = time.time()

	print("--- Debugging ---")
	environment_name = input("Environment name (eg. office): ")
	num_episodes = int(input("Number of episodes (eg. 100): "))

	# run algorithm with user specified parameters
	run_dqn(environment_name, num_episodes)

	pritn(f"--- finished {time.time()-t_start} s ---")
	exit(0)


	