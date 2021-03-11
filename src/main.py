import time, os.path, argparse

from agents.dqn_algorithm import DQNAgent, DQNParams
from envs.office import Office, OfficeParams
import numpy as np
import matplotlib.pyplot as plt

def run_dqn(environment_name, num_episodes):
	"""
	"""
	if environment_name == "office":
		params = OfficeParams() # office params
		env = Office(params) # office environment

		# network parameters for the environment
		dqn_params = DQNParams(gamma=0.99, epsilon=1.0, batch_size=1, n_actions=4, eps_end=0.01, input_dims=[params.m*params.n], lr=0.001)

	else:
		raise NotImplementedError("Only office is implemented.")
		exit(1)

	# load agent with parameters from DQNParams
	agent = DQNAgent(gamma=dqn_params.gamma, epsilon=dqn_params.epsilon, batch_size=dqn_params.batch_size, n_actions=dqn_params.n_actions,
					eps_end=dqn_params.eps_end, input_dims=dqn_params.input_dims, lr=dqn_params.lr)

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

	for episode in range(num_episodes):
		score = 0
		done = False
		observation = env.reset()
		iteration = 0
		while (not done) and (iteration < 100):
			action = agent.choose_action(observation)
			observation_, reward = env.step(action)
			env.game_display()
			score += reward
			agent.store_transition(observation, action, reward, observation_, done)
			agent.learn()
			observation = observation_
			iteration += 1
		scores.append(score)
		eps_history.append(agent.epsilon)

		avg_score = np.mean(scores[-10:])

		# print info of run
		print(str(episode) + "\t\t" + str(round(score,2)) + "\t" + str(round(avg_score,2)) + "\t" + str(agent.epsilon))

	# plotting
	x =[i+1 for i in range(num_episodes)]
	plt.plot(x, eps_history)
	plt.title("Epsilon history")
	plt.xlabel("Game episode")
	plt.ylabel("Epsilon")
	plt.show()

	plt.plot(x, scores)
	plt.title("Score")
	plt.xlabel("Game episode")
	plt.ylabel("Score")
	plt.show()


if __name__ == "__main__":
	"""
	Usage: python main.py --algorithm=<algorithm> --environment=<environment> --num_episodes=<num_episodes>
	"""

	# define valid inputs
	valid_environments = ["office", "minecraft"]
	valid_algorithms = ["q-learning", "dqn"]

	# structure of command-line arguments
	parser = argparse.ArgumentParser(prog="run experiments", description="Runs RL algorithm on specified environment.")
	parser.add_argument("--algorithm", default="dqn", type=str, help="RL agent to use, options are: "+str(valid_algorithms))
	parser.add_argument("--environment", default="office", type=str, help="Environment to use, options are: "+str(valid_environments))
	parser.add_argument("--num_episodes", default="20", type=int, help="Number of Games the algorithm plays. Number of episodes.")

	# verify input is valid
	args = parser.parse_args()
	if args.algorithm not in valid_algorithms:
		raise NotImplementedError(f"Algorithm not implemented. Valid inputs: {str(valid_algorithms)}")
	if args.environment not in valid_environments:
		raise NotImplementedError(f"Environment not implemented. Valid inputs: {str(valid_environments)}")

	if args.algorithm == "q-learning":
		raise NotImplementedError("To be merged.")

	if args.algorithm == "dqn":
		t_start = time.time()

		# run algorithm with user specified parameters
		run_dqn(args.environment, args.num_episodes)

		print(f"--- finished {time.time()-t_start} s ---")
		exit(0)
