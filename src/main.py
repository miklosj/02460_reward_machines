import time, gym, argparse
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

# Wrappers to define Observartion output
from gym_minigrid.wrappers import *

# Agents
from q_learning  import QParams, QAgent
from qrm_learning  import QRMParams, QRMAgent
from DQN.dqn_learning import DQNAgent
from DDQN.ddqn_learning import DDQNAgent

# utils functions (label function,....)
from utils import * 

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
    env.seed(0)       # sets the seed
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
    env.seed(0)       # sets the seed
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
    
    # fully obs wrapper for Event Listener
    envListener = FullyObsWrapper(env)
    envListener = ImgObsWrapper(envListener) # Get rid of the 'mission' field

    # partial obs wrapper for agent 
    # env = OneHotPartialObsWrapper(env)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env) # Get rid of the 'mission' field

    env.seed(0)       # sets the seed
    envListener.seed(0)

    obs = env.reset() # This now produces an RGB tensor only
    obsListener = envListener.reset()

    # Vector for storing intermediate results
    reward_history, avg_reward = [], []

    params = QRMParams(gamma=0.99, eps_start=1.0, eps_dec=5e-6, eps_end=0.01, n_actions=7, lr=5e-3, env_name=args.env_name)
    agent = QRMAgent(params)

    print("Episode num_steps avg_Reward")
    for i in range(args.num_games):
        num_step , accum_reward = 0, 0
        done = False
        s1 = env.reset()

        obsListener = envListener.reset()
        state_label = ""
        special_symbols = get_special_symbols(obsListener)

        u1 = agent.rm.u0 # initial state from reward machine
        while not done:

            if (num_step > MAX_NUM_STEPS): # To avoid it running forever
                raise CustomError(f"Maximum number of Steps reached ({MAX_NUM_STEPS})")
                exit(1)

            action = agent.choose_action(s1)
            s2, reward, done, info = env.step(action) # we dont want the action to come from the env but the rm

            # dirty  hack, we ran two parallel envs, one for the agent and one fully observable for the 
            # event listener that returns the label function used to make transitions of the Reward Machine 
            obsListener_ , _, _, _ = envListener.step(action)
            special_symbols_ = get_special_symbols(obsListener_)
            state_label = return_symbol(special_symbols, special_symbols_, state_label)

            #print(f"state_label: {state_label}")
            #print(f"reward_machine_state: {u1}")

            u2 = agent.rm.get_next_state(u1, state_label)
            reward_rm = agent.rm.delta_r[u1][u2].get_reward()

            agent.learn(s1, action, reward_rm, s2, done)

            # update params
            u1 = u2
            special_symbols = special_symbols_
            obsListener = deepcopy(obsListener_)

            s1 = deepcopy(s2)
            accum_reward += reward
            num_step += 1

        reward_history.append(accum_reward/num_step)
        avg_reward.append(np.mean(reward_history[-AVG_FREQ:]))

        if (i % PRINT_FREQ == 0): # Print training every PRINT_FREQ episodes
            print('%i \t %s \t %.3f' % (i, env.step_count, avg_reward[-1]))
    env.close()
    envListener.close()
    return avg_reward


def dqn_learning(args):
    """
    Runs an agent with DQN-learning algorihtm,
    Deep Q Network learning

    Parameters:
    ------------
    args: dict
        command line arguments (args.num_games, ...)

    Returns:
    ------------
    avg_reward: list
        list of average rewards every AVG_FREQ num episodes
 
    """
    algorithm = args.algo
    # make environement and define observation format with Wrappers
    env = gym.make(args.env_name)
    
    # fully obs wrapper for Event Listener
    envListener = FullyObsWrapper(env)
    envListener = ImgObsWrapper(envListener) # Get rid of the 'mission' field

    # partial obs wrapper for agent 
    # env = OneHotPartialObsWrapper(env)
    env = ImgObsWrapper(env) # Get rid of the 'mission' field

    env.seed(0)       # sets the seed
    envListener.seed(0)

    obs = env.reset()
    obsListener = envListener.reset()
    agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001, input_dims=(obs.shape),
            n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
            batch_size=32, replace=1000, eps_dec=1e-5,
            chkpt_dir='models/', algo=algorithm, env_name=args.env_name)

    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    printFile = open('results/' + 'results_' +algorithm+ args.env_name + '.txt', 'w')
    #fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
    #        + str(n_games) + 'games'
    #figure_file = 'plots/' + fname + '.png'

    scores, avg_scores, std_scores, eps_history, steps_array = [], [], [], [], []

    best_score = -np.inf

    print('Episode\t','Steps\t','Score\t',
            'Best_Score\t','Epsilon\t', file=printFile)

    for i in range(args.num_games):
        done = False
        observation = env.reset()

        n_steps = 0
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action,
                                     reward, observation_, done)
                agent.learn()

            observation = deepcopy(observation_)
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)
        avg_scores.append(np.mean(scores[-100:]))
        std_scores = np.append(std_scores, np.std(scores[-100:]))

        print('%d\t' %i, '%d\t' %n_steps,
                '%.2f\t' %score,'%.2f\t' %best_score,
                '%.2f\t' %agent.epsilon, file=printFile)

        if score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = score

        eps_history.append(agent.epsilon)

    #x = [i+1 for i in range(len(scores))]
    #plot_learning_curve(x, avg_scores, std_scores, eps_history, figure_file)

    return avg_scores



def ddqn_learning(args):
    """
    Runs an agent with DQN-learning algorihtm,
    Deep Q Network learning

    Parameters:
    ------------
    args: dict
        command line arguments (args.num_games, ...)

    Returns:
    ------------
    avg_reward: list
        list of average rewards every AVG_FREQ num episodes
 
    """
    algorithm = args.algo
    # make environement and define observation format with Wrappers
    env = gym.make(args.env_name)
    
    # fully obs wrapper for Event Listener
    envListener = FullyObsWrapper(env)
    envListener = ImgObsWrapper(envListener) # Get rid of the 'mission' field

    # partial obs wrapper for agent 
    # env = OneHotPartialObsWrapper(env)
    env = ImgObsWrapper(env) # Get rid of the 'mission' field

    env.seed(0)       # sets the seed
    envListener.seed(0)

    obs = env.reset()
    obsListener = envListener.reset()
    agent = DDQNAgent(gamma=0.99, epsilon=1, lr=0.0001, input_dims=(obs.shape),
            n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
            batch_size=32, replace=1000, eps_dec=1e-5,
            chkpt_dir='models/', algo=algorithm, env_name=args.env_name)

    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    printFile = open('results/' + 'results_' +algorithm+ args.env_name + '.txt', 'w')
    #fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
    #        + str(n_games) + 'games'
    #figure_file = 'plots/' + fname + '.png'

    scores, avg_scores, std_scores, eps_history, steps_array = [], [], [], [], []

    best_score = -np.inf

    print('Episode\t','Steps\t','Score\t',
            'Best_Score\t','Epsilon\t', file=printFile)

    for i in range(args.num_games):
        done = False
        observation = env.reset()

        n_steps = 0
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action,
                                     reward, observation_, done)
                agent.learn()

            observation = deepcopy(observation_)
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)
        avg_scores.append(np.mean(scores[-100:]))
        std_scores = np.append(std_scores, np.std(scores[-100:]))

        print('%d\t' %i, '%d\t' %n_steps,
                '%.2f\t' %score,'%.2f\t' %best_score,
                '%.2f\t' %agent.epsilon, file=printFile)

        if score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = score

        eps_history.append(agent.epsilon)

    #x = [i+1 for i in range(len(scores))]
    #plot_learning_curve(x, avg_scores, std_scores, eps_history, figure_file)

    return avg_scores

# Define some constants
MAX_NUM_STEPS = 10000 # max number of steps within episode/games
AVG_FREQ = 10        # compute reward avg of last AVG_FREQ num of episodes/games
PRINT_FREQ = 1       # every PRINT_FREQ number of episodes print train information

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

elif args.algo == "qrm_learning":
    avg_reward = qrm_learning(args=args)

elif args.algo == "dqn_learning":
    avg_reward = dqn_learning(args=args)

elif args.algo == "ddqn_learning":
    avg_reward = ddqn_learning(args=args)

else:
    raise NotImplementedError("To be implemented")
    exit(1)


# Plot results
plt.plot(avg_reward)
plt.xlabel("Episode")
plt.ylabel("Avg reward")
plt.title(f"Training Agent: {args.algo}, env: {args.env_name}")
fig_name = str(args.algo) + "_" + str(args.env_name) + "_" + str(args.num_games) + ".png"
plt.savefig("plots/" + fig_name,  format="png")

t1 = time.time()
dt = t1 - t0
print("--- finished %s ---" % round(dt, 3))
exit(0)
