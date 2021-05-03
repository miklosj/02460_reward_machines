import time, gym, argparse
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import os

# Wrappers to define Observartion output
from gym_minigrid.wrappers import *

# Agents
from Q.q_learning  import QParams, QAgent
from QRM.qrm_learning  import QRMParams, QRMAgent, QRM_network
from CRM.crm_learning  import CRMParams, CRMAgent
from DQN.dqn_learning import DQNAgent
from DDQN.ddqn_learning import DDQNAgent
from DCRM.dcrm_learning import DCRMAgent
from DDCRM.ddcrm_learning import DDCRMAgent
from DQRM.dqrm_learning import DQRMAgent

# utils functions (label function,....)
from utils import *

# Saving folders
if not os.path.exists("../plots"):
    os.makedirs("../plots")
if not os.path.exists("../models"):
    os.makedirs("../models")
if not os.path.exists("../results"):
    os.makedirs("../results")


def env_unwrapper(env):
    full_grid = env.grid.encode()
    full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([10, 0, env.agent_dir])
    return full_grid


# Algorithms
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
    env = RGBImgPartialObsWrapper(env) # Get pixel observations
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    env.seed(0)       # sets the seed
    obs = env.reset() # This now produces an RGB tensor only

    # Vector for storing intermediate results
    reward_history, avg_reward = [], []

    params = QParams(gamma=0.999, eps_start=1.0, eps_dec=5e-10, eps_end=0.05, n_actions=7, lr=1e-4)
    agent = QAgent(params)

    print("Episode num_steps avg_Reward")
    for i in range(args.num_games):
        num_step , accum_reward = 0, 0
        done = False
        obs = env.reset()
        while not done:

            action = agent.choose_action(obs)
            obs_, reward, done, info = env.step(action)
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

    # partial obs wrapper for agent 
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env) # Get rid of the 'mission' field

    # sets the seed
    env.seed(0)       

    obs = env.reset()
    obsListener = env_unwrapper(env)

    # Vector for storing intermediate results
    reward_history, avg_reward = [], []

    params = QRMParams(gamma=0.999, eps_start=1.0, eps_dec=5e-10, eps_end=0.05, n_actions=7, lr=1e-4, env_name=args.env_name)
    agent = QRMAgent(params)
    unique_states = agent.rm.unique_states[:-1]

    # logical symbols class
    ls = logical_symbols()
    
    print("Episode num_steps avg_Reward")
    for i in range(args.num_games):
        num_step , accum_reward = 0, 0
        done = False

        # reset the two envs
        s1 = env.reset()
        obsListener = env_unwrapper(env)

        state_label = "" # reset state_label
        special_symbols = ls.get_special_symbols(obsListener)
        u1 = agent.rm.u0 # initial state from reward machine
        while not done:

            action = agent.choose_action(s1, u1)
            s2, reward, done, info = env.step(action) # we dont want the action to come from the env but the rm
            u2 = agent.rm.get_next_state(u1, state_label)

            obsListener_ = env_unwrapper(env)
            special_symbols_ = ls.get_special_symbols(obsListener_)
            state_label = ls.return_symbol(special_symbols, special_symbols_, state_label)

            for u1_temp in range(len(unique_states)):
                u2_temp = agent.rm.get_next_state(u1_temp, state_label)
                if (u1_temp == u1): # this is the transition performed
                    pass
                else:
                    reward_rm = agent.rm.delta_r[u1_temp][u2_temp].get_reward()
                    agent.learn(s1, u1_temp, action, reward_rm, s2, u2_temp)
             
            reward_rm = agent.rm.delta_r[u1][u2].get_reward()
            agent.learn(s1, u1, action, reward_rm, s2, u2)

            # update params
            u1 = u2
            s1 = deepcopy(s2)
            special_symbols = special_symbols_
            obsListener = deepcopy(obsListener_)

            accum_reward += reward # note this reward comes from environement and not reward machine
            num_step += 1
            

        reward_history.append(accum_reward)
        avg_reward.append(np.mean(reward_history[-AVG_FREQ:]))

        if (i % PRINT_FREQ == 0): # Print training every PRINT_FREQ episodes
            print('%i \t %s \t %.3f' % (i, env.step_count, avg_reward[-1]))
    env.close()
    return avg_reward

def crm_learning(args):
    """
    Runs an agent with CRM-learning algorihtm,
    Q learning with counterfactual experiences for reward Machines

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

    # partial obs wrapper for agent 
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env) # Get rid of the 'mission' field

    # sets the seed
    env.seed(0)       

    obs = env.reset()
    obsListener = env_unwrapper(env)

    # Vector for storing intermediate results
    reward_history, avg_reward = [], []

    params = CRMParams(gamma=0.999, eps_start=1.0, eps_dec=5e-10, eps_end=0.05, n_actions=7, lr=1e-4, env_name=args.env_name)
    agent = CRMAgent(params)
    unique_states = agent.rm.unique_states[:-1]

    # logical symbols class
    ls = logical_symbols()
 
    print("Episode num_steps avg_Reward")
    for i in range(args.num_games):
        num_step , accum_reward = 0, 0
        done = False

        # reset the two envs
        s1 = env.reset()
        obsListener = env_unwrapper(env)

        state_label = "" # reset state_label
        special_symbols = ls.get_special_symbols(obsListener)
        u1 = agent.rm.u0 # initial state from reward machine
        while not done:

            action = agent.choose_action(s1, u1)
            s2, reward, done, info = env.step(action) # we dont want the action to come from the env but the rm
            u2 = agent.rm.get_next_state(u1, state_label)

            obsListener_ = env_unwrapper(env)
            special_symbols_ = ls.get_special_symbols(obsListener_)
            state_label = ls.return_symbol(special_symbols, special_symbols_, state_label)

            for u1_temp in range(len(unique_states)):
                u2_temp = agent.rm.get_next_state(u1_temp, state_label)
                if (u1_temp == u1): # this is the transition performed
                    pass
                else:
                    reward_rm = agent.rm.delta_r[u1_temp][u2_temp].get_reward()
                    agent.learn(s1, u1_temp, action, reward_rm, s2, u2_temp)

            reward_rm = agent.rm.delta_r[u1][u2].get_reward()
            agent.learn(s1, u1, action, reward_rm, s2, u2)

            # update params
            u1 = u2
            s1 = deepcopy(s2)
            special_symbols = special_symbols_
            obsListener = deepcopy(obsListener_)

            accum_reward += reward # note this reward comes from environement and not reward machine
            num_step += 1

        reward_history.append(accum_reward)
        avg_reward.append(np.mean(reward_history[-AVG_FREQ:]))

        if (i % PRINT_FREQ == 0): # Print training every PRINT_FREQ episodes
            print('%i \t %s \t %.3f' % (i, env.step_count, avg_reward[-1]))
    env.close()
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

    # partial obs wrapper for agent 
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env) # Get rid of the 'mission' field

    env.seed(0)       # sets the seed

    obs = env.reset()
    #print(obs.shape)
    obsListener = env_unwrapper(env)
    agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001, input_dims=(obs.shape),
            n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
            batch_size=32, replace=1000, eps_dec=1e-5,
            chkpt_dir='../models/', algo=algorithm, env_name=args.env_name)

    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    printFile = open('../results/' + 'results_' +algorithm+ args.env_name + '.txt', 'w')
    #fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
    #        + str(n_games) + 'games'
    #figure_file = 'plots/' + fname + '.png'

    scores, avg_scores, std_scores, eps_history, steps_array = [], [], [], [], []

    best_score = -np.inf

    # logical symbols class
    ls = logical_symbols()
 
    print('Episode\t','Steps\t','Score\t',
            'Best_Score\t','Epsilon\t', file=printFile)
    
    for i in range(args.num_games):
        done = False
        observation = env.reset()
        obsListener = env_unwrapper(env)
        state_label = ""
        
        special_symbols = ls.get_special_symbols(obsListener)
        u1 = agent.rm.u0 # initial state from reward machine

        n_steps = 0
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            
            # Run parallel environment to check for environment objects states
            obsListener_ = env_unwrapper(env)
            special_symbols_ = ls.get_special_symbols(obsListener_)
            state_label = ls.return_symbol(special_symbols, special_symbols_, state_label)

            # Get reward machine state 
            u2 = agent.rm.get_next_state(u1, state_label)
            reward_rm = agent.rm.delta_r[u1][u2].get_reward()
            
            if not load_checkpoint:
                agent.store_transition(observation, action,
                        reward_rm, observation_, done)
                agent.learn()
 
            # Update params
            u1 = deepcopy(u2)
            special_symbols = deepcopy(special_symbols_)
            #obsListener = deepcopy(obsListener_)
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
    Runs an agent with DDQN-learning algorihtm,
    Double Deep Q Network learning

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

    # partial obs wrapper for agent 
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env) # Get rid of the 'mission' field

    env.seed(0)       # sets the seed

    obs = env.reset()
    #print(obs.shape)
    obsListener = env_unwrapper(env)
    agent = DDQNAgent(gamma=0.99, epsilon=1, lr=0.0001, input_dims=(obs.shape),
            n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
            batch_size=32, replace=1000, eps_dec=1e-5,
            chkpt_dir='../models/', algo=algorithm, env_name=args.env_name)

    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    printFile = open('../results/' + 'results_' +algorithm+ args.env_name + '.txt', 'w')

    scores, avg_scores, std_scores, eps_history, steps_array = [], [], [], [], []

    best_score = -np.inf

    # logical symbols class
    ls = logical_symbols()
 
    print('Episode\t','Steps\t','Score\t',
            'Best_Score\t','Epsilon\t', file=printFile)
    
    for i in range(args.num_games):
        done = False
        observation = env.reset()
        #print("Episode: ", i)
        obsListener = env_unwrapper(env)
        
        state_label = ""
        special_symbols = ls.get_special_symbols(obsListener)
        u1 = agent.rm.u0 # initial state from reward machine

        n_steps = 0
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            
            # Run parallel environment to check for environment objects states
            obsListener_ = env_unwrapper(env)
            special_symbols_ = ls.get_special_symbols(obsListener_)
            state_label = ls.return_symbol(special_symbols, special_symbols_, state_label)
            # Get reward machine state 
            u2 = agent.rm.get_next_state(u1, state_label)
            reward_rm = agent.rm.delta_r[u1][u2].get_reward()
            
            if not load_checkpoint:
                agent.store_transition(observation, action,
                        reward_rm, observation_, done)
                agent.learn()

            if score>0:
                print("||",state_label,"||")
                #print(obsListener)
                #print(obsListener_)
                print(u1,u2,"\n",reward_rm,score)


            # Update params
            u1 = deepcopy(u2)
            special_symbols = deepcopy(special_symbols_)
            obsListener = deepcopy(obsListener_)
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

    return avg_scores

def dcrm_learning(args):
    """
    Runs an agent with DCRM-learning algorihtm,

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

    # partial obs wrapper for agent
    #env = FullyObsWrapper(env)
    env = RGBImgPartialObsWrapper(env)
    #env = FlatObsWrapper(env)
    env = ImgObsWrapper(env) # Get rid of the 'mission' field

    env.seed(0)       # sets the seed

    obs = env.reset()
    obsListener = env_unwrapper(env)
    print(obs.shape) 
    
    agent = DCRMAgent(gamma=0.90, epsilon=1, lr=0.0001, input_dims=(obs.shape),
            n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
            batch_size=32, replace=100, eps_dec=1e-6,
            chkpt_dir='../models/', algo=algorithm, env_name=args.env_name)

    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    printFile = open('../results/' + 'results_' +algorithm+ args.env_name + '.txt', 'w')

    scores, avg_scores, std_scores, eps_history, steps_array = [], [], [], [], []

    best_score = -np.inf

    # logical symbols class
    ls = logical_symbols()
 
    print('Episode\t','Steps\t','Score\t',
            'Best_Score\t','Epsilon\t', file=printFile)
    
    for i in range(args.num_games):
        done = False
        #print("Running episode: ", i)
        observation = env.reset()
        obsListener = env_unwrapper(env)
        
        state_label = ""
        
        special_symbols = ls.get_special_symbols(obsListener)

        u1 = agent.rm.u0 # initial state from reward machine

        n_steps = 0
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            # Run parallel environment to check for environment objects states
            obsListener_ = env_unwrapper(env)
            
            special_symbols_ = ls.get_special_symbols(obsListener_)
            state_label = ls.return_symbol(special_symbols, special_symbols_, state_label)

            # Get reward machine state
            for u1 in range(agent.n_rm_states):
                u2 = agent.rm.get_next_state(u1, state_label)
                reward_rm = agent.rm.delta_r[u1][u2].get_reward()

                agent.store_transition(observation, u1, action,
                    reward_rm, observation_, u2, done)


            #if done:
            #    print("||",state_label,"||")
            #    print(u1,u2,"\n",reward_rm,score)

            agent.learn()

            ## Update params
            special_symbols = special_symbols_
            
            obsListener = obsListener_
            observation = observation_

            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)
        avg_scores.append(np.mean(scores[-100:]))
        std_scores = np.append(std_scores, np.std(scores[-100:]))

        print('%d\t' %i, '%d\t' %n_steps,
                '%.2f\t' %score,'%.2f\t' %best_score,
                '%.2f\t' %agent.epsilon)#, file=printFile)

        if score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = score

        eps_history.append(agent.epsilon)

    return avg_scores


def ddcrm_learning(args):
    """
    Runs an agent with DDCRM-learning algorihtm,

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

    # partial obs wrapper for agent
    #env = FullyObsWrapper(env)
    env = RGBImgPartialObsWrapper(env)
    #env = FlatObsWrapper(env)
    env = ImgObsWrapper(env) # Get rid of the 'mission' field

    env.seed(0)       # sets the seed

    obs = env.reset()
    obsListener = env_unwrapper(env)
    print(obs.shape) 
    
    agent = DDCRMAgent(gamma=0.90, epsilon=1, lr=0.0001, input_dims=(obs.shape),
            n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
            batch_size=32, replace=100, eps_dec=1e-6,
            chkpt_dir='../models/', algo=algorithm, env_name=args.env_name)

    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    printFile = open('../results/' + 'results_' +algorithm+ args.env_name + '.txt', 'w')

    scores, avg_scores, std_scores, eps_history, steps_array = [], [], [], [], []

    best_score = -np.inf

    # logical symbols class
    ls = logical_symbols()
 
    print('Episode\t','Steps\t','Score\t',
            'Best_Score\t','Epsilon\t', file=printFile)
    
    for i in range(args.num_games):
        done = False
        #print("Running episode: ", i)
        observation = env.reset()
        obsListener = env_unwrapper(env)
        
        state_label = ""
        
        special_symbols = ls.get_special_symbols(obsListener)

        u1 = agent.rm.u0 # initial state from reward machine

        n_steps = 0
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward

            # Run parallel environment to check for environment objects states
            obsListener_ = env_unwrapper(env)
            
            special_symbols_ = ls.get_special_symbols(obsListener_)
            state_label = ls.return_symbol(special_symbols, special_symbols_, state_label)

            # Get reward machine state
            for u1 in range(agent.n_rm_states):
                u2 = agent.rm.get_next_state(u1, state_label)
                reward_rm = agent.rm.delta_r[u1][u2].get_reward()

                agent.store_transition(observation, u1, action,
                    reward_rm, observation_, u2, done)

            agent.learn()

            ## Update params
            special_symbols = special_symbols_
            
            obsListener = obsListener_
            observation = observation_

            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)
        avg_scores.append(np.mean(scores[-100:]))
        std_scores = np.append(std_scores, np.std(scores[-100:]))

        print('%d\t' %i, '%d\t' %n_steps,
                '%.2f\t' %score,'%.2f\t' %best_score,
                '%.2f\t' %agent.epsilon)#, file=printFile)

        if score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = score

        eps_history.append(agent.epsilon)

    return avg_scores




def dqrm_learning(args):
    """
    Runs an agent with DQRM-learning algorihtm,

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

    # partial obs wrapper for agent
    # env = OneHotPartialObsWrapper(env)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env) # Get rid of the 'mission' field

    env.seed(0)       # sets the seed

    obs = env.reset()
    obsListener = env_unwrapper(env)
    
    agent = DQRMAgent(gamma=0.90, epsilon=1, lr=0.0001, input_dims=(obs.shape),
            n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
            batch_size=32, replace=100, eps_dec=1e-5,
            chkpt_dir='../models/', algo=algorithm, env_name=args.env_name)

    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    printFile = open('../results/' + 'results_' +algorithm+ args.env_name + '.txt', 'w')

    scores, avg_scores, std_scores, eps_history, steps_array = [], [], [], [], []

    best_score = -np.inf

    # logical symbols class
    ls = logical_symbols()
 
    print('Episode\t','Steps\t','Score\t',
            'Best_Score\t','Epsilon\t', file=printFile)
    
    for i in range(args.num_games):
        done = False
        print("Running episode: ", i)
        observation = env.reset()

        obsListener = env_unwrapper(env)
        state_label = ""
        special_symbols = ls.get_special_symbols(obsListener)

        u1 = agent.rm.u0 # initial state from reward machine

        n_steps = 0
        score = 0
        while not done:
            action = agent.choose_action(observation, u1)
            observation_, reward, done, info = env.step(action)
            score += reward

            obsListener_ = env_unwrapper(env)
            special_symbols_ = ls.get_special_symbols(obsListener_)
            state_label = ls.return_symbol(special_symbols, special_symbols_, state_label)

            # Get reward machine state
            u1_ = np.zeros(agent.n_rm_states)
            u2_ = np.zeros(agent.n_rm_states)
            reward_rm = np.zeros(agent.n_rm_states)
            for j in range(agent.n_rm_states):
                k = agent.rm.get_next_state(j, state_label)
                rew_rm = agent.rm.delta_r[j][k].get_reward()
                u1_[j] = j
                u2_[j] = k
                reward_rm[j] = rew_rm

            agent.store_transition(observation, u1_, action,
                reward_rm, observation_, u2_, done)

            agent.learn()

            u2 = agent.rm.get_next_state(u1, state_label)
            ## Update params
            special_symbols = special_symbols_
            obsListener = obsListener_
            observation = observation_
            u1 = u2
            n_steps += 1

        scores.append(score)
        steps_array.append(n_steps)
        avg_scores.append(np.mean(scores[-100:]))
        std_scores = np.append(std_scores, np.std(scores[-100:]))

        print('%d\t' %i, '%d\t' %n_steps,
                '%.2f\t' %score,'%.2f\t' %best_score,
                '%.2f\t' %agent.epsilon)#, file=printFile)

        if score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = score

        eps_history.append(agent.epsilon)

    return avg_scores



# Define some constants
MAX_NUM_STEPS = 10000 # max number of steps within episode/games
AVG_FREQ = 100        # compute reward avg of last AVG_FREQ num of episodes/games
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

elif args.algo == "crm_learning":
    avg_reward = crm_learning(args=args)

elif args.algo == "dqn_learning":
    avg_reward = dqn_learning(args=args)

elif args.algo == "ddqn_learning":
    avg_reward = ddqn_learning(args=args)

elif args.algo == "dcrm_learning":
    avg_reward = dcrm_learning(args=args)

elif args.algo == "ddcrm_learning":
    avg_reward = ddcrm_learning(args=args)

elif args.algo == "dqrm_learning":
    avg_reward = dqrm_learning(args=args)

else:
    raise NotImplementedError("To be implemented")
    exit(1)

# save dataframe
name = str(args.algo) + "_" + str(args.env_name) + "_" + str(args.num_games)

episodes = np.array([i for i in range(args.num_games) if i%PRINT_FREQ == 0])
df = pd.DataFrame(data= {"episodes": episodes, "avg_reward": avg_reward})
df.to_csv(path_or_buf=name + ".csv")

# Plot results
fig_name = name + ".png"
plt.plot(avg_reward)
plt.xlabel("Episode")
plt.ylabel("Avg reward")
plt.title(f"Training Agent: {args.algo}, env: {args.env_name}")
plt.savefig("../plots/" + fig_name,  format="png")
t1 = time.time()
dt = t1 - t0
print("--- finished %s ---" % round(dt, 3))
exit(0)
