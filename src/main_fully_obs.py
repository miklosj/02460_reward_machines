import time, gym, argparse
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F

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
from PPO.PPOutils import Storage, orthogonal_init
from PPO.ppo_learning import *

# utils functions (label function,....)
from utils import *
from reward_machine import RewardMachine

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
    env = RGBImgObsWrapper(env) # Get pixel observations
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

        if (i % 1000 == 0): # Print training every PRINT_FREQ episodes
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
    env = RGBImgObsWrapper(env) # Get pixel observations
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    env.seed(0)       # sets the seed
    obs = env.reset() # This now produces an RGB tensor only

    # Vector for storing intermediate results
    reward_history, avg_reward = [], []

    params = QParams(gamma=0.999, eps_start=1.0, eps_dec=5e-10, eps_end=0.05, n_actions=7, lr=1e-5)
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
    #env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env) # Get rid of the 'mission' field

    # sets the seed
    env.seed(0)       

    obs = env.reset()

    # Vector for storing intermediate results
    reward_history, avg_reward = [], []

    params = QRMParams(gamma=0.999, eps_start=1.0, eps_dec=5e-10, eps_end=0.05, n_actions=7, lr=1e-5, env_name=args.env_name)
    agent = QRMAgent(params)
    unique_states = agent.rm.unique_states[:-1]

    # logical symbols class
    ls = logical_symbols()
    
    print("Episode num_steps avg_Reward")
    for i in range(args.num_games):
        num_step , accum_reward = 0, 0
        done = False

        s1 = env.reset()

        state_label = "" # reset state_label
        special_symbols = ls.get_special_symbols(s1)
        u1 = agent.rm.u0 # initial state from reward machine
        while not done:

            action = agent.choose_action(s1, u1)
            s2, reward, done, info = env.step(action) # we dont want the action to come from the env but the rm
            u2 = agent.rm.get_next_state(u1, state_label)

            special_symbols_ = ls.get_special_symbols(s2)
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

    env = ImgObsWrapper(env) # Get rid of the 'mission' field

    # sets the seed
    env.seed(0)

    obs = env.reset()

    # Vector for storing intermediate results
    reward_history, avg_reward = [], []

    params = CRMParams(gamma=0.999, eps_start=1.0, eps_dec=5e-10, eps_end=0.05, n_actions=7, lr=1e-5, env_name=args.env_name)
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

        state_label = "" # reset state_label
        special_symbols = ls.get_special_symbols(s1)
        u1 = agent.rm.u0 # initial state from reward machine
        while not done:

            action = agent.choose_action(s1, u1)
            s2, reward, done, info = env.step(action) # we dont want the action to come from the env but the rm
            u2 = agent.rm.get_next_state(u1, state_label)

            special_symbols_ = ls.get_special_symbols(s2)
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
    algorithm = args.algo + "_fully"
    # make environement and define observation format with Wrappers
    env = gym.make(args.env_name)
    
    # fully obs wrapper for Event Listener
    envListener = deepcopy(env)
    envListener = FullyObsWrapper(envListener)
    envListener = ImgObsWrapper(envListener) # Get rid of the 'mission' field

    # partial obs wrapper for agent 
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env) # Get rid of the 'mission' field

    env.seed(0)       # sets the seed
    envListener.seed(0)

    obs = env.reset()
    #print(obs.shape)
    obsListener = envListener.reset()
    agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001, input_dims=(obs.shape),
            n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
            batch_size=32, replace=1000, eps_dec=1e-7,
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
        obsListener = envListener.reset()
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
            obsListener_ , _, _, _ = envListener.step(action)
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
    algorithm = args.algo + "_fully"
    # make environement and define observation format with Wrappers
    env = gym.make(args.env_name)
    
    # fully obs wrapper for Event Listener
    envListener = deepcopy(env)
    envListener = FullyObsWrapper(envListener)
    envListener = ImgObsWrapper(envListener) # Get rid of the 'mission' field

    # partial obs wrapper for agent 
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env) # Get rid of the 'mission' field

    env.seed(0)       # sets the seed
    envListener.seed(0)

    obs = env.reset()
    #print(obs.shape)
    obsListener = envListener.reset()
    agent = DDQNAgent(gamma=0.99, epsilon=1, lr=0.0001, input_dims=(obs.shape),
            n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
            batch_size=32, replace=1000, eps_dec=1e-7,
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
        obsListener = envListener.reset()
        
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
            obsListener_ , _, _, _ = envListener.step(action)
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
    algorithm = args.algo + "_fully"
    # make environement and define observation format with Wrappers
    env = gym.make(args.env_name)
    
    # fully obs wrapper for Event Listener
    envListener = deepcopy(env)
    envListener = FullyObsWrapper(envListener)
    envListener = ImgObsWrapper(envListener) # Get rid of the 'mission' field

    # partial obs wrapper for agent
    #env = FullyObsWrapper(env)
    env = RGBImgObsWrapper(env)
    #env = FlatObsWrapper(env)
    env = ImgObsWrapper(env) # Get rid of the 'mission' field

    env.seed(0)       # sets the seed
    envListener.seed(0)

    obs = env.reset()
    obsListener = envListener.reset()
    print(obs.shape) 
    
    agent = DCRMAgent(gamma=0.99, epsilon=1, lr=0.0001, input_dims=(obs.shape),
            n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
            batch_size=128, replace=1000, eps_dec=1e-7,
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
        obsListener = envListener.reset()
        
        state_label = ""
        
        special_symbols = ls.get_special_symbols(obsListener)

        u1 = agent.rm.u0 # initial state from reward machine
        n_steps = 0
        score = 0
        while not done:
            
            action = agent.choose_action(observation, u1)
            observation_, reward, done, info = env.step(action)
            score += reward

            # Run parallel environment to check for environment objects states
            obsListener_ , _, _, _ = envListener.step(action)
            
            special_symbols_ = ls.get_special_symbols(obsListener_)
            state_label = ls.return_symbol(special_symbols, special_symbols_, state_label)
    
            # Get reward machine state
            for u1_ in range(agent.n_rm_states):
                u2_ = agent.rm.get_next_state(u1_, state_label)
                reward_rm = agent.rm.delta_r[u1_][u2_].get_reward()

                u1_ = rm_state_onehot(u1_, agent.n_rm_states)
                u2_ = rm_state_onehot(u2_, agent.n_rm_states)
                agent.store_transition(observation, u1_, action,
                    reward_rm, observation_, u2_, done)


            #if done:
            #    print("||",state_label,"||")
            #    print(u1,u2,"\n",reward_rm,score)
            
            u2 = agent.rm.get_next_state(u1, state_label)
            agent.learn()

            ## Update params
            special_symbols = special_symbols_
            
            #obsListener = obsListener_
            u1 = u2
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
    algorithm = args.algo + "_fully"
    # make environement and define observation format with Wrappers
    env = gym.make(args.env_name)
    
    # fully obs wrapper for Event Listener
    envListener = deepcopy(env)
    envListener = FullyObsWrapper(envListener)
    envListener = ImgObsWrapper(envListener) # Get rid of the 'mission' field

    # partial obs wrapper for agent
    #env = FullyObsWrapper(env)
    env = RGBImgObsWrapper(env)
    #env = FlatObsWrapper(env)
    env = ImgObsWrapper(env) # Get rid of the 'mission' field

    env.seed(0)       # sets the seed
    envListener.seed(0)

    obs = env.reset()
    obsListener = envListener.reset()
    print(obs.shape) 
    
    agent = DDCRMAgent(gamma=0.99, epsilon=1, lr=0.0001, input_dims=(obs.shape),
            n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
            batch_size=128, replace=1000, eps_dec=1e-7,
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
        obsListener = envListener.reset()
        
        state_label = ""
        
        special_symbols = ls.get_special_symbols(obsListener)

        u1 = agent.rm.u0 # initial state from reward machine
        n_steps = 0
        score = 0
        while not done:
            
            action = agent.choose_action(observation, u1)
            observation_, reward, done, info = env.step(action)
            score += reward

            # Run parallel environment to check for environment objects states
            obsListener_ , _, _, _ = envListener.step(action)
            
            special_symbols_ = ls.get_special_symbols(obsListener_)
            state_label = ls.return_symbol(special_symbols, special_symbols_, state_label)
    
            # Get reward machine state
            for u1_ in range(agent.n_rm_states):
                u2_ = agent.rm.get_next_state(u1_, state_label)
                reward_rm = agent.rm.delta_r[u1_][u2_].get_reward()

                u1_ = rm_state_onehot(u1_, agent.n_rm_states)
                u2_ = rm_state_onehot(u2_, agent.n_rm_states)
                agent.store_transition(observation, u1_, action,
                    reward_rm, observation_, u2_, done)


            #if done:
            #    print("||",state_label,"||")
            #    print(u1,u2,"\n",reward_rm,score)
            
            u2 = agent.rm.get_next_state(u1, state_label)
            agent.learn()

            ## Update params
            special_symbols = special_symbols_
            
            #obsListener = obsListener_
            u1 = u2
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

def ppo_learning(args):
    """
    Runs an agent with PPO algorihtm,
    Proximal Policy Optimization

    Parameters:
    ------------
    args: dict
        command line arguments (args.num_games, ...)

    Returns:
    ------------
    avg_reward: list
        list of average rewards every AVG_FREQ num episodes
 
    """
    algorithm = args.algo + "_fully"
    env_name = args.env_name

    # make environement and define observation format with Wrappers
    env = gym.make(args.env_name)
    
    # fully obs wrapper for Event Listener
    envListener = deepcopy(env)
    envListener = FullyObsWrapper(envListener)
    envListener = ImgObsWrapper(envListener) # Get rid of the 'mission' field

    # partial obs wrapper for agent 
    env = RGBImgObsWrapper(env)
    env = ImgObsWrapper(env) # Get rid of the 'mission' field

    env.seed(0)       # sets the seed
    envListener.seed(0)

    obs = env.reset()
    #print(obs.shape)
    obsListener = envListener.reset()

    # PPO hyperparameters
    num_epochs = 3
    mem_size = 512*5
    batch_size = 512
    eps = 0.2
    grad_eps = 0.5
    value_coef = 0.5
    entropy_coef = 0.01
	
    temp = env_name.split("-")[1]
    name2indx_dict = {"DoorKey":1, "Unlock":2, "Empty":3, "KeyCorridorS3R1":4,
                "KeyCorridorS3R2":5, "KeyCorridorS3R3":6, "KeyCorridorS4R3":7}
    env_indx = name2indx_dict[temp]
    rm = RewardMachine("minigrid_reward_machines.json", env_indx) # load Reward Machine

    # Define network and optimizer
    obs = reshape_obs(obs)
    in_channels = obs.shape[0]
    input_dims = obs.shape
    print(obs.shape)
    feature_dim = 256
    num_actions = env.action_space.n
    encoder = PPOEncoder(in_channels=in_channels, input_dims=input_dims, feature_dim=feature_dim)
    policy = PPOPolicy(encoder=encoder, feature_dim=feature_dim, num_actions=num_actions)

    # Define optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4,eps=1e-5)

    # Define temporary storage
    storage = Storage(obs.shape, mem_size)

    printFile = open('../results/' + 'results_' +algorithm+ args.env_name + '.txt', 'w')

    scores, avg_scores, std_scores, eps_history, steps_array = [], [], [], [], []

    best_score = -np.inf


    # logical symbols class
    ls = logical_symbols()
 
    
    "Run training"
    
    print('Episode\t','Steps\t','Score\t',
            'Best_Score\t','Epsilon\t', file=printFile)
    for i in range(args.num_games):
        done = False
        observation = env.reset()
        observation = T.tensor([reshape_obs(observation)], dtype=T.float)
        obsListener = envListener.reset()
        
        state_label = ""
        special_symbols = ls.get_special_symbols(obsListener)
        u1 = rm.u0 # initial state from reward machine

        n_steps = 0
        score = 0
        policy.eval()
        while not done:
            action, log_prob, value = policy.act(observation)
            observation_, reward, done, info = env.step(action)
            
            score += reward
            
            # Run parallel environment to check for environment objects states
            obsListener_ , _, _, _ = envListener.step(action)
            special_symbols_ = ls.get_special_symbols(obsListener_)
            state_label = ls.return_symbol(special_symbols, special_symbols_, state_label)
            # Get reward machine state 
            u2 = rm.get_next_state(u1, state_label)
            reward_rm = rm.delta_r[u1][u2].get_reward()
            
            # Store data            
            storage.store(observation, action, reward_rm, 
                    done, info, log_prob, value)

            # Update params
            u1 = deepcopy(u2)
            special_symbols = deepcopy(special_symbols_)
            #obsListener = deepcopy(obsListener_)
            observation_ =T.tensor([reshape_obs(observation_)], dtype=T.float)
            observation = observation_
            n_steps += 1

        # Add the last observation to collected data
        _, _, value = policy.act(observation)
        storage.store_last(observation, value)
        # Compute return and advantage
        storage.compute_return_advantage()

        # Learning process of the policy
        policy.train()
        for epoch in range(num_epochs):
            # Iterate over batches of transitions
            generator = storage.get_generator(batch_size)
            for batch in generator:
                b_obs, b_action, b_log_prob, b_value, \
                        b_returns, b_advantage = batch

                # Get current policy outputs
                new_dist, new_value = policy(b_obs)
                new_log_prob = new_dist.log_prob(b_action)

                # Clipped policy objective
                ratio = torch.exp(new_log_prob - b_log_prob)
                surr1 = ratio * b_advantage
                surr2 = torch.clamp(ratio, 1.0-eps, 1.0+eps)*b_advantage
                pi_loss = -torch.min(surr1, surr2).mean()

                # Clipped value function objective
                clipped_value = b_value + (new_value+b_value).clamp(min=-eps, max=eps)
                vf_loss = torch.max((new_value-b_returns).pow(2), (clipped_value - b_returns).pow(2))
                value_loss = vf_loss.mean()

                # Entropy loss
                entropy_loss = new_dist.entropy().mean()

                # Backpropagate losses
                loss = pi_loss + value_coef*value_loss - entropy_coef*entropy_loss
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_eps)

                # Update policy
                optimizer.step()
                optimizer.zero_grad()


        scores.append(score)
        steps_array.append(n_steps)
        avg_scores.append(np.mean(scores[-100:]))
        std_scores = np.append(std_scores, np.std(scores[-100:]))

        print('%d\t' %i, '%d\t' %n_steps,
                '%.2f\t' %score,'%.2f\t' %best_score, file=printFile)

        if score > best_score:
            best_score = score

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

elif args.algo == "ppo_learning":
    avg_reward = ppo_learning(args=args)
else:
    raise NotImplementedError("To be implemented")
    exit(1)

# save dataframe
name = str(args.algo) + "_" + str(args.env_name) + "_" + str(args.num_games)

episodes = np.array([i for i in range(args.num_games) if i%PRINT_FREQ == 0])
df = pd.DataFrame(data= {"episodes": episodes, "avg_reward": avg_reward})
df.to_csv(path_or_buf="../results/"+name + "_fully" + ".csv")

# Plot results
fig_name = name + ".png"
plt.plot(avg_reward)
plt.xlabel("Episode")
plt.ylabel("Avg reward")
plt.title(f"Training Agent: {args.algo}, env: {args.env_name}")
plt.savefig("../plots/"+"fully_" + fig_name,  format="png")
t1 = time.time()
dt = t1 - t0
print("--- finished %s ---" % round(dt, 3))
exit(0)
