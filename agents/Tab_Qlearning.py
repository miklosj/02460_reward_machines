import numpy as np
from agents.utils import maxAction_Qlearning as maxAction

def Tab_Qlearning_agent(env): 
	
    alpha = 0.5
    gamma = 0.9
    eps = 0.1
    
    totalMeanRewards = np.array([0.0])
    Episodebatch = np.array([1])
	
    Q = {}
    for state in range(env.state_space[0]):
        for action in env.actions:
            Q[state, action] = 0.0
            
    numGames = 100000
    totalRewards = np.array([0])
    env.render()
    for i in range(numGames):
        done = False
        epRewards = 0
        state , rmstate = env.reset()
        max_steps = 3000
        step = 0
        while not done:
            step += 1
            rand = np.random.random()
            action = maxAction(Q, state, env.actions) if rand<(1-eps) else env.actionSpaceSample()
			
            state_, rmstate_, reward, done, info = env.step(action)
            
            epRewards += reward
			
            action_ = maxAction(Q, state_, env.actions)
            Q[state, action] = Q[state, action] + alpha*(reward+ \
                                            gamma *Q[state_, action_] - Q[state, action])
            state = state_; rmstate= rmstate_
        #if eps-2/numGames > 0:
        #    eps -= 2 / numGames
        #else:
        #    eps = 0
            #if step >= max_steps:
             #   done = true

        totalRewards = np.append(totalRewards, epRewards)
        
        if i % 1000 == 0:
            totalMeanRewards = np.append(totalMeanRewards, np.mean(totalRewards[:-1000]))
            Episodebatch = np.append(Episodebatch, i)
            print("numGames:", i, "\tMean reward: ", totalMeanRewards[-1], "\n")
