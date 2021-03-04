import numpy as np
from minecraft import Minecraft
import matplotlib.pyplot as plt
    
def maxAction(Q, state, actions):
    values = np.array([Q[state[0], a] for a in actions])
    action = np.argmax(values)
    return actions[action]

if __name__ == '__main__':
    resources = [6, 20]
    env = Minecraft(5,5,resources)
    
    alpha = 0.1
    gamma = 0.99
    eps = 1.0
    
    totalMeanRewards = np.array([0.0])
    Episodebatch = np.array([1])
    Q = {}
    for state in range(env.state_space[0]+1):
        for action in env.actions:
            Q[state, action] = 0
            
    numGames = 10000
    totalRewards = np.zeros(numGames)
    env.render()
    for i in range(numGames):
        done = False
        epRewards = 0
        state = env.reset()
        max_steps = 3000
        step = 0
        while not done:
            step += 1
            rand = np.random.random()
            action = maxAction(Q, state, env.actions) if rand<(1-eps)else env.actionSpaceSample()
            state_, reward, done, info = env.step(action)
            
            epRewards += reward
            action_ = maxAction(Q, state_, env.actions)
            Q[state[0], action] = Q[state[0], action] + alpha*(reward+ \
                                            gamma *Q[state_[0], action_] - Q[state[0], action])
            state[0] = state_[0]

            if step >= max_steps:
                # Sets dead end
                done = True

        if eps-2/numGames > 0:
            eps -= 2 / numGames
        else:
            eps = 0

        totalRewards[i] = epRewards
        
        if i % 1000 == 0:
            totalMeanRewards = np.append(totalMeanRewards, np.mean(totalRewards[:-1000]))
            Episodebatch = np.append(Episodebatch, i)
            print("numGames:", i, "\tMean reward: ", totalMeanRewards[-1], "\n")
        
    plt.plot(Episodebatch, totalMeanRewards)
    plt.show()
