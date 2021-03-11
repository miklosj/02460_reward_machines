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
    
    Q = {}
    for state in range(env.state_space[0]+1):
        for action in env.actions:
            Q[state, action] = 0
            
    numGames = 10000
    totalRewards = np.zeros(numGames)
    env.render()
    for i in range(numGames):
        if i % 1000 == 0:
            print("Starting game...", i)
            print("Mean reward: ", np.mean(totalRewards[:-1000]))
        
        done = False
        epRewards = 0
        state = env.reset()
        
        while not done:
            rand = np.random.random()
            action = maxAction(Q, state, env.actions) if rand<(1-eps)else env.actionSpaceSample()
            state_, reward, done, info = env.step(action)
            
            epRewards += reward
            action_ = maxAction(Q, state_, env.actions)
            Q[state[0], action] = Q[state[0], action] + alpha*(reward+ \
                                            gamma *Q[state_[0], action_] - Q[state[0], action])
            state[0] = state_[0]
            
        if eps-2/numGames > 0:
            eps -= 2 / numGames
        else:
            eps = 0
        totalRewards[i] = epRewards
    plt.plot(totalRewards)
    plt.show()
