from agents.utils import maxAction_Hierarchical as maxAction
import numpy as np

def Hierarchical(env):

    alpha = 0.5
    gamma = 0.9
    eps = 0.1

    totalMeanRewards = np.array([0.0])
    Episodebatch = np.array([1])

    Q = {}
    for state in range(env.state_space[0]):
        for action in env.actions:
            for rmstate in range(env.rmstate_space):
                Q[state, action, rmstate] = 0.0

    numGames = 100000
    totalRewards = np.array([0])
    env.render()
    for i in range(numGames):
        done = False
        epRewards = 0
        state, rmstate = env.reset()
        max_steps = 3000
        step = 0
        while not done:
            step += 1
            rand = np.random.random()
            action = maxAction(Q, state, env.actions, rmstate) if rand<(1-eps)else env.actionSpaceSample()

            state_, rmstate_, reward, done, info = env.step(action)

            epRewards += reward

            action_ = maxAction(Q, state_, env.actions, rmstate_)

            # Hierarchical learning
            Q_rmstate_ = env.give_rmstate(rmstate_, state_)
            Q_reward = env.give_reward(rmstate, Q_rmstate_)
            if Q_rmstate_ == 2:
                Q_rmstate_ = 1
            Q[state, action, rmstate] = Q[state, action, rmstate] + alpha* (Q_reward+ gamma *Q[state_, action_, Q_rmstate_] - Q[state, action, rmstate])

            state = state_
            rmstate = rmstate_

        #if eps-2/numGames > 0:
        #    eps -= 2 / numGames
        #else:
        #    eps = 0

        totalRewards = np.append(totalRewards, epRewards)

        if i % 1000 == 0:
            totalMeanRewards = np.append(totalMeanRewards, np.mean(totalRewards[:-1000]))
            Episodebatch = np.append(Episodebatch, i)
            print("numGames:", i, "\tMean reward: ", totalMeanRewards[-1], "\n")

