import numpy as np

def maxAction_Qlearning(Q, state, actions):
    values = np.array([Q[state, a] for a in actions])
    action = np.argmax(values)
    return actions[action]

def maxAction_QRM(Q, state, actions, rmstate):
    values = np.array([Q[state, a, rmstate] for a in actions])
    action = np.argmax(values)
    return actions[action]

def maxAction_Hierarchical(Q, state, actions, rmstate):
    values = np.array([Q[state, a, rmstate] for a in actions])
    action = np.argmax(values)
    return actions[action]


