import collections
import numpy as np
import matplotlib.pyplot as plt
import gym

def plot_learning_curve(x, scores, std_scores,epsilons, filename, lines=None):
    fig=plt.figure()
    ax=fig.add_subplot(111, label="1")
    ax2=fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="#FF6F00")
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Epsilon", color="#FF6F00")
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')
    ax.tick_params(axis='y', colors="#FF6F00")


    ax2.plot(x, scores, color="#5E35B1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.set_ylabel('Score', color="#5E35B1")
    ax2.yaxis.set_label_position('left')
    ax2.tick_params(axis='y', colors="#5E35B1")
    ax2.fill_between(x, scores+std_scores, scores-std_scores,
            color = "#5E35B1", edgecolor="#FFFFFF", alpha=0.2)


    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

def make_env(env_name, shape=(84,84,1), repeat=4, clip_rewards=False,
             no_ops=0, fire_first=False):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)

    return env
