# Reinforcement Learning for Rewards Machines
 
<p align="center">
<img src="/figures/minigrid.PNG" width=250>
</p>

 ## About
 Reward assignment in Reinforcement learning is commonly treated as a black box under the assumption that the agent should learn to navigate the environment without prior knowledge of the rewards and learning from experience. However recently there has been some research regarding the benefits of exploiting the reward structure to speed up learning. We revisit some of the algorithms proposed for this goal and test their performance outside customized environments in a set of more complex environments commonly used for testing other RL algorithms. 
 
 Reinforcement Learning for Reward Machines. Relevant papers:
 * *Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning* [[pdf]](https://arxiv.org/pdf/2010.03950.pdf)
 * *Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning* [[pdf]](http://proceedings.mlr.press/v80/icarte18a/icarte18a.pdf)
 * *LTL and Beyond: Formal Languages for Reward Function Specification in Reinforcement Learning* [[pdf]](https://www.ijcai.org/Proceedings/2019/0840.pdf)
 * *Learning Reward Machines for Partially Observable Reinforcement Learning* [[pdf]](https://papers.nips.cc/paper/2019/file/532435c44bec236b471a47a88d63513d-Paper.pdf)
 * *Joint Inference of Reward Machines and Policies for Reinforcement Learning* [[pdf]](https://arxiv.org/pdf/1909.05912.pdf)

 ## Usage
* Install requirements: `pip install -r requirements.txt`
* Syntax: `python main_fully_obs.py --algo=<algorithm> --env_name=<environment> --num_games=<num_steps>`
* (Example): `python main_fuly_obs.py --algo=qrm_learning --env_name=MiniGrid-DoorKey-5x5-v0 --num_games=100`
 
 ## Options
 * --algorithm: [`random_baseline, q_learning, qrm_learning, crm_learning`] (default `random_baseline`)
 * --env_name: [`MiniGrid-DoorKey-5x5-v0, MiniGrid-DoorKey-8x8-v0` ...] (default `MiniGrid-Empty-8x8-v0`)
 * --num_steps: integer (default 100)

 ## Structure
```
--- requirements.txt
--- /figures
--- /models
--- /plots
--- /scripts
--- /src
      |
      --- main_fully_obs.py
      --- main_part_obs.py
      --- rm_utils.py
      --- utils.py
      --- reward_machine.py
      --- minigrid_reward_machines.json
      --- /Q
            |
            --- __init__.py
            --- q_learning.py
      --- /QRM
            |
            --- __init__.py
            --- qrm_learning.py
      --- /CRM
            |
            --- __init__.py
            --- crm_learning.py
      --- /DQN
            |
            --- __init__.py
            --- deep_q_network.py
            --- dqn_learning.py
            --- replay_memory.py
      --- /DDQN
            |
            --- __init__.py
            --- deep_q_network.py
            --- ddqn_learning.py
            --- replay_memory.py
      --- /DCRM
            |
            --- __init__.py
            --- deep_q_network.py
            --- dcrm_learning.py
            --- replay_memory.py
