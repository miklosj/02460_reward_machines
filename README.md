# Reinforcement Learning for Rewards Machines
 
 ## About
 Reinforcement Learning for Reward Machines. Relevant papers:
 * *Reward Machines: Exploiting Reward Function Structure in Reinforcement Learning* [[pdf]](https://arxiv.org/pdf/2010.03950.pdf)
 * *Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning* [[pdf]](http://proceedings.mlr.press/v80/icarte18a/icarte18a.pdf)
 * *LTL and Beyond: Formal Languages for Reward Function Specification in Reinforcement Learning* [[pdf]](https://www.ijcai.org/Proceedings/2019/0840.pdf)
 * *Learning Reward Machines for Partially Observable Reinforcement Learning* [[pdf]](https://papers.nips.cc/paper/2019/file/532435c44bec236b471a47a88d63513d-Paper.pdf)
 * *Joint Inference of Reward Machines and Policies for Reinforcement Learning* [[pdf]](https://arxiv.org/pdf/1909.05912.pdf)
 
 ## Requirements
 * torch==1.8.0
 * numpy==1.20.1

 ## Usage
* Install requirements: `pip install -r requirements.txt`
* Syntax: `python main.py --algorithm=<algorithm> --environment=<environment> --num_steps=<num_steps>`
 
 ## Options
 * --algorithm: dqn (default dqn)
 * --environment: office (default office)
 * --num_steps: (default 10)
 
 ## Structure

```
--- /src
      |
      --- main.py
      --- requirements.txt
      --- __init__.py
      --- /agents
              |
              --- __init__.py
              --- dqn_algorithm.pu
              --- run_dqn.py
              --- utils.py
              --- TabQ_learning.py
              --- QRM.py
              --- /DQRM
                    |
                    --- 
      --- /RewardMachines
              |
              --- reward_machine.py
              --- rm_utils.py
              --- office.json
      --- /envs
              |
              --- __init__.py
              --- make_env.py
              --- main.py
              --- minecraft_RM.py
              --- minecraft.py
              --- office.py
