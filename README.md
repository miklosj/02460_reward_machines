# Reinforcement Learning for Rewards Machines
 
 ## About
 ## Requirements
 ## Usage
* Syntax: `python main.py --algorithm=<algorithm> --environment=<environment> --num_steps=<num_steps>`
 
 ## Structure

```
--- /src
      |
      --- main.py
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
