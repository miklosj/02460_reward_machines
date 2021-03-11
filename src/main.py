from envs import make_env, minecraft_RM
from agents.Tab_Qlearning import Tab_Qlearning_agent
from agents.QRM import QRM_agent
from agents.DQRM.train_dqn import DQRM_train

m = 5
n = 5
resources = [6, 20]
env_name = "Minecraft"

env = make_env.make(env_name, m, n, resources)
#DQRM_train(env)
##Tab_Qlearning_agent(env)
QRM_agent(env)


