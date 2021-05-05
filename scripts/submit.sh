#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J PPO
#BSUB -n 1
#BSUB -W 24:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o logs/log_%J.out
#BSUB -e logs/log_%J.err

mkdir -p logs
pip3 install pandas --user
pip3 install numpy --user
pip3 install torch --user
pip3 install matplotlib --user
pip3 install gym-minigrid --user 
pip3 install gym --user
echo "Running script..."

ALGO_LIST=("ppo_learning")
ENV_NAME_LIST=("MiniGrid-DoorKey-8x8-v0")
NUM_GAMES=10000

for ALGO in $ALGO_LIST
do
    for ENV_NAME in $ENV_NAME_LIST
    do
    python3 main_fully_obs.py --algo=$ALGO --env_name=$ENV_NAME --num_games=$NUM_GAMES
    done
done
