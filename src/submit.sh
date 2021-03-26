#!/bin/sh
#BSUB -q gpuv100
#BSUB -gpu "num=1"
#BSUB -J Job
#BSUB -n 1
#BSUB -W 03:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -o logs/log_%J.out
#BSUB -e logs/log_%J.err

mkdir -p logs
pip3 install numpy --user
pip3 install torch --user
pip3 install matplotlib --user
pip3 install gym-minigrid --user 
pip3 install gym --user
echo "Running script..."

ALGO_LIST=("ddqn_learning")
ENV_NAME_LIST=("MiniGrid-DoorKey-5x5-v0")
NUM_GAMES=1000

for ALGO in $ALGO_LIST
do
    for ENV_NAME in $ENV_NAME_LIST
    do
    python3 main.py --algo=$ALGO --env_name=$ENV_NAME --num_games=$NUM_GAMES
    done
done
