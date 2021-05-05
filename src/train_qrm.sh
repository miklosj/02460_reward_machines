#!/bin/sh
#BSUB -q hpc
#BSUB -J QRM_fully_obs
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

python3 main_fully_obs.py --algo=qrm_learning --env_name=MiniGrid-Empty-6x6-v0 --num_games=10000
