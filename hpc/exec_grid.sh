#!/bin/bash

#SBATCH --time=23:55:0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#module load tensorflow/1.13.1-py36-gpu
module load tensorflow/2.0.0a-py36-gpu
python3 scripts/benchmark_grid.py $@
