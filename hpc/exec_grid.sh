#!/bin/bash

#SBATCH --time=1:59:59
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

module load tensorflow/1.14.0-py36-gpu
python3 scripts/benchmark_grid.py $@
