#!/bin/bash

#SBATCH --time=47:55:0
#SBATCH --mem=6g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

module load tensorflow/1.13.1-py36-gpu
cd gcn && python3 benchmark.py $@
