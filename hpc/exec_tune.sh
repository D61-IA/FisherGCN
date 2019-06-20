#!/bin/bash

#SBATCH --time=23:55:0
#SBATCH --nodes=1
#SBATCH --mem=128g
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:1

module load tensorflow/1.13.1-py36-gpu

module load python/3.6.5
pip install --user --upgrade --ignore-installed numpy hyperopt ray

python3 scripts/hyperopt_search.py $@
