#!/bin/bash

MODELS=( "gcn" "fishergcn" "gcnT" "fishergcnT" )
DATAS=( "cora" "citeseer" "pubmed" )
EXEC="hpc/exec_grid.sh"

CONFIG="--lrate 0.01 --dropout 0.5 --weight_decay 0.0005 --hidden 64 --fisher_noise 0.1 --fisher_rank 10"

for DATA in ${DATAS[@]}; do
  for MODEL in ${MODELS[@]}; do
    if [[ $DATA == "cora" ]]; then
        SBATCH_MEM=8g
    elif [[ $DATA == "citeseer" ]]; then
        SBATCH_MEM=8g
    else
        SBATCH_MEM=48g
    fi

    sbatch --mem ${SBATCH_MEM} ${EXEC} $DATA $MODEL --randomsplit 0 --repeat 50 ${CONFIG}
  done
done

echo all done
