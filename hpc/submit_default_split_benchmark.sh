#!/bin/bash

MODELS=( "gcn" "fishergcn" "gcnT" "fishergcnT" )
DATAS=( "cora" "citeseer" "pubmed" "amazon_electronics_computers" "amazon_electronics_photo" )

for DATA in ${DATAS[@]}; do
  for MODEL in ${MODELS[@]}; do
    if [[ $DATA == "cora" ]]; then
        SBATCH_MEM=8g
    elif [[ $DATA == "citeseer" ]]; then
        SBATCH_MEM=8g
    else
        SBATCH_MEM=48g
    fi

    sbatch --mem ${SBATCH_MEM} hpc/benchmark.sh $DATA $MODEL --randomsplit 0 --repeat 20
  done
done

echo all done
