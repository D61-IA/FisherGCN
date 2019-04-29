#!/bin/bash

MODELS=( "gcn" "fishergcn" "gcnT" "fishergcnT" )
DATAS=( "cora" "citeseer" "pubmed" "amazon_electronics_computers" "amazon_electronics_photo" )

for DATA in ${DATAS[@]}; do
  for MODEL in ${MODELS[@]}; do
    sbatch hpc/benchmark.sh $DATA $MODEL
  done
done
echo all done
