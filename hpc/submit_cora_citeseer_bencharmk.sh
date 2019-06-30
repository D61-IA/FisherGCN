#!/bin/bash

DATAS=( "cora" "citeseer" )

for DATA in ${DATAS[@]}; do
    sbatch --mem  8g hpc/benchmark.sh $DATA gcn        --randomsplit 30
    sbatch --mem  8g hpc/benchmark.sh $DATA fishergcn  --randomsplit 30
    sbatch --mem 32g hpc/benchmark.sh $DATA gcnT       --randomsplit 15 --data_seed 2019
    sbatch --mem 32g hpc/benchmark.sh $DATA gcnT       --randomsplit 15 --data_seed 2034
    sbatch --mem 32g hpc/benchmark.sh $DATA fishergcnT --randomsplit 15 --data_seed 2019
    sbatch --mem 32g hpc/benchmark.sh $DATA fishergcnT --randomsplit 15 --data_seed 2034
done

echo all done
