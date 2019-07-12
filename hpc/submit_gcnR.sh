#!/bin/bash

SMALLDATA=( "cora" "citeseer" )

EXEC="hpc/exec_grid.sh"

GCN_CONFIG="--lrate 0.01 --dropout 0.5 --weight_decay 0.0005 --hidden 64"
REPEAT_ARGS="--randomsplit 20 --repeat 10 --data_seed 2019"

for DATA in ${SMALLDATA[@]}; do
  for noise in 1e-4 1e-3 1e-2 0.1 0.2 0.5; do
    sbatch --mem 8g -o ${DATA}_gcnR_noise${noise}.log ${EXEC} ${DATA} gcnR        ${REPEAT_ARGS} ${GCN_CONFIG} --flip_prob ${noise}
  done
done

echo you are all done
