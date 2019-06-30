#!/bin/bash

MODELS=( "gcn" "fishergcn" "gcnT" "fishergcnT" )
SMALLDATA=( "cora" "citeseer" )
LARGEDATA=( "pubmed" )

EXEC="hpc/exec_grid.sh"

# these configurations are selected based on testing accuracy on cora+citeseer
# see hyperopt_search.py for details
GCN_CONFIG="--lrate 0.01 --dropout 0.5 --weight_decay 0.0005 --hidden 64"
FISHER_CONFIG="$GCN_CONFIG --fisher_noise 0.1 --fisher_rank 10"

REPEAT_ARGS="--randomsplit 20 --repeat 10 --data_seed 2019"

REPEAT_ARR=(
    "--randomsplit 4 --repeat 10 --data_seed 2019"
    "--randomsplit 4 --repeat 10 --data_seed 2023"
    "--randomsplit 4 --repeat 10 --data_seed 2027"
    "--randomsplit 4 --repeat 10 --data_seed 2031"
    "--randomsplit 4 --repeat 10 --data_seed 2035"
)

for DATA in ${SMALLDATA[@]}; do
  sbatch --mem 8g -o ${DATA}_gcn.log        ${EXEC} ${DATA} gcn        ${REPEAT_ARGS} ${GCN_CONFIG}
  sbatch --mem 8g -o ${DATA}_fishergcn.log  ${EXEC} ${DATA} fishergcn  ${REPEAT_ARGS} ${FISHER_CONFIG}
  sbatch --mem 8g -o ${DATA}_gcnT.log       ${EXEC} ${DATA} gcnT       ${REPEAT_ARGS} ${GCN_CONFIG}
  sbatch --mem 8g -o ${DATA}_fishergcnT.log ${EXEC} ${DATA} fishergcnT ${REPEAT_ARGS} ${FISHER_CONFIG}
done

for DATA in ${LARGEDATA[@]}; do
  for i in 0 1 2 3 4; do
    sbatch --mem 48g -o ${DATA}_gcn_${i}.log        ${EXEC} ${DATA} gcn        ${REPEAT_ARR[i]} ${GCN_CONFIG}
    sbatch --mem 48g -o ${DATA}_fishergcn_${i}.log  ${EXEC} ${DATA} fishergcn  ${REPEAT_ARR[i]} ${FISHER_CONFIG}
    sbatch --mem 64g -o ${DATA}_gcnT_${i}.log       ${EXEC} ${DATA} gcnT       ${REPEAT_ARR[i]} ${GCN_CONFIG}
    sbatch --mem 64g -o ${DATA}_fishergcnT_${i}.log ${EXEC} ${DATA} fishergcnT ${REPEAT_ARR[i]} ${FISHER_CONFIG}
  done
done

echo you are all done
