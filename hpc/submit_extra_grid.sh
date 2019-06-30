#!/bin/bash

MODELS=( "gcn" "fishergcn" "gcnT" "fishergcnT" )
DATA=( "ms_academic_cs" "ms_academic_phy" "amazon_electronics_photo" "amazon_electronics_computers" )
EXEC="hpc/exec_grid.sh"

GCN_CONFIG="--lrate 0.01 --dropout 0.5 --weight_decay 0.0005 --hidden 64"
FISHER_CONFIG="$GCN_CONFIG --fisher_noise 0.2 --fisher_rank 50"
REPEAT_ARGS="--randomsplit 5 --repeat 10 --data_seed 2019 --epochs 100 --early 1"

for DATA in ${SMALLDATA[@]}; do
  sbatch --mem 24g -o ${DATA}_gcn.log        ${EXEC} ${DATA} gcn        ${REPEAT_ARGS} ${GCN_CONFIG}
  sbatch --mem 24g -o ${DATA}_fishergcn.log  ${EXEC} ${DATA} fishergcn  ${REPEAT_ARGS} ${FISHER_CONFIG}
  sbatch --mem 24g -o ${DATA}_gcnT.log       ${EXEC} ${DATA} gcnT       ${REPEAT_ARGS} ${GCN_CONFIG}
  sbatch --mem 24g -o ${DATA}_fishergcnT.log ${EXEC} ${DATA} fishergcnT ${REPEAT_ARGS} ${FISHER_CONFIG}
done

echo you are all done
