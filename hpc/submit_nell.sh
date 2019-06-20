#!/bin/bash

SBATCH_TIME="47:59:0"
SBATCH_MEM=48g
REPEAT=20
FISHERNOISE=(0.1 0.3 1.0)

# use the default configuration provided by the GCN paper
# note this experiment maybe very time/resource consuming
mkdir -p "outputs/"

sbatch -o outputs/nell_gcn.txt -e outputs/nell_gcn.txt --job-name=nell --mem ${SBATCH_MEM} --time ${SBATCH_TIME} \
  hpc/exec.sh --model gcn --dataset nell.0.001 --dropout 0.1 --learning_rate 0.01 --repeat ${REPEAT} \
              --weight_decay 1e-5 --early_stop 1 --hidden1 64 --save

for noise in ${FISHERNOISE[@]}; do
  JOB="nell_fishergcn${noise}"
  sbatch -o outputs/${JOB}.txt -e outputs/${JOB}.txt --job-name=${JOB} --mem ${SBATCH_MEM} --time ${SBATCH_TIME} \
      hpc/exec.sh --model fishergcn --dataset nell.0.001 --dropout 0.1 --learning_rate 0.01 --repeat ${REPEAT} \
                  --weight_decay 1e-5 --early_stop 1 --hidden1 64 --fisher_noise 0.1 --save
done
