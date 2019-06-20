#!/bin/bash

EXEC="hpc/exec_tune.sh"
sbatch ${EXEC} gcn
sbatch ${EXEC} fishergcn
