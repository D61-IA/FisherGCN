#!/bin/bash

MODELS=( "gcn" "fishergcn" "gcnT" "fishergcnT" )
DATAS=( "cora" "citeseer" "pubmed" "amazon_electronics_computers" "amazon_electronics_photo" )

for MODEL in ${MODELS[@]}; do
    sbatch --mem 8g hpc/benchmark.sh cora     $MODEL --randomsplit 20
    sbatch --mem 8g hpc/benchmark.sh citeseer $MODEL --randomsplit 20
done

sbatch --mem 48g hpc/benchmark.sh pubmed gcn        --randomsplit 20
sbatch --mem 48g hpc/benchmark.sh pubmed fishergcn  --randomsplit 20
sbatch --mem 48g hpc/benchmark.sh pubmed gcnT       --randomsplit 20
sbatch --mem 48g hpc/benchmark.sh pubmed fishergcnT --randomsplit 5 --seed 2019
sbatch --mem 48g hpc/benchmark.sh pubmed fishergcnT --randomsplit 5 --seed 2024
sbatch --mem 48g hpc/benchmark.sh pubmed fishergcnT --randomsplit 5 --seed 2029
sbatch --mem 48g hpc/benchmark.sh pubmed fishergcnT --randomsplit 5 --seed 2034

sbatch --mem 64g hpc/benchmark.sh amazon_electronics_computers gcn       --randomsplit 20
sbatch --mem 64g hpc/benchmark.sh amazon_electronics_computers gcnT      --randomsplit 20
sbatch --mem 64g hpc/benchmark.sh amazon_electronics_computers fishergcn --randomsplit 10 --seed 2019
sbatch --mem 64g hpc/benchmark.sh amazon_electronics_computers fishergcn --randomsplit 10 --seed 2029
sbatch --mem 64g hpc/benchmark.sh amazon_electronics_computers fishergcnT --randomsplit 5 --seed 2019
sbatch --mem 64g hpc/benchmark.sh amazon_electronics_computers fishergcnT --randomsplit 5 --seed 2024
sbatch --mem 64g hpc/benchmark.sh amazon_electronics_computers fishergcnT --randomsplit 5 --seed 2029
sbatch --mem 64g hpc/benchmark.sh amazon_electronics_computers fishergcnT --randomsplit 5 --seed 2034

sbatch --mem 64g hpc/benchmark.sh amazon_electronics_photo gcn        --randomsplit 20
sbatch --mem 64g hpc/benchmark.sh amazon_electronics_photo gcnT       --randomsplit 20
sbatch --mem 64g hpc/benchmark.sh amazon_electronics_photo fishergcn  --randomsplit 20
sbatch --mem 64g hpc/benchmark.sh amazon_electronics_photo fishergcnT --randomsplit 10 --seed 2019
sbatch --mem 64g hpc/benchmark.sh amazon_electronics_photo fishergcnT --randomsplit 10 --seed 2029

echo all done
