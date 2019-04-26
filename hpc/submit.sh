#!/bin/bash
trim()
{
    echo $(echo -e "$@" | tr -d '[:space:]')
}

DATASETS=("all")
LRATES=(0.01)
DROPOUTS=(0.5)
REGS=(0.0005)
HIDDEN=(64)
FISHERNOISE=(0.01 0.1 1.0 2.0)
FISHERRANK=(10)
FISHERFREQ=(0 1)
FISHERADV=(0 1)
EARLY=(1)

MODEL=$1
SBATCH_TIME="47:59:0"
REPEAT=20

for ds in "${DATASETS[@]}"; do
    if [[ "$ds" == "pubmed" ]]; then
        SBATCH_MEM=15g
    elif [[ "$ds" == "citeseer" ]]; then
        SBATCH_MEM=15g
    else
        SBATCH_MEM=5g
    fi

    for lrate in ${LRATES[@]}; do
    for dropout in ${DROPOUTS[@]}; do
    for reg in ${REGS[@]}; do
    for early in ${EARLY[@]}; do
    for hidden in ${HIDDEN[@]}; do

                    if [[ "${MODEL}" == "fishergcn" ]] || \
                         [[ "${MODEL}" == "fishergcnT" ]] ; then

                        for noise in ${FISHERNOISE[@]}; do
                        for rank in ${FISHERRANK[@]}; do
                        for freq in ${FISHERFREQ[@]}; do
                        for adv in ${FISHERADV[@]}; do
                            JOB="${MODEL}_$(trim ${ds})_drop${dropout}_lr${lrate}_reg${reg}_hidden${hidden}_early${early}_noise${noise}_rank${rank}_freq${freq}_adv${adv}"
                            mkdir -p "outputs/"
                            sbatch -o outputs/log_${JOB}.txt -e outputs/log_${JOB}.txt --job-name=${JOB} --mem ${SBATCH_MEM} --time ${SBATCH_TIME} \
                              hpc/exec.sh --model ${MODEL} --dataset ${ds} --dropout ${dropout} --learning_rate ${lrate} --repeat ${REPEAT} \
                                 --weight_decay ${reg} --early_stop ${early} --hidden1 ${hidden} \
                                 --fisher_noise ${noise} --fisher_rank ${rank} --fisher_freq ${freq} --fisher_adversary ${adv}
                            sleep 3
                        done
                        done
                        done
                        done

                    else
                        JOB="${MODEL}_$(trim ${ds})_drop${dropout}_lr${lrate}_reg${reg}_hidden${hidden}_early${early}"
                        mkdir -p "outputs/"
                        sbatch -o outputs/log_${JOB}.txt -e outputs/log_${JOB}.txt --job-name=${JOB} --mem ${SBATCH_MEM} --time ${SBATCH_TIME} \
                          hpc/exec.sh --model ${MODEL} --dataset ${ds} --dropout ${dropout} --learning_rate ${lrate} --repeat ${REPEAT} \
                                      --weight_decay ${reg} --early_stop ${early} --hidden1 ${hidden}
                        sleep 3
                    fi

    done
    done
    done
    done
    done
done

