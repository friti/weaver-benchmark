#!/bin/bash

PREFIX=$1
MODEL_CONFIG=$2
DATA_CONFIG=$3
PATH_TO_SAMPLES=$4
CLUSTERID=$5
WORKDIR=`pwd`

echo ${MODEL_CONFIG}

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/etc/profile.d/conda.sh" ]; then
        . "/usr/etc/profile.d/conda.sh"
    else
        export PATH="/usr/bin:$PATH"
    fi
fi
unset __conda_setup

#conda init 
conda activate weaver
cd /afs/cern.ch/work/f/friti/softtaus/parT/part_training/training 
cd weaver-benchmark/weaver-core/weaver 
mkdir -p output
rm -rf output/*
pwd



# Gerosa : https://gitlab.nrp-nautilus.io/rgerosa/particlenetrun2ul/-/blob/main/ak4_training_latest/config_1gpu_48gb/weaver-job-ak4-transformer-ch.yaml?ref_type=heads
# Training, using 1 GPU

echo "Starting training"

python3 train.py \
 --data-train ${PATH_TO_SAMPLES}'/samplesv2_job_0_thread0.root' \
 --data-val ${PATH_TO_SAMPLES}'/samplesv2_job_0_thread3.root' \
 --data-config tau_tagging/data/${DATA_CONFIG} \
 --network-config tau_tagging/networks/${MODEL_CONFIG} \
 --model-prefix output/${PREFIX} \
 --gpus 0 --batch-size-train 512 --batch-size-val 512 --start-lr 5e-3 --num-epochs 20 --optimizer ranger \
 --log output/${PREFIX}.train.log \
 --fetch-step-train 1 --fetch-step-val 1 --num-workers-train 1 --num-workers-val 1 \
 --weaver-mod class #--remake-weights
 #--data-train ${PATH_TO_SAMPLES}'/*8*thread0.root' \
 #--data-val ${PATH_TO_SAMPLES}'/*8*thread4.root' \
 #--data-train ${PATH_TO_SAMPLES}'/*thread0.root' ${PATH_TO_SAMPLES}'/*thread1.root' ${PATH_TO_SAMPLES}'/*thread2.root' ${PATH_TO_SAMPLES}'/*thread3.root' \


echo "Starting prediction"

python3 train.py --predict \
 --data-test ${PATH_TO_SAMPLES}'/samplesv2_job_0_thread0.root' \
 --data-config tau_tagging/data/${DATA_CONFIG} \
 --network-config tau_tagging/networks/${MODEL_CONFIG} \
 --model-prefix output/${PREFIX} \
 --gpus 0 --batch-size-test 1024 \
 --fetch-step-test 1 --num-workers-test 1 \
 --predict-output output/${PREFIX}_predict.root \
 --weaver-mod class
 #--data-test ${PATH_TO_SAMPLES}'/*.root' \

[ -d "runs/" ] && tar -caf output.tar output/ runs/ || tar -caf output.tar output/
mv output.tar /afs/cern.ch/work/f/friti/softtaus/parT/part_training/training/weaver-benchmark/condor/outputs/output_${CLUSTERID}.tar




