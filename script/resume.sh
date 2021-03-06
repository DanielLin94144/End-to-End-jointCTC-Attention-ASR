#!/bin/bash

# $1 : experiment name
# $2 : cuda id

CONFIG="librispeech_asr_resume"

DIR="/Home/daniel094144/End-to-end-ASR-Pytorch/"

echo "Start resuming training process of E2E ASR"
CUDA_VISIBLE_DEVICES=$2 python3 main.py --config config/${CONFIG}.yaml \
    --name $1 \
    --njobs 12 \
    --seed 0 \
    --logdir ${DIR}/log/ \
    --ckpdir ${DIR}/ckpt/ \
    --outdir ${DIR}/result/ \
    --reserve_gpu 0 \
    # --load ${DIR}/ckpt/$1/best_ctc_LibriSpeech.pth \
