#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

TIMESTAMP=$(date "+%Y%m%d%H%M")
DATASET=cifar10
#DATASET=cifar100

python train_cifar_triplet_semihard.py --num-gpus 1 --n-classes 8 --n-samples 16 --model cifar_resnet20_v2 \
                            --num-epochs 400 --save-period 50 --dataset ${DATASET}\
                            --save-dir snapshots/${DATASET}/triplet_semihard/${TIMESTAMP}/
