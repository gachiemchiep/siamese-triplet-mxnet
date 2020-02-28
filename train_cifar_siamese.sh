#!/usr/bin/env bash

TIMESTAMP=$(date "+%Y%m%d%H%M")
DATASET=cifar10
#DATASET=cifar100

python train_cifar_siamese.py --num-gpus 1 --batch-size 512 --model cifar_resnet20_v2 \
                            --num-epochs 400 --save-period 50 --dataset ${DATASET}\
                            --save-dir snapshots/${DATASET}/siamese/${TIMESTAMP}/
