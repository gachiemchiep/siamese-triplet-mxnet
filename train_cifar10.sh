#!/usr/bin/env bash

TIMESTAMP=$(date "+%Y%m%d%H")

python train_cifar10.py --num-gpus 1 --batch-size 512 --model cifar_resnet20_v2 \
                            --num-epochs 40 \
                            --save-dir snapshots/${TIMESTAMP}
