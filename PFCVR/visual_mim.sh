#!/bin/bash
DATASET_NAME="T2I_VeRi"

CUDA_VISIBLE_DEVICES=0 \
python mim_visualize.py \
    --name VPT2I \
    --img_aug \
    --batch_size 1 \
    --MLM \
    --dataset_name $DATASET_NAME \
    --loss_names 'sdm+mlm+id+part+mim' 