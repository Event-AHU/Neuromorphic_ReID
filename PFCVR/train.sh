#!/bin/bash
DATASET_NAME="T2I_VeRi_new"

CUDA_VISIBLE_DEVICES=1 \
python train.py \
--name VPT2I_detr_all \
--batch_size 5 \
--lr 1e-5 \
--MLM \
--dataset_name $DATASET_NAME \
--loss_names 'sdm+mlm+id+part+mim' \
--num_epoch 60