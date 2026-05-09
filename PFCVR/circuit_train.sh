#!/bin/bash

DATASET_NAME="T2I_VeRi"

# 初始化浮点数变量
mlm_loss_weight=0.1
id_loss_weight=0.5
part_loss_weight=0.2
mim_loss_weight=0.1
mim_masked_rate=0.25

GPU=0
# 设置循环次数
num_iterations=10

# 执行循环
for i in {1..15}; do
    # 输出当前的浮点数
    # echo "当前的 mlm_loss_weight 值: $mlm_loss_weight"
    # echo "当前的 id_loss_weight 值: $id_loss_weight"
    # echo "当前的 part_loss_weight 值: $part_loss_weight"
    # echo "当前的 mim_loss_weight 值: $mim_loss_weight"

    name="mlm_$mlm_loss_weight+id_$id_loss_weight+part_$part_loss_weight+mim_$mim_loss_weight+mask_rate_$mim_masked_rate"
    # name="mlm_$mlm_loss_weight+id_$id_loss_weight+part_$part_loss_weight"
    # echo $name

    CUDA_VISIBLE_DEVICES=$GPU \
    python train.py \
    --name $name \
    --img_aug \
    --batch_size 5 \
    --lr 1.5e-5 \
    --MLM \
    --dataset_name $DATASET_NAME \
    --loss_names 'sdm+mlm+id+part' \
    --mim_masked_rate $mim_masked_rate \
    --mlm_loss_weight $mlm_loss_weight \
    --id_loss_weight $id_loss_weight \
    --part_loss_weight $part_loss_weight \
    --mim_loss_weight $mim_loss_weight \
    --num_epoch 60 \
    --augmentation

    # 使用 bc 进行浮点数运算，counter 加 0.1
    mlm_loss_weight=$(echo "$mlm_loss_weight + 0.1" | bc)
    # id_loss_weight=$(echo "$id_loss_weight + 0.1" | bc)
    # part_loss_weight=$(echo "$part_loss_weight + 0.1" | bc)
    mim_loss_weight=$(echo "$mim_loss_weight + 0.1" | bc)
    # mim_masked_rate=$(echo "$mim_masked_rate + 0.1" | bc)
    # attr_loss_weight=$(echo "$attr_loss_weight + 0.5" | bc)
    # sleep 5
done
