# PFCVR

Official PyTorch implementation of

> **T2I-VeRW: Part-level Fine-grained Perception for Text-to-Image Vehicle Retrieval**

We propose **PFCVR**, a Part-level Fine-grained Cross-modal Vehicle Retrieval
framework, and release **T2I-VeRW**, a new large-scale text-to-image vehicle
Re-ID benchmark with 14,668 images of 1,796 identities.

## Requirements

```bash
pip install torch torchvision ftfy regex prettytable easydict tqdm pyyaml matplotlib
```

A single NVIDIA RTX 4090 24 GB GPU is sufficient.

## Training

Edit `train.sh` to point to your data root and run:

```bash
bash train.sh
```

The default command is:

```bash
CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --name PFCVR \
    --batch_size 5 \
    --lr 1e-5 \
    --MLM \
    --dataset_name T2I_VeRi_new \
    --loss_names 'sdm+mlm+id+part+mim' \
    --num_epoch 60
```

Use `--dataset_name T2I_VeRi` to train on the original T2I-VeRI benchmark.
In this codebase, `T2I_VeRi_new` corresponds to **T2I-VeRW** in the paper.

## Evaluation

```bash
python test.py --config_file logs/<RUN_DIR>/configs.yaml
```

## Results

### T2I-VeRI

| Method      | Rank-1 | Rank-5 | Rank-10 | mAP   |
|:-----------:|:------:|:------:|:-------:|:-----:|
| TIPCB       | 16.8   | 44.0   | 58.8    | 18.0  |
| LGUR        | 11.3   | 29.0   | 45.9    | 9.2   |
| SSAN        | 14.2   | 34.3   | 52.6    | 12.2  |
| TFAF        | 20.1   | 49.0   | 66.3    | 15.5  |
| TransReID   | 7.5    | 24.3   | 35.1    | 7.4   |
| HAT         | 16.0   | 40.7   | 56.2    | 13.7  |
| ALBEF       | 22.9   | 49.6   | 66.1    | 21.8  |
| MCANet      | 25.1   | 54.7   | 69.1    | 18.1  |
| IRRA        | 25.1   | 57.8   | 72.4    | 23.7  |
| UP-Person   | 18.8   | 46.4   | 61.3    | 18.5  |
| FMFA        | 21.8   | 53.6   | 69.1    | 21.6  |
| GA-DMS      | 22.2   | 53.6   | 69.0    | 21.5  |
| VFE-TPS     | 22.2   | 53.1   | 69.4    | 21.7  |
| MARS        | 25.5   | 54.3   | 67.6    | 24.9  |
| **PFCVR**   | **29.2** | **60.1** | **75.4** | **25.3** |

### T2I-VeRW

| Method      | Rank-1 | Rank-5 | Rank-10 | mAP   |
|:-----------:|:------:|:------:|:-------:|:-----:|
| TIPCB       | 29.1   | 53.3   | 64.5    | 17.2  |
| LGUR        | 44.2   | 73.0   | 83.1    | 18.5  |
| SSAN        | 39.1   | 67.3   | 78.6    | 21.6  |
| TransReID   | 22.0   | 41.7   | 51.0    | 12.4  |
| ALBEF       | 22.8   | 49.2   | 52.3    | 15.6  |
| IRRA        | 47.3   | 75.6   | 85.3    | 22.8  |
| UP-Person   | 50.7   | 80.6   | 88.7    | 22.6  |
| FMFA        | 47.8   | 78.9   | 87.6    | 22.1  |
| GA-DMS      | 52.2   | 81.4   | 89.9    | 23.5  |
| VFE-TPS     | 53.7   | 82.0   | 89.7    | 24.4  |
| MARS        | 52.8   | 78.9   | 86.3    | 25.2  |
| **PFCVR**   | **55.2** | **82.8** | **90.3** | **26.2** |