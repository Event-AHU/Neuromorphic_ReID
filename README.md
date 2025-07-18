# Neuromorphic_ReID

**Requirements**
```
We use a single RTX3090 24G GPU for training and evaluation.
```

**Basic Environment**
```
Python 3.9.16
pytorch 1.12.1
torchvision 0.13.1
```

**Basic Environment**
```
Download the datasets (MARS, EvReID) and then unzip them to your_dataset_dir.
You can get these datasets in the following **Links**:
**MARS:** ----
**EvReID:** ----
```

## :car: Run TriPro-ReID
For example, if you want to run this method on MARS, you need to modify the bottom of configs/vit_mars_clipreid.yml to
```
DATASETS:
   NAMES: ('MARS')
   ROOT_DIR: ('your_dataset_dir')
OUTPUT_DIR: 'your_output_dir'
```
And you need to organize your dataset as:
```
MARS:
  --- RGB
    --- train
    --- test
  --- event
    --- train
    --- test
  --- info
 ```
 
Then, run
```
CUDA_VISIBLE_DEVICES=0 python train_mars.py
```

## :car: Evaluation
For example, if you want to test methods on MARS, run
```
CUDA_VISIBLE_DEVICES=0 python test.py
```

