from utils.logger import setup_logger
from datasets.make_dataloader import make_dataloader
from model.make_model_clipreid_mm import make_model_mm
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage, make_optimizer_mid_stage
from solver.scheduler_factory import create_scheduler
from solver.lr_scheduler import WarmupMultiStepLR
from loss.make_loss import make_loss
from processor.processor_clipreid_stage1_mm import do_train_stage1
from processor.processor_clipreid_stage2_mm import do_train_stage2
import random
import torch
import numpy as np
import os
import argparse
# from config import cfg_mars
from config import cfg_mars
import os.path as osp
from scipy.io import loadmat
import json

import ast

def bool_str_to_tensor(bool_str: str) -> torch.BoolTensor:
    bool_list = ast.literal_eval(bool_str.strip())
    return torch.tensor(bool_list, dtype=torch.bool)
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/zz1/TriPro/configs/person/vit_aer_clipreid_mm.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg_mars.merge_from_file(args.config_file)
    cfg_mars.merge_from_list(args.opts)
    cfg_mars.freeze()

    set_seed(cfg_mars.SOLVER.SEED)

    if cfg_mars.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)

    output_dir = cfg_mars.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg_mars.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg_mars))

    if cfg_mars.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg_mars)

    model = make_model_mm(cfg_mars, num_class=num_classes, camera_num=camera_num, view_num = view_num)

    loss_func, center_criterion = make_loss(cfg_mars, num_classes=num_classes)

    optimizer_1stage = make_optimizer_1stage(cfg_mars, model)
    scheduler_1stage = create_scheduler(optimizer_1stage, num_epochs = cfg_mars.SOLVER.STAGE1.MAX_EPOCHS, lr_min = cfg_mars.SOLVER.STAGE1.LR_MIN, \
                        warmup_lr_init = cfg_mars.SOLVER.STAGE1.WARMUP_LR_INIT, warmup_t = cfg_mars.SOLVER.STAGE1.WARMUP_EPOCHS, noise_range = None)
    track_train_info_path = osp.join('/media/amax/836e911f-c5c3-4c4b-91f2-41bb8f3f5cb6/DATA/zz1/AER_video/rgb_degrade', 'info/tracks_train_info.mat')

    track_train_mat = loadmat(track_train_info_path)
    track_train = np.array([track_train_mat['start_frame'], track_train_mat['end_frame'], 
                            track_train_mat['person_id'], track_train_mat['cam_id']]).squeeze(1).transpose(1,0)
    train_pid_list = list(set(track_train[:, 2].tolist()))

    label2pid = {label: pid for label, pid in enumerate(train_pid_list)}
    with open(cfg_mars.DATASETS.TRAIN_ATTR, 'r', encoding='utf-8') as f:
        train_pid2attrs = json.load(f)
    pid2attrs = {}
    for k,v in train_pid2attrs.items():
        pid2attrs[k] = bool_str_to_tensor(v)
        
    do_train_stage1(
        cfg_mars,
        model,
        train_loader_stage1,
        optimizer_1stage,
        scheduler_1stage,
        args.local_rank,
        label2pid,
        pid2attrs,
        image_pth=cfg_mars.DATASETS.STAGE1_IMG_FEAT
    )

    mid_optimizer = make_optimizer_mid_stage(cfg_mars, model)

    
    optimizer_2stage, optimizer_center_2stage = make_optimizer_2stage(cfg_mars, model, center_criterion)
    scheduler_2stage = WarmupMultiStepLR(optimizer_2stage, cfg_mars.SOLVER.STAGE2.STEPS, cfg_mars.SOLVER.STAGE2.GAMMA, cfg_mars.SOLVER.STAGE2.WARMUP_FACTOR,
                                  cfg_mars.SOLVER.STAGE2.WARMUP_ITERS, cfg_mars.SOLVER.STAGE2.WARMUP_METHOD)
    
    do_train_stage2(
        cfg_mars,
        model,
        center_criterion,
        train_loader_stage2,
        val_loader,
        mid_optimizer,
        optimizer_2stage,
        optimizer_center_2stage,
        scheduler_2stage,
        loss_func,
        num_query, args.local_rank, label2pid, pid2attrs, cfg_mars.DATASETS.TEST_ATTR
    )