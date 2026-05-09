import sys
sys.path.append('..')
import os
import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import os.path as op
import torch
import time

from datasets import build_dataloader
from processor.processor import do_train
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(args,num_classes,chkpt_dir):
    # build model
    model = build_model(args, num_classes)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    # breakpoint() 
    model.load_state_dict(checkpoint['model'], strict=False)
    return model.cuda()

def visual_image(loader, model):
    for n_iter, batch in enumerate(loader):
        batch = { k: (v.to(device) if k !="boxes" else v) for k, v in batch.items() }
        # run MAE
        y, x, mask = model.forward_mim(batch)
        y = model.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()
        y = (y - y.min()) / (y.max() - y.min())

        x = model.unpatchify(model.patchify(x))
        # breakpoint()
        # print(y)

        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, 16**2 *3)  # (N, H*W, p*p*3)
        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        
        x = torch.einsum('nchw->nhwc', x).detach().cpu()

        # masked image
        im_masked = x * (1 - mask)

        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask

        # make the plt figure larger
        plt.rcParams['figure.figsize'] = [24, 24]

        plt.subplot(1, 4, 1)
        show_image(x[0], "original")

        plt.subplot(1, 4, 2)
        show_image(im_masked[0], "masked")

        plt.subplot(1, 4, 3)
        show_image(y[0], "reconstruction")

        plt.subplot(1, 4, 4)
        show_image(im_paste[0], "reconstruction + visible")

        plt.savefig("test.jpg")
        break



def visualize_image(image):
    # 将图像归一化到 [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    plt.imshow(image)
    plt.axis('off')  # 关闭坐标轴
    plt.show()


if __name__ == '__main__':
    args = get_args()
    args.MLM = True
    args.img_aug = True
    name = args.name
    
    device = "cuda"

    # get image-text pair datasets dataloader
    train_loader, val_img_loader, val_txt_loader, num_classes = build_dataloader(args)
    model = prepare_model(args,num_classes,chkpt_dir="logs/T2I_VeRi/20241016_204633_mlm_0.5+id_0.5+part_.2+mim_1.0_R1_28.9/best_28.9.pth")

    visual_image(train_loader,model)

