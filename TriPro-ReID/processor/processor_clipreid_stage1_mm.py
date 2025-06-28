import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
from tqdm import tqdm
def do_train_stage1(cfg,
             model,
             train_loader_stage1,
             optimizer,
             scheduler,
             local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD 

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    loss_meter = AverageMeter()
    scaler = torch.amp.GradScaler('cuda')
    xent = SupConLoss(device)
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    image_features_rgb = []
    image_features_eve = []
    labels = []
    with torch.no_grad():
        for (img, vid, target_cam, target_view) in tqdm(train_loader_stage1, desc="Stage 1 Image Feature Extract"):
            img = img.to(device)
            target = vid.to(device)
            if len(img.size())-1 == 6:
                # method = 'dense'
                b, m, n, t, c, h, w = img.size()
                assert (b == 1)
                img = img.view(b, m, n*t, c, h, w)  # torch.Size([5, 8, 3, 256, 128])
            with torch.amp.autocast('cuda'):
                image_feature_rgb, image_feature_eve = model(img, target, get_image = True)
                for i, img_feat_rgb, img_feat_eve in zip(target, image_feature_rgb, image_feature_eve):
                    labels.append(i)
                    image_features_rgb.append(img_feat_rgb.cpu())
                    image_features_eve.append(img_feat_eve.cpu())
        labels_list = torch.stack(labels, dim=0).cuda() #N
        image_features_list_rgb = torch.stack(image_features_rgb, dim=0).cuda()
        image_features_list_eve = torch.stack(image_features_eve, dim=0).cuda()

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch
    del labels, image_feature_rgb, image_feature_eve

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image).to(device)
        for i in range(i_ter+1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i*batch:(i+1)* batch]
            else:
                b_list = iter_list[i*batch:num_image]
            
            target = labels_list[b_list]
            image_features_rgb = image_features_list_rgb[b_list]
            image_features_eve = image_features_list_eve[b_list]
            with torch.amp.autocast('cuda'):
                text_features_rgb, text_features_eve = model(label = target, get_text = True)
            loss_i2t_rgb = xent(image_features_rgb, text_features_rgb, target, target)
            loss_t2i_rgb = xent(text_features_rgb, image_features_rgb, target, target)

            loss_i2t_eve = xent(image_features_eve, text_features_eve, target, target)
            loss_t2i_eve = xent(text_features_eve, image_features_eve, target, target)

            loss = (loss_i2t_rgb + loss_t2i_rgb + loss_i2t_eve + loss_t2i_eve)/2

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), len(train_loader_stage1),
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))
