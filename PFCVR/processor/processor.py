import logging
import time
import torch
from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter
from prettytable import PrettyTable


def do_train(start_epoch, args, model, train_loader, evaluator, optimizer,
             scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("IRRA.train")
    logger.info('start training')

    meters = {
        "loss": AverageMeter(),
        "sdm_loss": AverageMeter(),
        "itc_loss": AverageMeter(),
        "part_loss": AverageMeter(),
        "id_loss": AverageMeter(),
        "mlm_loss": AverageMeter(),
        "mim_loss": AverageMeter(),
        "attr_loss": AverageMeter(),
        "img_acc": AverageMeter(),
        "txt_acc": AverageMeter(),
        "mlm_acc": AverageMeter(),
        "part_acc": AverageMeter()
    }

    tb_writer = SummaryWriter(log_dir=args.output_dir)

    best_top1 = 0.0

    # train
    previous_losses = None
    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()
        model.train()

        for n_iter, batch in enumerate(train_loader):
            batch = {
                k: (v.to(device) if hasattr(v, "to") and k != "boxes" and k != "mask" else v)
                for k, v in batch.items()
            }
            # breakpoint()
            ret = model(batch)
            total_loss = sum([v for k, v in ret.items() if "loss" in k and "ori" not in k])

            batch_size = batch['images'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)
            meters['sdm_loss'].update(ret.get('sdm_loss_ori', 0), batch_size)
            meters['itc_loss'].update(ret.get('itc_loss_ori', 0), batch_size)
            meters['part_loss'].update(ret.get('part_loss_ori', 0), batch_size)
            meters['id_loss'].update(ret.get('id_loss_ori', 0), batch_size)
            meters['mlm_loss'].update(ret.get('mlm_loss_ori', 0), batch_size)
            meters['mim_loss'].update(ret.get('mim_loss_ori', 0), batch_size)
            meters['attr_loss'].update(ret.get('attr_loss_ori', 0), batch_size)
            

            meters['img_acc'].update(ret.get('img_acc', 0), batch_size)
            meters['txt_acc'].update(ret.get('txt_acc', 0), batch_size)
            meters['part_acc'].update(ret.get('part_acc', 0), batch_size)
            meters['mlm_acc'].update(ret.get('mlm_acc', 0), batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            # breakpoint()

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                # log loss and acc info
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.4f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)
        
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)


        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch,
                        train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    top1 = evaluator.eval(model.module.eval())
                else:
                    top1 = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_top1 < top1:
                    best_top1 = top1
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)


        # if previous_losses != None:
        #     print(previous_losses)
        #     print(model.losses)
        #     print("update loss weights")
        #     print("previous weights:", model.loss_weights)
        #     # 动态调整权重
        #     for loss_name, current_loss in model.losses.items():
        #         if current_loss > previous_losses[loss_name]:
        #             model.loss_weights[loss_name.replace("_loss","")] *= (1-model.update_ratio)  # 减小权重
        #         else:
        #             model.loss_weights[loss_name.replace("_loss","")] *= (1+model.update_ratio)  # 增加权重
            
        #     print("currrent weights:", model.loss_weights)

        # previous_losses = model.losses
        # model.losses = None
            
            
    if get_rank() == 0:
        logger.info(f"best R1: {best_top1} at epoch {arguments['epoch']}")


def do_inference(model, test_img_loader, test_txt_loader):

    logger = logging.getLogger("IRRA.test")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_img_loader, test_txt_loader)
    top1 = evaluator.eval(model.eval(),visual=True)
