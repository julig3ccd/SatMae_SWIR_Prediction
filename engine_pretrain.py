# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch
import wandb

import util.misc as misc
import util.lr_sched as lr_sched
import os
from util.print import save_comparison_fig_from_tensor

@torch.no_grad()
def evaluate(data_loader, model, device, print_comparison=False, args=None):
    
##### 1.rewrite criterion to mean squared error
    criterion = torch.nn.MSELoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    i_size = args.input_size
    p_size = args.patch_size
    num_patches_per_axis = (i_size // p_size)
    # switch to evaluation mode
    #TODO see if mae even has .eval()
    #model.eval()

    for idx, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
#####2. provide images with cropped swir channels as input        
        images = batch[0]
        if idx==0:
            inputswir = images[0,[8,9],:,:]   # first image of batch, only swir channels, all spatial dimensions
            print("THESE ARE THE SWIR CHANNELS IN THE INPUT: ", inputswir)


##### 3. provide real swir channel as target (pbbly rewrite the dataloader to provide the right target)

        swir_targets = batch[-1]
        images = images.to(device, non_blocking=True)
        swir_targets = swir_targets.to(device, non_blocking=True)
        #print("INPUT SHAPE IN EVALUATE: ", images.shape)

        # print("before pass model")
        # compute output
        with torch.cuda.amp.autocast():
            _, pred, mask = model(images, swir_only=args.swir_only, mask_ratio=args.mask_ratio)
            #print("PRED SHAPE: ", pred.shape) #--> ([16, 10, 144, 64]) [Batch, Channels, SeqLen, p^2]
            #reshape predition to make it comparable to target swir
            b_size = pred.shape[0]
            #tokens -> patches
            swirpred = pred.view(b_size,pred.shape[1],num_patches_per_axis,num_patches_per_axis,p_size,p_size)
            swirpred = swirpred.permute(0, 1, 2, 4, 3, 5).contiguous()
            #patches -> image
            swirpred = swirpred.view(b_size,pred.shape[1],i_size,i_size)
            #not needed anymore bc of changed model output ->
            #full image -> only swir channels
            swirpred = swirpred[:,[8,9],:,:]
            loss = criterion(swirpred, swir_targets)
            #print("loss in autocast " , loss)

        if print_comparison:
              if idx % 100 == 0:
                save_comparison_fig_from_tensor(swirpred,f'eval_comparison_fig_b_{idx}',target_images=swir_targets,num_channels=2,mask=mask,input=images)
                print('saved comparison figures for batch ',idx)
        # acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # print(acc1, acc5, flush=True)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        # print(min_mse_per_batch(output, target))
        # metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    #metric_logger.synchronize_between_processes()
    # print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
    #       .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        #print("SAMPLES / MODEL INPUT SHAPE: ", samples.shape) --> [16, 10, 96, 96]
        #print("FIRST SAMPLE SHAPE : ", samples[0].shape) --> ([10, 96, 96])
        with torch.cuda.amp.autocast():
            loss, pred, mask = model(samples, swir_only=args.swir_only, targets=targets, mask_ratio=args.mask_ratio)
            #print("PRED SHAPE: ", pred.shape) --> ([16, 10, 144, 64]) [Batch, Channels, SeqLen, p^2]
            #print("MASK SHAPE: ", mask.shape) --> ([16, 3, 144])


        # if args.print_comparison:
        #         print("DATA_ITER_STEP: " ,data_iter_step)
        #         predImages = pred.view(16,10,12,12,8,8)
        #         predImages = predImages.permute(0, 1, 2, 4, 3, 5).contiguous()
        #         predImages = predImages.view(16,10,96,96)
        #         predImages = predImages.detach()
        #         save_comparison_fig_from_tensor(predImages,f'comparison_fig_b_{data_iter_step}',num_channels=10)
        #         print('saved comparison figures for batch ',data_iter_step)

        
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise ValueError(f"Loss is {loss_value}, stopping training")
            # sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

            # Wandb logging
            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log({'train_loss_step': loss_value_reduce,
                               'train_lr_step': lr, 'epoch_1000x': epoch_1000x})
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch_temporal(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, timestamps, _) in \
            enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        timestamps = timestamps.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss, _, _ = model(samples, timestamps, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

            # Use wandb
            if args.local_rank == 0 and args.wandb is not None:
                try:
                    wandb.log({'train_loss_step': loss_value_reduce,
                               'train_lr_step': lr, 'epoch_1000x': epoch_1000x})
                except ValueError:
                    pass

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}