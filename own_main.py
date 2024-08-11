
import timm
from timm.data.mixup import Mixup
from timm.utils import accuracy
from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.datasets import SentinelIndividualImageDataset_OwnData, SentinelIndividualImageDataset, SentinelNormalizeRevert, build_fmow_dataset, build_own_sentineldataset
import util.lr_decay as lrd

import numpy as np
import os
import time
import datetime
import json
import rasterio


import wandb
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

from main_finetune import get_args_parser
import models_mae_group_channels
import models_vit_group_channels
import models_vit

from engine_finetune import (train_one_epoch, train_one_epoch_temporal)
from util.pos_embed import interpolate_pos_embed






def create_img_from_tensor(image): 
    image = image.cpu().numpy()
    npimgtransposed = np.transpose(image, (1, 2, 0))
    stacked_image = np.zeros((npimgtransposed.shape[0], npimgtransposed.shape[1], 3))
    
    # Assign the red channel to the first channel
    np.copyto(stacked_image[:, :, 0], npimgtransposed[:,:,0])
    # Assign the green channel to the second channel
    np.copyto(stacked_image[:, :, 1], npimgtransposed[:,:,1])

    return stacked_image

def save_comparison_fig_from_tensor(final_images,target_images,name):  # final_image shape: [8,2,96,96]


    # first_final_img = final_images[0]      # only print first of batch for now
    # first_target_img = target_images[0]
    
    for idx, img in enumerate(final_images) :
    
        output = create_img_from_tensor(img)
        target = create_img_from_tensor(target_images[idx])

        # Display the image using matplotlib
         #print("image shape: ", image_np.shape)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5)) 
        ax[0].imshow(output) 
        ax[0].set_title('Output')
        ax[0].axis('off')  # Hide axes

        # Plot the 'target' image
        ax[1].imshow(target) 
        ax[1].set_title('Target')
        ax[1].axis('off')  # Hide axes

        plt.savefig(f'imgOut/{name}_img_{idx}.png')
    

def min_mse_per_batch(output, target):
    """
    Computes the lowest Mean Squared Error (MSE) for each sample in the batch.

    Args:
        output (torch.Tensor): Model output of shape [batch_size, num_channels, height, width]
        target (torch.Tensor): Target tensor of shape [batch_size, num_channels, height, width]

    Returns:
        min_mse (torch.Tensor): Tensor of shape [batch_size] containing the lowest MSE for each sample in the batch.
    """
    batch_size = output.size(0)
    
    # Initialize tensor to store MSE values
    mse_values = torch.zeros(batch_size, 2,dtype=torch.float32, device=output.device)
    
    for i in range(batch_size):
        # Get the output and target for the current sample
        output_sample = output[i]
        target_sample = target[i]
        #TODO verify that it compares not the whole batch but a single image
        # Compute MSE for the current sample
        mse = torch.nn.functional.mse_loss(output_sample, target_sample, reduction='none')
        mse_flat = mse.flatten(1,-1)

        # Compute the minimum MSE for the current sample
        avg_mse_channel_one = torch.mean(mse_flat[0])
        avg_mse_channel_two = torch.mean(mse_flat[1])
        # Store the minimum MSE in the tensor
        mse_values[i:0] = avg_mse_channel_one
        mse_values[i:1] = avg_mse_channel_two

    return mse_values


#customized evaluate function to evaluate accuracy of swir prediction not classification
@torch.no_grad()
def evaluate(data_loader, model, device):
    
##### 1.rewrite criterion to mean squared error
    criterion = torch.nn.MSELoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()


    for idx, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
#####2. provide images with cropped swir channels as input        
        images = batch[0]


##### 3. provide real swir channel as target (pbbly rewrite the dataloader to provide the right target)

        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # print("before pass model")
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            save_comparison_fig_from_tensor(output,target,f'comparison_fig_b_{idx}')
            loss = criterion(output, target)
            print("loss in autocast " , loss, autoflush=True)

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




def main(args):


    
    print("Torch Version:        ", torch.__version__)
    print("Torch CUDA Version:   ", torch.version.cuda)
    print("Torch CUDNN Version:  ", torch.backends.cudnn.version())
    
    print ("cuda available: ",torch.cuda.is_available())
    print("cuda devices :" ,torch.cuda.device_count())
    # args: directory_path, masked_bands
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    dataset_val = build_own_sentineldataset(is_train=False, args=args)
    print("OWN DATASET  " ,dataset_val.df.head(10))
    #TODO devide dataset into train and val
    dataset_train = build_own_sentineldataset(is_train=True, args=args)

    
    global_rank = misc.get_rank()
    if False:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None



    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    
    data_loader_train = torch.utils.data.DataLoader(
      dataset_train, sampler=sampler_train,
       batch_size=args.batch_size,
       num_workers=args.num_workers,
       pin_memory=args.pin_mem,
       drop_last=True,
    )


    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    # Define the model
    if args.model_type == 'group_c':
        # Workaround because action append will add to default list
        if len(args.grouped_bands) == 0:
            args.grouped_bands = [[0, 1, 2, 6], [3, 4, 5, 7], [8, 9]]
        print(f"Grouping bands {args.grouped_bands}")
        model = models_vit_group_channels.__dict__[args.model](
            patch_size=args.patch_size, img_size=args.input_size,
            in_chans=dataset_val.in_c, 
            channel_groups=args.grouped_bands,
            num_classes=args.nb_classes, drop_path_rate=args.drop_path, 
            #global_pool=args.global_pool, --> uncommented bc it doesnt seem to be used in init of model (hardcoded to False)
        )
    elif args.model_type == 'mae_group_c':
        model = models_mae_group_channels.__dict__[args.model]()

    else:
        model = models_vit.__dict__[args.model](
            patch_size=args.patch_size, img_size=args.input_size,
                     # in_chans=dataset_train.in_c, seems not to be used and would require dataset_train to be defined
            num_classes=args.nb_classes, drop_path_rate=args.drop_path, global_pool=args.global_pool,
        )


   #not used for now bc only evaluation
   # if args.finetune and not args.eval:
   #trying to use it even in eval so that the model can be loaded because of error in position embedding when using mae
    if args.finetune is not None:
        checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()

        # if 'patch_embed.proj.weight' in checkpoint_model and 'patch_embed.proj.weight' in state_dict:
        #     ckpt_patch_embed_weight = checkpoint_model['patch_embed.proj.weight']
        #     model_patch_embed_weight = state_dict['patch_embed.proj.weight']
        #     if ckpt_patch_embed_weight.shape[1] != model_patch_embed_weight.shape[1]:
        #         print('Using 3 channels of ckpt patch_embed')
        #         model.patch_embed.proj.weight.data[:, :3, :, :] = ckpt_patch_embed_weight.data[:, :3, :, :]

        # Do something smarter?
        for k in ['pos_embed', 'patch_embed.proj.weight', 
                  'patch_embed.proj.bias','patch_embed.0.proj.weight','patch_embed.1.proj.weight','patch_embed.2.proj.weight','head.weight','head.bias']:
            # TODO see if head.weight has to be removed if its not in state_dict or if it works after initializing it in the model
            if (k in checkpoint_model and k not in state_dict) or (k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        #  change assert msg based on patch_embed
        if args.global_pool:
            print(set(msg.missing_keys))
            # assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        else:
            print(set(msg.missing_keys))
            # assert set(msg.missing_keys) == {'head.weight', 'head.bias'}

        # manually initialize fc layer
        # remove trunc_normal for model.head.weight since weights are initalized in the decoder module
        #trunc_normal_(model.head.weight, std=2e-5)
    # elif args.eval :
    #     checkpoint = torch.load(args.resume, map_location='cpu')

    #     print("Load pre-trained checkpoint from: %s" % args.resume)
    #     checkpoint_model = checkpoint['model']
    #     state_dict = model.state_dict()

    #     # Do something smarter?
    #     for k in ['pos_embed', 'patch_embed.proj.weight', 
    #               'patch_embed.proj.bias','patch_embed.0.proj.weight','patch_embed.1.proj.weight','patch_embed.2.proj.weight','head.weight','head.bias']:
    #         if (k in checkpoint_model and k not in state_dict) or (k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape):
    #             print(f"Removing key {k} from pretrained checkpoint")
    #             del checkpoint_model[k]

    #     # interpolate position embedding
    #     interpolate_pos_embed(model, checkpoint_model)

    #     # load pre-trained model
    #     msg = model.load_state_dict(checkpoint_model, strict=False)
    #     print(msg)

    #     #  change assert msg based on patch_embed
    #     if args.global_pool:
    #         print(set(msg.missing_keys))
    #         # assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    #     else:
    #         print(set(msg.missing_keys))
    #         # assert set(msg.missing_keys) == {'head.weight', 'head.bias'}


    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # build optimizer with layer-wise lr decay (lrd)
    
    #print("NO WEIGHT DECAY LIST FOR VIT MODEL : " ,model_without_ddp.no_weight_decay())

    # param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
    #                                         no_weight_decay_list=model_without_ddp.no_weight_decay(),
    #                                         layer_decay=args.layer_decay)
    
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                            no_weight_decay_list=model_without_ddp.no_weight_decay(), #empty for now bc there was no no_weight_decay_list in mae model
                                            layer_decay=args.layer_decay)


    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()
    criterion = torch.nn.MSELoss()
    # if mixup_fn is not None:
    #     # smoothing is handled with mixup label transform
    #     criterion = SoftTargetCrossEntropy()
    # elif args.smoothing > 0.:
    #     criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    # else:
    #     criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Set up wandb
    if global_rank == 0 and args.wandb is not None:
        #wandb.init(project=args.wandb, entity="swir-prediction-thesis")
        wandb.init(project=args.wandb)
        wandb.config.update(args)
        wandb.watch(model)

    
    if args.eval: #set to true for now since we are only evaluating
        test_stats = evaluate(data_loader_val, model, device)
        print("TEST STATS: ", test_stats) 
        print(f"Evaluation on {len(dataset_val)} test images- acc1: {test_stats['acc1']:.2f}%, "
              )
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        if args.model_type == 'temporal':
            train_stats = train_one_epoch_temporal(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, mixup_fn,
                log_writer=log_writer,
                args=args
            )
        else:
            train_stats = train_one_epoch(
                model, criterion, data_loader_train,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, mixup_fn,
                log_writer=log_writer,
                args=args
            )

        if args.output_dir and (epoch % args.save_every == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            print("saved model to ", args.output_dir)

        
        test_stats = evaluate(data_loader_val, model, device)

        # print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        # max_accuracy = max(max_accuracy, test_stats["acc1"])
        # print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            # log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            # log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
            
            if args.wandb is not None:
                try:
                    wandb.log(log_stats)
                except ValueError:
                    print(f"Invalid stats?")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)





