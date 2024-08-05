
import timm
from timm.data.mixup import Mixup
from timm.utils import accuracy
from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.datasets import SentinelIndividualImageDataset_OwnData, SentinelIndividualImageDataset
from util.datasets import (build_own_sentineldataset, build_fmow_dataset)
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


#TODO use to try to print target to see if it is correct , bc currently it is just green


class SentinelNormalizeRevert:
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.min_value = self.mean - 2 * self.std
        self.max_value = self.mean + 2 * self.std

    def __call__(self, x):
        # Ensure input is in float format
        x = x.astype(np.float32)
        # Revert normalization
        x = x / 255.0 * (self.max_value - self.min_value) + self.min_value
        return x

def save_as_img_with_normalization_revert(image_tensor, output_raster_file):
    # Revert normalization


    revert_normalization = SentinelNormalizeRevert(SentinelIndividualImageDataset.mean, SentinelIndividualImageDataset.std)
    
    # Convert torch tensor to numpy array if necessary
    if isinstance(image_tensor, torch.Tensor):
        image_tensor = image_tensor.detach().cpu().numpy()
    
    # Apply revert normalization to each channel
    for i in range(image_tensor.shape[2]):
        image_tensor[:, :, i] = revert_normalization(image_tensor[:, :, i])
    
    # Define metadata for the new raster file
    metadata = {
        'driver': 'GTiff',
        'count': image_tensor.shape[2],  # Number of channels/bands
        'width': image_tensor.shape[1],  # Width of the raster
        'height': image_tensor.shape[0],  # Height of the raster
        'dtype': 'float32',  # Data type of the raster values
        'crs': 'EPSG:4326',  # Coordinate Reference System (replace as needed)
        'transform': rasterio.transform.from_origin(0, image_tensor.shape[0], 1, 1)  # Affine transform (replace as needed)
    }

    # Create and write to the raster file
    with rasterio.open(output_raster_file, 'w', **metadata) as dst:
        for i in range(image_tensor.shape[2]):
            dst.write(image_tensor[:, :, i], i + 1)  # Write each channel to a separate band

    print(f'Raster file saved to {output_raster_file}')



def create_raster_file_from_tensor(image_tensor, path):

    image_tensor= image_tensor.cpu().numpy()
    print("image tensor type before raster file creation", image_tensor.dtype, image_tensor.shape)
    output_raster_file = f'{path}.tif'
    metadata = {
    'driver': 'GTiff',
    'count': image_tensor.shape[0],  # Number of channels/bands
    'width': image_tensor.shape[1],  # Width of the raster
    'height': image_tensor.shape[2],  # Height of the raster
    'dtype': 'float32',  # Data type of the raster values
    'crs': 'EPSG:4326',  # Coordinate Reference System (replace as needed)
     # Affine transform (replace as needed)
}

    with rasterio.open(output_raster_file, 'w', **metadata) as dst:
        for i in range(image_tensor.shape[0]):
             dst.write(image_tensor[:, :, i], i + 1)  # Write each channel to a separate band

    print(f'Raster file saved to {output_raster_file}') 


def save_as_img_with_normalization(image_tensor, path):
    nptensor= image_tensor.cpu().numpy()
    tensor = (nptensor * 255).astype(np.uint8)

    output_raster_file = f'{path}.tif'

    metadata = {
    'driver': 'GTiff',
    'count': tensor.shape[2],  # Number of channels/bands
    'width': tensor.shape[1],  # Width of the raster
    'height': tensor.shape[0],  # Height of the raster
    'dtype': 'uint8',  # Data type of the raster values
    'crs': 'EPSG:4326',  # Coordinate Reference System (replace as needed)
    'transform': rasterio.transform.from_origin(0, tensor.shape[0], 1, 1)  # Affine transform (replace as needed)
}

# Create and write to the raster file
    with rasterio.open(output_raster_file, 'w', **metadata) as dst:
        for i in range(tensor.shape[2]):
            dst.write(tensor[:, :, i], i + 1)  # Write each channel to a separate band

    print(f'Raster file saved to {output_raster_file}')


def create_img_from_tensor(image,img_size): 

    image = (image - image.min()) / (image.max() - image.min())
    to_pil_image = transforms.ToPILImage()

    # Convert each channel to a PIL image and show
    for i in range(image.size(0)):  # Loop through channels
        channel_image = to_pil_image(image[i])
        channel_image.save(f'{""}_channel_{i}.png')
         
    image_np = image.permute(1, 2, 0).cpu().numpy() # [2,96,96] -> [96,96,2] permute channels only for matplotlib
      #add black channel so that it can be displayed(imshow requests 3 channels)
    # black = np.zeros((img_size,img_size), dtype=np.uint8)
    # image_np = np.dstack((image_np, black))
    # image_np = image_np.astype(np.float32)

    return image_np

def save_comparison_fig_from_tensor(final_images,target_images,img_size):  # final_image shape: [8,2,96,96]



    first_final_img = final_images[0]      # only print first of batch for now
    first_target_img = target_images[0]    # only print first of batch for now

    # Normalize to [0, 1] for visualization if necessary
    output = create_img_from_tensor(first_final_img,img_size)
    target = create_img_from_tensor(first_target_img,img_size)


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

    plt.savefig(f'imgOut/comparison.png')
    

#customized evaluate function to evaluate accuracy of swir prediction not classification
@torch.no_grad()
def evaluate(data_loader, model, device):
    
##### 1.rewrite criterion to mean squared error
    criterion = torch.nn.MSELoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
#####2. provide images with cropped swir channels as input        
        images = batch[0]

##### 3. provide real swir channel as target (pbbly rewrite the dataloader to provide the right target)

        target = batch[-1]
        print('images [0] shape -->' ,images[0].shape ,'images [-1] shape -->',images[-1].shape, 'target shape [-1] --> ',target[-1].shape, 'target shape [0] --> ',target[0].shape)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # print("before pass model")
        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            save_as_img_with_normalization_revert(output[0], 'imgOut/output_with_normalization_rev')
            #save_as_img_with_normalization(output[0], 'imgOut/output_normalized_to_rgb')
            save_as_img_with_normalization_revert(target[0], 'imgOut/target_with_normalization_rev')
            #save_as_img_with_normalization(target[0], 'imgOut/target_normalized_to_rgb')
            #save_comparison_fig_from_tensor(output,target,img_size=96)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        # print(acc1, acc5, flush=True)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}







#create sentinelindivudualdataset from own data

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

    # firstimg = dataset_val.__getitem__(0)
    # print("FIRST IMG", firstimg)
    # inputimg = firstimg[0]
    # print("input img shape in OWN DATA", inputimg.shape)
    # targetimg = firstimg[1]
    
    # create_raster_file_from_tensor(inputimg, 'imgOut/input_after_dataset_creation')
    # create_raster_file_from_tensor(targetimg, 'imgOut/target_after_dataset_creation')

    

    #not used anyways for now, but needs to be changed for actual training
    #dataset_train = build_fmow_dataset(is_train=True, args=args)

    #taken out of if so it can be used for evaluation case
    #directly using sequentialsampler bc eval only (set if True for training)
    
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
        #sampler_train = torch.utils.data.RandomSampler(dataset_train)
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
    
    ##not used for now ppbly needs to be changed for actual training
    #data_loader_train = torch.utils.data.DataLoader(
    #   dataset_train, sampler=sampler_train,
    #    batch_size=args.batch_size,
     #   num_workers=args.num_workers,
      #  pin_memory=args.pin_mem,
       # drop_last=True,
    #)


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
        checkpoint = torch.load(args.resume, map_location='cpu')

        print("Load pre-trained checkpoint from: %s" % args.resume)
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

    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # Set up wandb
    if global_rank == 0 and args.wandb is not None:
        #wandb.init(project=args.wandb, entity="swir-prediction-thesis")
        wandb.init(project=args.wandb)
        wandb.config.update(args)
        wandb.watch(model)

    
    # if args.eval: #set to true for now since we are only evaluating
    if True:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Evaluation on {len(dataset_val)} test images- acc1: {test_stats['acc1']:.2f}%, "
              f"acc5: {test_stats['acc5']:.2f}%")
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

        
        test_stats = evaluate(data_loader_val, model, device)

        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
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





