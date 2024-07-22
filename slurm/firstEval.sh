#!/usr/bin/env bash
#
#SBATCH --job-name satmae_eval
#SBATCH --output=res.txt
#SBATCH --ntasks=1
#SBATCH --time=10:00

#SBATCH --gres=gpu:1

# debug info
hostname
which python3
nvidia-smi

env

echo $CUDA_VISIBLE_DEVICES

source ./venv/satmae_env/bin/activate

cd ./src/SatMae_SWIR_Prediction

export CUDA_VISIBLE_DEVICES=1
# venv
# For CUDA 11, we need to explicitly request the correct version

echo $CUDA_VISIBLE_DEVICES
# download example script for CNN training


# eval
python3 -m torch.distributed.launch --nproc_per_node=1 validate_only.py \
--wandb satmae_first_eval \
--batch_size 8 --accum_iter 16 --blr 0.0002 \
--num_workers 1 \
--input_size 96 --patch_size 8  \
--model_type group_c  \
--dataset_type sentinel \
--directory_path /../../nfs/data3/CNLNG/  \
--masked_bands 11 12 \
--eval \
--output_dir ~/out \
--log_dir ~/log \