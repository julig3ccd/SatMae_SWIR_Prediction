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

mkdir -p ~/data
sshfs geissinger@zandalar:/mnt/data3/CNLNG ~/data

cd src/SatMae_SWIR_Prediction

# venv
source ./venv/test/bin/activate
# For CUDA 11, we need to explicitly request the correct version


# download example script for CNN training


# eval
python3 -m torch.distributed.launch --nproc_per_node=8 validate_only.py \
--wandb satmae_first_eval \
--batch_size 8 --accum_iter 16 --blr 0.0002 \
--num_workers 16 \
--input_size 96 --patch_size 8  \
--model_type group_c  \
--dataset_type sentinel \
--directory_path ~/data \
--masked_bands 11 12 \
--eval \
--output_dir ~/out \
--log_dir ~/log \

fusermount -u ~/data