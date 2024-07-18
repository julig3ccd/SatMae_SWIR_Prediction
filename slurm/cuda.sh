#!/usr/bin/env bash
#
#SBATCH --job-name pytorch_test
#SBATCH --output=res.txt
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --gres=gpu:1


hostname
which python3
nvidia-smi

env


source ./venv/satmae_env/bin/activate

cd ../

python3 -c cudaTest.py 