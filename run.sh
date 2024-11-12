#!/bin/bash
#SBATCH --job-name=cty
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=8 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
module purge
module load anaconda/2024.02 cuda/12.1 cudnn/9.1.0_cu12x
# export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
source activate pyG



cd /ailab/user/cuitaoyong/AI2BMD-ViSNet_beifen
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --conf /ailab/user/cuitaoyong/AI2BMD-ViSNet_beifen/examples/ViSNet-Chignolin.yml --dataset-root /ailab/user/cuitaoyong/AI2BMD-ViSNet_beifen/chignolin_data --log-dir /ailab/user/cuitaoyong/AI2BMD-ViSNet_beifen/path/st828