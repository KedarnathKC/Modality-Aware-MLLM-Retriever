#!/bin/bash
#SBATCH --job-name=MLLM
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=2
#SBATCH --mem=30GB
#SBATCH -t 0-12:00:00
#SBATCH --output=slurm_logs/<jobname>_%A-%a.out 

bash script/model_train_HFModel.sh