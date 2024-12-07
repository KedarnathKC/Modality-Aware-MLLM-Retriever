#!/bin/bash
#SBATCH --job-name=0_Image
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=vram80
#SBATCH --cpus-per-task=2
#SBATCH --mem=50GB
#SBATCH -t 2-00:00:00
#SBATCH --output=slurm_logs/<jobname>_%A-%a.out 

bash script/hardNeg_0_image.sh