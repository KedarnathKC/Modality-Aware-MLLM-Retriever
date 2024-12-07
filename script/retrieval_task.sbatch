#!/bin/bash
#SBATCH --job-name=Run-Eval
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH -t 0-12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=128G
#SBATCH --output=slurm_logs/<jobname>_%A-%a.out 

bash script/retrieval_task.sh