#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH --time=150:00:00
#SBATCH --output=/scratch/asb862/slurm_%j.out
#SBATCH --job-name=Resnet_f1
#SBATCH --mail-type=END
#SBATCH --mail-user=asb862@nyu.edu

python ResNet_train.py
#fadvise(POSIX_FADV_DONTNEED)
