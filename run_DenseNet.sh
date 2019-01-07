#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --mem=80GB
#SBATCH --time=150:00:00
#SBATCH --output=/scratch/asb862/slurm_%j.out
#SBATCH --job-name=densenet
#SBATCH --mail-type=END
#SBATCH --mail-user=asb862@nyu.edu
python DenseNet_train.py 
#fadvise(POSIX_FADV_DONTNEED)
