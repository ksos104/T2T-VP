#!/bin/bash -e
#SBATCH -p part1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o slurm_log/%j_out.txt
#SBATCH -e slurm_log/%j_err.txt
#SBATCH --gres=gpu

python main.py
