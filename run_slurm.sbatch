#!/bin/bash
#SBATCH --job-name=fedxai
#SBATCH --array=1-3
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --output=logs/slurm-%A_%a.out

module load python3.8/3.8
source ./shivay_env/bin/activate

python -u scripts/run_slurm_experiments.py