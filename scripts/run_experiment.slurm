#!/bin/bash
#SBATCH --job-name=fed_xai_exp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=your_partition_name  # Replace with your partition name

# Activate your conda environment if needed
# source activate your_env_name

# Set Python path
export PYTHONPATH=$PWD:$PYTHONPATH

# Run the experiment
python scripts/run_distributed_experiments.py \
    --output_dir ./results \
    --experiments $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} 