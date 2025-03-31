#!/bin/bash
#SBATCH --job-name=fed_xai_exp
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Activate your conda environment if you're using one
# source activate your_env_name
module load python3.8/3.8
source ./shivay_env/bin/activate
# Create logs directory if it doesn't exist
mkdir -p logs

# Print some information about the job
echo "Job started at $(date)"
echo "Running on host: $(hostname)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"

# Run the experiment script with quick test mode
python scripts/run_distributed_experiments.py \
    --output_dir ./results \
    --max_workers 4 \
    --quick_test

# Print completion message
echo "Job finished at $(date)" 