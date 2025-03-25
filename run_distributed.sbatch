#!/bin/bash
#SBATCH --job-name=federated_xai
#SBATCH --output=federated_xai_%A_%a.out
#SBATCH --error=federated_xai_%A_%a.err
#SBATCH --array=1-10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00


# Load necessary modules
module load python3.8/3.8


# Activate virtual environment (if using)
source ./shivay_env/bin/activate


# Run the experiment
python -u ./scripts/run_multiple.py --config experiments.json --exp-id $SLURM_ARRAY_TASK_ID --output-dir ./results
