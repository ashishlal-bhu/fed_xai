#!/bin/bash
#SBATCH --job-name=xai_train          # Job name
#SBATCH --output=output.log        # Standard output and error log
#SBATCH --error=error.log          # Error log
#SBATCH --time=04:00:00            # Time limit hh:mm:ss

#SBATCH --nodes=2                 # Number of nodes
#SBATCH --ntasks=8                # Number of tasks (processes)
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --mem=32G                   # Memory per node


# Load necessary modules
module load python3.8/3.8

# Activate virtual environment (if needed)
source ./shivay_env/bin/activate

# Run Python script
python -u ./scripts/train_xai_model.py
