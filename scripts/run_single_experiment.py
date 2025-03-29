import os
import sys
import json
import logging
import argparse
from typing import Dict, Any

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the run_single_experiment function from the main script
from scripts.run_distributed_experiments import run_single_experiment

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('single_experiment')

def main():
    parser = argparse.ArgumentParser(description='Run a single federated XAI experiment')
    parser.add_argument('--experiment_name', type=str, required=True,
                      help='Name of the experiment')
    parser.add_argument('--config', type=str, required=True,
                      help='JSON string containing experiment configuration')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save experiment results')
    args = parser.parse_args()
    
    try:
        # Parse configuration
        config = json.loads(args.config)
        
        # Run experiment
        results = run_single_experiment(
            experiment_name=args.experiment_name,
            config=config,
            output_dir=args.output_dir
        )
        
        logger.info(f"Experiment {args.experiment_name} completed successfully")
        
    except Exception as e:
        logger.error(f"Error running experiment {args.experiment_name}: {str(e)}")
        logger.error("Detailed error: ", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 