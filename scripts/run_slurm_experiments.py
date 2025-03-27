"""
Federated XAI Experiment Runner for Slurm Environments
"""
import os
import sys
import json
import logging
import numpy as np
from datetime import datetime

# Configure paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from scripts.preprocess_data import preprocess_data
from server.orchestrator import FederatedOrchestrator
from configuration.xai_config import FederatedXAIConfig
from utils.fed_visualization import save_training_plots
from utils.fed_xai_visualization import create_all_visualizations
from integration.xai_integration import initialize_xai_components

# Slurm configuration
SLURM_ID = os.getenv('SLURM_ARRAY_TASK_ID', '1')
EXPERIMENT_ID = int(SLURM_ID)
CONFIG_FILE = os.path.join(BASE_DIR, 'configs', 'experiments.json')

# Setup logging
def configure_logging(exp_name: str):
    log_dir = os.path.join(BASE_DIR, 'logs', f'exp_{EXPERIMENT_ID:04d}')
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'{exp_name}.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('slurm_runner')

def load_experiment_config():
    """Load experiment configuration from JSON file"""
    with open(CONFIG_FILE) as f:
        configs = json.load(f)
    
    experiments = []
    for group in configs['experiment_groups']:
        for exp in group['experiments']:
            experiments.append({
                'group': group['group_name'],
                **exp
            })
    
    if EXPERIMENT_ID > len(experiments):
        raise ValueError(f"Invalid experiment ID: {EXPERIMENT_ID}")
    
    return experiments[EXPERIMENT_ID - 1]

def run_experiment(params):
    """Main experiment workflow"""
    logger = configure_logging(params['name'])
    
    # Data preprocessing
    X_train, X_test, y_train, y_test, features = preprocess_data(
        task=params['task'],
        sample_fraction=params['data']['sample_fraction']
    )

    # XAI configuration
    xai_config = FederatedXAIConfig.from_dict(params['xai_config'])

    # Model configuration
    model_config = {
        'input_dim': X_train.shape[1],
        **params['model']
    }

    # Initialize orchestrator
    orchestrator = FederatedOrchestrator(
        input_dim=X_train.shape[1],
        features=features,
        model_config=model_config,
        xai_config=xai_config
    )
    initialize_xai_components(orchestrator, xai_config)

    # Simulate federated data
    client_data = simulate_federated_data(
        X_train, y_train,
        num_clients=params['federated']['num_clients'],
        distribution=params['data']['distribution']
    )

    # Add clients
    for cid, (X, y) in client_data.items():
        orchestrator.add_client(cid, X, y)

    # Training
    history = orchestrator.train_federated(
        num_rounds=params['federated']['rounds'],
        local_epochs=params['federated']['local_epochs'],
        client_fraction=params['federated']['client_fraction'],
        batch_size=params['federated']['batch_size'],
        validation_data=(X_test, y_test)
    )

    # Save results
    save_results(orchestrator, history, params)

def simulate_federated_data(X, y, num_clients=8, distribution='dirichlet'):
    """Simulate federated data distribution"""
    client_data = {}
    
    if distribution == 'dirichlet':
        proportions = np.random.dirichlet(np.ones(num_clients))
    elif distribution == 'uniform':
        proportions = np.ones(num_clients)/num_clients
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    start_idx = 0
    for i in range(num_clients):
        end_idx = start_idx + int(len(X) * proportions[i])
        if i == num_clients - 1:
            end_idx = len(X)
        
        client_data[f'client_{i+1}'] = (
            X.iloc[start_idx:end_idx],
            y.iloc[start_idx:end_idx]
        )
        start_idx = end_idx
    
    return client_data

def save_results(orchestrator, history, params):
    """Save all experiment results"""
    output_dir = os.path.join(
        BASE_DIR, 'results',
        f"exp_{EXPERIMENT_ID:04d}_{params['name']}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump({
            'final_accuracy': history['global_metrics'][-1]['accuracy'],
            'final_auc': history['global_metrics'][-1]['auc'],
            'config': params
        }, f)
    
    # Save visualizations
    save_training_plots(history, output_dir=output_dir)
    
    if orchestrator.explanation_history:
        create_all_visualizations(
            orchestrator.explanation_history,
            orchestrator.client_explanations,
            feature_names=orchestrator.server.features,
            output_dir=os.path.join(output_dir, 'explanations')
        )

if __name__ == "__main__":
    experiment_config = load_experiment_config()
    run_experiment(experiment_config)
