import sys
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from scripts.preprocess_data import preprocess_data
from server.orchestrator import FederatedOrchestrator
from configuration.xai_config import FederatedXAIConfig, ExplainabilityConfig, PrivacyConfig
from utils.fed_visualization import save_training_plots
from utils.fed_xai_visualization import create_all_visualizations

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('experiment_name')

def run_experiment(experiment_params):
    """Run experiment with given parameters"""
    # Create results directory for this experiment
    experiment_name = experiment_params.get('name', 'experiment')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(os.getcwd(), 'results', f"{experiment_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save experiment parameters
    with open(os.path.join(results_dir, 'parameters.json'), 'w') as f:
        import json
        json.dump(experiment_params, f, indent=2, default=str)
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, features = preprocess_data(
        task=experiment_params.get('task', 'mortality'),
        sample_fraction=experiment_params.get('sample_fraction', 0.4)
    )
    
    # Create model configuration
    model_config = experiment_params.get('model_config', {
        'input_dim': X_train.shape[1],
        'hidden_sizes': [128, 64],
        'dropout': 0.3,
        'learning_rate': 0.001
    })
    
    # Create XAI configuration
    xai_config = FederatedXAIConfig(
        explainability=ExplainabilityConfig(
            use_lime=experiment_params.get('use_lime', True),
            use_shap=experiment_params.get('use_shap', True),
            lime_samples=experiment_params.get('lime_samples', 1000),
            shap_samples=experiment_params.get('shap_samples', 100)
        ),
        privacy=PrivacyConfig(
            enable_privacy=experiment_params.get('enable_privacy', True),
            epsilon=experiment_params.get('epsilon', 1.0)
        ),
        collect_explanations=True,
        explanation_rounds=experiment_params.get('explanation_rounds', [5, 10, 15, 20]),
        explanations_path=os.path.join(results_dir, 'explanations')
    )
    
    # Create orchestrator
    orchestrator = FederatedOrchestrator(
        input_dim=X_train.shape[1],
        features=features,
        model_config=model_config,
        xai_config=xai_config
    )
    
    # Simulate federated data
    num_clients = experiment_params.get('num_clients', 8)
    client_data = simulate_federated_data(X_train, y_train, num_clients=num_clients)
    
    # Add clients to orchestrator
    for client_id, (X, y) in client_data.items():
        orchestrator.add_client(client_id, X, y)
    
    # Train federated model
    logger.info(f"Starting experiment: {experiment_name}")
    history = orchestrator.train_federated(
        num_rounds=experiment_params.get('num_rounds', 20),
        local_epochs=experiment_params.get('local_epochs', 2),
        client_fraction=experiment_params.get('client_fraction', 0.8),
        batch_size=experiment_params.get('batch_size', 32),
        min_clients=experiment_params.get('min_clients', 3),
        validation_data=(X_test, y_test)
    )
    
    # Create visualizations in experiment directory
    save_training_plots(history, output_dir=results_dir)
    
    if orchestrator.explanation_history:
        create_all_visualizations(
            orchestrator.explanation_history,
            orchestrator.client_explanations,
            feature_names=features,
            output_dir=os.path.join(results_dir, 'visualizations')
        )
    
    # Return results for analysis
    return {
        'orchestrator': orchestrator,
        'history': history,
        'results_dir': results_dir
    }

def simulate_federated_data(X, y, num_clients=8):
    """Simulate federated data by splitting into multiple clients"""
    client_data = {}
    
    # Create uneven splits to simulate realistic scenario
    client_proportions = np.random.dirichlet(np.ones(num_clients))
    
    start_idx = 0
    X = X.reset_index(drop=True)  # Reset index for proper splitting
    y = y.reset_index(drop=True)
    
    for i in range(num_clients):
        # Calculate end index for this client
        end_idx = start_idx + int(len(X) * client_proportions[i])
        if i == num_clients - 1:
            end_idx = len(X)
            
        # Get client data
        X_client = X.iloc[start_idx:end_idx]
        y_client = y.iloc[start_idx:end_idx]
        
        client_data[f'client_{i+1}'] = (X_client, y_client)
        start_idx = end_idx
    
    return client_data

def main():
    # Define experiments to run
    experiments = [
        {
            'name': 'baseline_experiment',
            'num_rounds': 20,
            'num_clients': 8,
            'use_lime': True,
            'use_shap': True,
            'enable_privacy': False
        },
        {
            'name': 'lime_only_experiment',
            'num_rounds': 20,
            'num_clients': 8,
            'use_lime': True,
            'use_shap': False,
            'enable_privacy': False
        },
        {
            'name': 'privacy_experiment',
            'num_rounds': 20,
            'num_clients': 8,
            'use_lime': True,
            'use_shap': True,
            'enable_privacy': True,
            'epsilon': 0.5
        }
    ]
    
    # Run experiments
    results = {}
    for experiment_params in experiments:
        try:
            logger.info(f"Starting experiment: {experiment_params['name']}")
            experiment_result = run_experiment(experiment_params)
            results[experiment_params['name']] = experiment_result
            logger.info(f"Completed experiment: {experiment_params['name']}")
        except Exception as e:
            logger.error(f"Error in experiment {experiment_params['name']}: {str(e)}")
            logger.error("Detailed error: ", exc_info=True)
    
    return results

if __name__ == "__main__":
    main()