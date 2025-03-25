# scripts/run_experiment.py
import argparse
import json
import logging
import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.preprocess_data import preprocess_data
from server.orchestrator import FederatedOrchestrator
from configuration.xai_config import FederatedXAIConfig, ExplainabilityConfig, PrivacyConfig
from utils.fed_visualization import save_training_plots
from utils.fed_xai_visualization import create_all_visualizations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('run_experiment')

def main():
    parser = argparse.ArgumentParser(description='Run federated XAI experiment')
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config JSON')
    parser.add_argument('--exp-id', type=int, required=True, help='Experiment ID (1-indexed)')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    args = parser.parse_args()
    
    # Load experiment configuration
    with open(args.config, 'r') as f:
        all_experiments = json.load(f)
    
    # Flatten experiments into a list
    experiments = []
    for set_name, exps in all_experiments.items():
        for exp_name, params in exps.items():
            experiments.append((set_name, exp_name, params))
    
    # Select experiment based on exp-id
    if args.exp_id > len(experiments):
        logger.error(f"Invalid experiment ID: {args.exp_id}")
        sys.exit(1)
    
    set_name, exp_name, params = experiments[args.exp_id - 1]
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, f"{set_name}_{exp_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the experiment
    run_single_experiment(params, output_dir)

def run_single_experiment(params, output_dir):
    # Load and preprocess data
    X_train, X_test, y_train, y_test, features = preprocess_data(
        task=params.get('task', 'mortality'),
        sample_fraction=params.get('sample_fraction', 0.4)
    )
    
    # Create XAI configuration
    xai_config = FederatedXAIConfig(
        explainability=ExplainabilityConfig(
            use_lime=params.get('use_lime', True),
            use_shap=params.get('use_shap', True),
            lime_samples=params.get('lime_samples', 1000),
            shap_samples=params.get('shap_samples', 100),
            max_features=params.get('max_features', 15)
        ),
        privacy=PrivacyConfig(
            enable_privacy=params.get('enable_privacy', False),
            epsilon=params.get('epsilon', 1.0),
            delta=params.get('delta', 1e-5)
        ),
        collect_explanations=True,
        explanation_rounds=params.get('explanation_rounds', [5, 10, 15, 20]),
        save_explanations=True,
        explanations_path=os.path.join(output_dir, 'explanations')
    )
    
    # Create model configuration
    model_config = params.get('model_config', {
        'input_dim': X_train.shape[1],
        'hidden_sizes': [128, 64],
        'dropout': 0.3,
        'learning_rate': 0.001
    })
    
    # Create orchestrator and run experiment
    orchestrator = FederatedOrchestrator(
        input_dim=X_train.shape[1],
        features=features,
        model_config=model_config,
        xai_config=xai_config
    )
    
    # Simulate federated data
    num_clients = params.get('num_clients', 8)
    client_data = simulate_federated_data(X_train, y_train, num_clients=num_clients)
    
    # Add clients to orchestrator
    for client_id, (X, y) in client_data.items():
        orchestrator.add_client(client_id, X, y)
    
    # Train federated model
    history = orchestrator.train_federated(
        num_rounds=params.get('num_rounds', 20),
        local_epochs=params.get('local_epochs', 2),
        client_fraction=params.get('client_fraction', 0.8),
        batch_size=params.get('batch_size', 32),
        min_clients=params.get('min_clients', 3),
        validation_data=(X_test, y_test)
    )
    
    # Save training history
    save_training_plots(history, output_dir=output_dir)
    
    # Save explanations visualizations if available
    if orchestrator.explanation_history:
        create_all_visualizations(
            orchestrator.explanation_history,
            orchestrator.client_explanations,
            feature_names=features,
            output_dir=os.path.join(output_dir, 'visualizations')
        )
    
    # Save results for later analysis
    save_results(orchestrator, history, output_dir)

def simulate_federated_data(X, y, num_clients=8):
    import numpy as np
    
    client_data = {}
    client_proportions = np.random.dirichlet(np.ones(num_clients))
    
    start_idx = 0
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    
    for i in range(num_clients):
        end_idx = start_idx + int(len(X) * client_proportions[i])
        if i == num_clients - 1:
            end_idx = len(X)
        
        X_client = X.iloc[start_idx:end_idx]
        y_client = y.iloc[start_idx:end_idx]
        
        client_data[f'client_{i+1}'] = (X_client, y_client)
        start_idx = end_idx
    
    return client_data

def save_results(orchestrator, history, output_dir):
    import pickle
    
    with open(os.path.join(output_dir, 'history.pkl'), 'wb') as f:
        pickle.dump(history, f)
    
    if orchestrator.explanation_history:
        with open(os.path.join(output_dir, 'explanation_history.pkl'), 'wb') as f:
            pickle.dump(orchestrator.explanation_history, f)
    
    if orchestrator.client_explanations:
        with open(os.path.join(output_dir, 'client_explanations.pkl'), 'wb') as f:
            pickle.dump(orchestrator.client_explanations, f)

if __name__ == "__main__":
    main()
