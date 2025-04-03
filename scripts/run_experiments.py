# run_experiments.py

import sys
import os
import json
import logging
import argparse
from datetime import datetime
import copy

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from scripts.preprocess_data import preprocess_data
from server.orchestrator import FederatedOrchestrator
from utils.fed_visualization import save_training_plots, save_client_contributions
from utils.fed_xai_visualization import create_all_visualizations, save_explanation_dashboard
from configuration.xai_config import (
    FederatedXAIConfig, 
    LIME_ONLY_CONFIG, 
    SHAP_ONLY_CONFIG, 
    STRICT_PRIVACY_CONFIG
)
from integration.xai_integration import initialize_xai_components

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('experiment_runner')

def simulate_federated_data(X, y, num_clients=3):
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
        
        logger.info(f"Client {i+1} data shape: {X_client.shape}")
        logger.info(f"Client {i+1} positive samples: {sum(y_client)}")
    
    return client_data

def run_experiment(exp_name, exp_config, base_dir):
    """
    Run a single experiment with the specified configuration
    
    Args:
        exp_name (str): Name of the experiment
        exp_config (dict): Experiment configuration
        base_dir (str): Base directory for all experiments
    
    Returns:
        tuple: (orchestrator, history) from the experiment
    """
    # Create experiment directory
    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save configuration to the experiment directory
    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(exp_config, f, indent=2)
    
    logger.info(f"Running experiment: {exp_name}")
    logger.info(f"Results will be saved to: {exp_dir}")
    
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        X_train, X_test, y_train, y_test, features = preprocess_data(
            task=exp_config.get('task', 'mortality'), 
            sample_fraction=exp_config.get('sample_fraction', 0.4)
        )
        logger.info(f"Full dataset shape: {X_train.shape}")
        logger.info(f"Number of features: {len(features)}")
        
        # Define model architecture
        hidden_sizes = exp_config.get('hidden_sizes', [256, 128, 64])
        model_config = {
            'input_dim': X_train.shape[1],
            'hidden_sizes': hidden_sizes,
            'dropout': exp_config.get('dropout', 0.3),
            'learning_rate': exp_config.get('learning_rate', 0.001),
            'l2_reg': exp_config.get('l2_reg', 0.01)
        }
        
        # Create XAI configuration
        if 'xai_config_type' in exp_config:
            # Use one of the predefined configurations
            if exp_config['xai_config_type'] == 'lime_only':
                xai_config = LIME_ONLY_CONFIG
            elif exp_config['xai_config_type'] == 'shap_only':
                xai_config = SHAP_ONLY_CONFIG
            elif exp_config['xai_config_type'] == 'strict_privacy':
                xai_config = STRICT_PRIVACY_CONFIG
            else:
                # Create a custom XAI configuration
                explainability = exp_config.get('explainability', {})
                privacy = exp_config.get('privacy', {})
                aggregation = exp_config.get('aggregation', {})
                
                xai_config = FederatedXAIConfig(
                    explainability=explainability,
                    privacy=privacy,
                    aggregation=aggregation,
                    collect_explanations=exp_config.get('collect_explanations', True),
                    explanation_rounds=exp_config.get('explanation_rounds', [5, 10, 15, 20, 25]),
                    save_explanations=exp_config.get('save_explanations', True)
                )
        else:
            # Create a custom XAI configuration
            explainability = exp_config.get('explainability', {})
            privacy = exp_config.get('privacy', {})
            aggregation = exp_config.get('aggregation', {})
            
            xai_config = FederatedXAIConfig(
                explainability=explainability,
                privacy=privacy,
                aggregation=aggregation,
                collect_explanations=exp_config.get('collect_explanations', True),
                explanation_rounds=exp_config.get('explanation_rounds', [5, 10, 15, 20, 25]),
                save_explanations=exp_config.get('save_explanations', True)
            )
        
        # Set up paths for saving explanations
        xai_config.explanations_path = os.path.join(exp_dir, 'explanations')
        
        # Initialize orchestrator
        orchestrator = FederatedOrchestrator(
            input_dim=X_train.shape[1],
            features=features,
            model_config=model_config,
            xai_config=xai_config
        )
        
        # Add this line to initialize XAI components
        initialize_xai_components(orchestrator, xai_config)

        # Simulate federated data
        num_clients = exp_config.get('num_clients', 10)
        client_data = simulate_federated_data(X_train, y_train, num_clients=num_clients)
        
        # Add clients to orchestrator
        for client_id, (X, y) in client_data.items():
            orchestrator.add_client(client_id, X, y)
        
        # After adding clients, initialize explainers with some training data
        for client_id, client_info in orchestrator.clients.items():
            client = client_info['client']
            if hasattr(client, 'model') and hasattr(client.model, 'initialize_explainers'):
                # Use a small sample of client's training data to initialize explainers
                X_sample = client_info.get('X_train')[:100]  # Use first 100 samples
                client.model.initialize_explainers(X_sample)
                
        # Train federated model
        logger.info(f"Starting federated training with {num_clients} clients and {exp_config.get('num_rounds', 25)} rounds...")
        history = orchestrator.train_federated(
            num_rounds=exp_config.get('num_rounds', 25),
            local_epochs=exp_config.get('local_epochs', 2),
            client_fraction=exp_config.get('client_fraction', 0.8),
            batch_size=exp_config.get('batch_size', 32),
            min_clients=exp_config.get('min_clients', 3),
            validation_data=(X_test, y_test)
        )
        
        # Save visualizations to experiment directory
        vis_dir = os.path.join(exp_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        #save_training_plots(history, output_dir=vis_dir)
        #save_client_contributions(history, output_dir=vis_dir)
        
        # Generate explanation visualizations if explanations were collected
        if orchestrator.explanation_history:
            logger.info("Generating explanation visualizations...")
            create_all_visualizations(
                orchestrator.explanation_history,
                orchestrator.client_explanations,
                feature_names=features,
                output_dir=vis_dir
            )
            
            # Create explanation dashboard
            dashboard_path = save_explanation_dashboard(
                orchestrator.explanation_history,
                output_dir=vis_dir
            )
        
        # Save final model weights
        model_path = os.path.join(exp_dir, 'final_model.h5')
        orchestrator.server.model.save_weights(model_path)
        logger.info(f"Model weights saved to {model_path}")
        
        # Save final performance metrics
        summary = orchestrator.get_training_summary()
        summary_path = os.path.join(exp_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Experiment {exp_name} completed successfully")
        return orchestrator, history
    
    except Exception as e:
        logger.error(f"Experiment {exp_name} failed: {str(e)}")
        logger.error("Detailed error: ", exc_info=True)
        
        # Save error information
        error_path = os.path.join(exp_dir, 'error.log')
        with open(error_path, 'w') as f:
            f.write(f"Error: {str(e)}\n")
        
        return None, None

def run_experiments(config_file):
    """
    Run multiple experiments from a configuration file
    
    Args:
        config_file (str): Path to the configuration file
    """
    # Load experiment configurations
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    base_config = config.get('base_config', {})
    experiments = config.get('experiments', [])
    
    # Create base directory for experiments with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = os.path.join(
        config.get('output_dir', 'experiments'),
        f"{config.get('experiment_group', 'federated_learning')}_{timestamp}"
    )
    os.makedirs(base_dir, exist_ok=True)
    
    # Save the complete configuration
    with open(os.path.join(base_dir, 'all_experiments.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run each experiment
    results = {}
    for exp in experiments:
        exp_name = exp.get('name')
        if not exp_name:
            # Generate name if not provided
            exp_name = f"experiment_{len(results) + 1}"
        
        # Create experiment configuration by merging base config with experiment-specific config
        exp_config = copy.deepcopy(base_config)
        exp_config.update(exp.get('config', {}))
        
        # Run the experiment
        orchestrator, history = run_experiment(exp_name, exp_config, base_dir)
        
        # Store results
        results[exp_name] = {
            'success': orchestrator is not None,
            'config': exp_config
        }
        
        # Add final metrics if available
        if orchestrator is not None:
            results[exp_name]['final_metrics'] = orchestrator.get_training_summary().get('final_metrics', {})
    
    # Save overall results
    results_path = os.path.join(base_dir, 'results_summary.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"All experiments completed. Summary saved to {results_path}")
    logger.info(f"Experiment results are in: {base_dir}")
    
    return base_dir, results

if __name__ == "__main__":
    import numpy as np
    
    parser = argparse.ArgumentParser(description="Run federated learning experiments")
    parser.add_argument('config', help='Path to experiment configuration file')
    args = parser.parse_args()
    
    run_experiments(args.config)
