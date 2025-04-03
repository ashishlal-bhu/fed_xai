import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import hashlib

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from scripts.preprocess_data import preprocess_data
from server.orchestrator import FederatedOrchestrator
from configuration.xai_config import (
    FederatedXAIConfig,
    ExplainabilityConfig,
    PrivacyConfig,
    AggregationConfig
)
from utils.fed_visualization import save_training_plots, save_client_contributions
from utils.fed_xai_visualization import create_all_visualizations, save_explanation_dashboard
from integration.xai_integration import initialize_xai_components

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('distributed_experiments')

def get_cache_key(task: str, sample_fraction: float) -> str:
    """Generate a unique cache key for the preprocessing parameters"""
    params = f"{task}_{sample_fraction}"
    return hashlib.md5(params.encode()).hexdigest()

def load_cached_data(cache_key: str, cache_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Load preprocessed data from cache if available"""
    cache_file = os.path.join(cache_dir, f"{cache_key}.npz")
    if not os.path.exists(cache_file):
        return None
    
    try:
        logger.info(f"Loading cached data from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        return (
            pd.DataFrame(data['X_train']),
            pd.DataFrame(data['X_test']),
            pd.Series(data['y_train']),
            pd.Series(data['y_test']),
            data['features'].tolist()
        )
    except Exception as e:
        logger.warning(f"Error loading cached data: {str(e)}")
        return None

def save_to_cache(data: Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]], 
                 cache_key: str, cache_dir: str):
    """Save preprocessed data to cache"""
    cache_file = os.path.join(cache_dir, f"{cache_key}.npz")
    try:
        X_train, X_test, y_train, y_test, features = data
        np.savez(
            cache_file,
            X_train=X_train.values,
            X_test=X_test.values,
            y_train=y_train.values,
            y_test=y_test.values,
            features=np.array(features)
        )
        logger.info(f"Saved preprocessed data to cache: {cache_file}")
    except Exception as e:
        logger.warning(f"Error saving to cache: {str(e)}")

def get_preprocessed_data(task: str, sample_fraction: float, cache_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Get preprocessed data, using cache if available"""
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key
    cache_key = get_cache_key(task, sample_fraction)
    
    # Try to load from cache
    cached_data = load_cached_data(cache_key, cache_dir)
    if cached_data is not None:
        return cached_data
    
    # If not in cache, preprocess data
    logger.info("Preprocessing data (not found in cache)")
    data = preprocess_data(task=task, sample_fraction=sample_fraction)
    
    # Save to cache
    save_to_cache(data, cache_key, cache_dir)
    
    return data

def create_experiment_configs(quick_test: bool = False) -> Dict[str, Dict[str, Any]]:
    # Define baseline configuration first
    baseline_config = {
        "name": "baseline",
        "federated": {
            "num_clients": 10,
            "rounds": 25,
            "local_epochs": 2,
            "client_fraction": 0.8,
            "batch_size": 32
        },
        "xai_config": {
            "explainability": {
                "use_lime": True,
                "use_shap": True,
                "lime_samples": 1000,
                "shap_samples": 100,
                "max_features": 15
            },
            "privacy": {
                "enable_privacy": False
            }
        }
    }

    # Create all configurations using the baseline
    all_configs = {
        "baseline": baseline_config,
        "lime_only": {
            **baseline_config,  # Inherit all settings from baseline
            "name": "lime_only",
            "xai_config": {
                "explainability": {
                    "use_lime": True,
                    "use_shap": False,
                    "lime_samples": 2000  # Increased since it's the only method
                },
                "privacy": {
                    "enable_privacy": False
                }
            }
        },
        "shap_only": {
            **baseline_config,  # Inherit all settings from baseline
            "name": "shap_only",
            "xai_config": {
                "explainability": {
                    "use_lime": False,
                    "use_shap": True,
                    "shap_samples": 200  # Increased since it's the only method
                },
                "privacy": {
                    "enable_privacy": False
                }
            }
        }
    }

    if quick_test:
        # Modify configurations for quick testing
        for config in all_configs.values():
            config["federated"]["rounds"] = 2
            config["federated"]["num_clients"] = 3
            config["federated"]["local_epochs"] = 1

    return all_configs

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configurations, with override taking precedence"""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged

def simulate_federated_data(X: pd.DataFrame, y: pd.Series, num_clients: int = 3) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
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

def run_single_experiment(experiment_name: str, config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Run a single experiment with given configuration"""
    try:
        logger.info(f"Starting experiment: {experiment_name}")
        
        # Create experiment directory
        exp_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        # Create cache directory
        cache_dir = os.path.join(output_dir, 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load and preprocess data (with caching)
        X_train, X_test, y_train, y_test, features = get_preprocessed_data(
            task=config.get('task', 'mortality'),
            sample_fraction=config.get('data', {}).get('sample_fraction', 0.4),
            cache_dir=cache_dir
        )
        
        # Log data statistics
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Number of features: {len(features)}")
        logger.info(f"Positive samples in training: {sum(y_train)}")
        
        # Create and validate XAI configuration
        xai_config_dict = config.get('xai_config', {})
        explainability_config = xai_config_dict.get('explainability', {})
        privacy_config = xai_config_dict.get('privacy', {})
        aggregation_config = xai_config_dict.get('aggregation', {})
        
        # Set default values for explainability if not provided
        if not explainability_config:
            explainability_config = {
                'use_lime': True,
                'use_shap': False,
                'lime_samples': 1000,
                'max_features': 15,
                'explanations_per_client': 10,
                'explanation_batch_size': 5
            }
        
        # Set default values for privacy if not provided
        if not privacy_config:
            privacy_config = {
                'enable_privacy': False,
                'epsilon': 1.0,
                'delta': 1e-5,
                'clip_values': False,
                'min_samples': 5
            }
        
        # Set default values for aggregation if not provided
        if not aggregation_config:
            aggregation_config = {
                'aggregation_method': 'weighted_average',
                'temporal_decay': 0.8,
                'discard_outliers': True,
                'outlier_threshold': 2.0,
                'consistency_threshold': 0.5,
                'min_clients_per_round': 2
            }
        
        # Log XAI configuration
        logger.info("\nXAI Configuration:")
        logger.info(f"Explainability: {json.dumps(explainability_config, indent=2)}")
        logger.info(f"Privacy: {json.dumps(privacy_config, indent=2)}")
        logger.info(f"Aggregation: {json.dumps(aggregation_config, indent=2)}")
        
        # Validate explainability configuration
        if explainability_config.get('use_lime'):
            if explainability_config.get('lime_samples', 0) < 100:
                logger.warning("LIME samples too low, increasing to 1000")
                explainability_config['lime_samples'] = 1000
            if explainability_config.get('max_features', 0) < 5:
                logger.warning("Max features too low for LIME, increasing to 10")
                explainability_config['max_features'] = 10
        
        if explainability_config.get('use_shap'):
            if explainability_config.get('shap_samples', 0) < 10:
                logger.warning("SHAP samples too low, increasing to 100")
                explainability_config['shap_samples'] = 100
        
        # Create XAI configuration with validated parameters
        explanations_dir = os.path.join(exp_dir, 'explanations')
        os.makedirs(explanations_dir, exist_ok=True)
        
        # Calculate explanation rounds based on total rounds
        total_rounds = config.get('federated', {}).get('rounds', 25)
        explanation_rounds = [5, 10, 15, 20, 25]  # Default rounds
        # Ensure we don't exceed total rounds
        explanation_rounds = [r for r in explanation_rounds if r <= total_rounds]
        # Ensure we have at least one round
        if not explanation_rounds:
            explanation_rounds = [total_rounds]
        
        xai_config = FederatedXAIConfig(
            explainability=ExplainabilityConfig(
                **explainability_config
            ),
            privacy=PrivacyConfig(
                **privacy_config
            ),
            aggregation=AggregationConfig(
                **aggregation_config
            ),
            collect_explanations=True,
            explanation_rounds=explanation_rounds,  # Use calculated rounds
            save_explanations=True,
            explanations_path=explanations_dir
        )
        
        # Log XAI configuration details
        logger.info("\nDetailed XAI Configuration:")
        logger.info(f"Explanations directory: {explanations_dir}")
        logger.info(f"Explanation rounds: {xai_config.explanation_rounds}")
        logger.info(f"Total training rounds: {total_rounds}")
        logger.info(f"Collect explanations: {xai_config.collect_explanations}")
        logger.info(f"Save explanations: {xai_config.save_explanations}")
        
        # Validate XAI configuration
        if xai_config.explainability.use_lime:
            logger.info(f"LIME configuration validated:")
            logger.info(f"- Samples: {xai_config.explainability.lime_samples}")
            logger.info(f"- Max features: {xai_config.explainability.max_features}")
        
        if xai_config.explainability.use_shap:
            logger.info(f"SHAP configuration validated:")
            logger.info(f"- Samples: {xai_config.explainability.shap_samples}")
        
        # Create model configuration
        model_config = config.get('model', {
            'input_dim': X_train.shape[1],
            'hidden_sizes': [256, 128, 64],
            'dropout': 0.3,
            'learning_rate': 0.001
        })
        
        # Create orchestrator
        orchestrator = FederatedOrchestrator(
            input_dim=X_train.shape[1],
            features=features,
            model_config=model_config,
            xai_config=xai_config
        )
        
        # Initialize XAI components
        initialize_xai_components(orchestrator, xai_config)
        
        # Simulate federated data
        num_clients = config.get('federated', {}).get('num_clients', 10)
        client_data = simulate_federated_data(X_train, y_train, num_clients=num_clients)
        
        # Add clients to orchestrator
        for client_id, (X, y) in client_data.items():
            orchestrator.add_client(client_id, X, y)
            logger.info(f"Client {client_id} data shape: {X.shape}, positive samples: {sum(y)}")
        
        # Train federated model
        history = orchestrator.train_federated(
            num_rounds=config.get('federated', {}).get('rounds', 25),
            local_epochs=config.get('federated', {}).get('local_epochs', 2),
            client_fraction=config.get('federated', {}).get('client_fraction', 0.8),
            batch_size=config.get('federated', {}).get('batch_size', 32),
            min_clients=3,
            validation_data=(X_test, y_test)
        )
        
        # Save visualizations
        save_training_plots(history, output_dir=exp_dir)
        save_client_contributions(history, output_dir=exp_dir)
        
        # Generate explanation visualizations if explanations were collected
        if orchestrator.explanation_history:
            logger.info("\nGenerating explanation visualizations...")
            logger.info(f"Number of explanation rounds: {len(orchestrator.explanation_history)}")
            
            # Log explanation statistics and save explanations
            for round_num, round_data in orchestrator.explanation_history.items():
                round_dir = os.path.join(explanations_dir, f'round_{round_num}')
                os.makedirs(round_dir, exist_ok=True)
                
                # Save round explanations
                round_file = os.path.join(round_dir, 'explanations.json')
                with open(round_file, 'w') as f:
                    json.dump(round_data, f, indent=2)
                logger.info(f"Saved explanations for round {round_num} to {round_file}")
                
                if 'lime' in round_data:
                    lime_values = round_data['lime']
                    logger.info(f"\nRound {round_num} LIME statistics:")
                    logger.info(f"Number of features: {len(lime_values)}")
                    logger.info(f"Non-zero values: {sum(1 for v in lime_values.values() if v != 0)}")
                    logger.info(f"Max importance: {max(lime_values.values()):.4f}")
                    logger.info(f"Min importance: {min(lime_values.values()):.4f}")
                    
                    # Save LIME values separately
                    lime_file = os.path.join(round_dir, 'lime_values.json')
                    with open(lime_file, 'w') as f:
                        json.dump(lime_values, f, indent=2)
                    logger.info(f"Saved LIME values to {lime_file}")
            
            visualization_files = create_all_visualizations(
                orchestrator.explanation_history,
                orchestrator.client_explanations,
                feature_names=features,
                output_dir=exp_dir
            )
            logger.info(f"Generated visualization files: {visualization_files}")
        else:
            logger.warning("No explanation history found in orchestrator")
        
        # Get final metrics
        summary = orchestrator.get_training_summary()
        
        # Save experiment results
        results = {
            'experiment_name': experiment_name,
            'config': config,
            'summary': summary,
            'history': history
        }
        
        results_path = os.path.join(exp_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Experiment {experiment_name} completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error in experiment {experiment_name}: {str(e)}")
        logger.error("Detailed error: ", exc_info=True)
        raise

def run_experiments_parallel(experiments: Dict[str, Dict[str, Any]], output_dir: str, max_workers: int = 4):
    """Run multiple experiments in parallel"""
    results = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_exp = {
            executor.submit(run_single_experiment, exp_name, config, output_dir): exp_name
            for exp_name, config in experiments.items()
        }
        
        for future in future_to_exp:
            exp_name = future_to_exp[future]
            try:
                results[exp_name] = future.result()
            except Exception as e:
                logger.error(f"Experiment {exp_name} failed: {str(e)}")
                results[exp_name] = {'error': str(e)}
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run distributed federated XAI experiments')
    parser.add_argument('--output_dir', type=str, default='./results',
                      help='Directory to save experiment results')
    parser.add_argument('--max_workers', type=int, default=4,
                      help='Maximum number of parallel workers')
    parser.add_argument('--experiments', type=str, nargs='+',
                      help='Specific experiments to run (default: all)')
    parser.add_argument('--quick_test', action='store_true',
                      help='Run quick test mode with minimal settings')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get experiment configurations
    all_configs = create_experiment_configs(quick_test=args.quick_test)
    
    # Filter experiments if specified
    if args.experiments:
        configs = {name: config for name, config in all_configs.items() 
                  if name in args.experiments}
    else:
        configs = all_configs
    
    # Run experiments in parallel
    logger.info(f"Starting {len(configs)} experiments with {args.max_workers} workers")
    logger.info(f"Quick test mode: {args.quick_test}")
    results = run_experiments_parallel(configs, args.output_dir, args.max_workers)
    
    # Save overall results
    summary_path = os.path.join(args.output_dir, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"All experiments completed. Summary saved to {summary_path}")

if __name__ == "__main__":
    main() 