# examples/federated_xai_example.py
"""
Example usage of the Federated XAI framework.

This script demonstrates how to use the configuration system and
integration components to enable explainability in federated learning.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from server.orchestrator import FederatedOrchestrator
from configuration.xai_config import (
    FederatedXAIConfig, 
    ExplainabilityConfig,
    PrivacyConfig, 
    AggregationConfig,
    STRICT_PRIVACY_CONFIG,
    NO_PRIVACY_CONFIG
)
from integration.xai_integration import initialize_xai_components
from utils.fed_visualization import (
    save_training_plots, 
    save_client_contributions
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('federated_xai_example')

# Example function to simulate federated data
def simulate_federated_data(X, y, num_clients=3):
    """Simulate federated data by splitting into multiple clients."""
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

def main():
    try:
        # Define model architecture (example)
        model_config = {
            'hidden_sizes': [256, 128, 64],
            'dropout': 0.3,
            'learning_rate': 0.001,
            'l2_reg': 0.01
        }

        # Load and preprocess data (usually from preprocess_data.py)
        # For this example, we'll create synthetic data
        from sklearn.datasets import make_classification
        
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=10,
            n_redundant=5,
            n_classes=2,
            random_state=42
        )
        
        # Convert to DataFrame for more realistic scenario
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=feature_names)
        y = pd.Series(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}")
        
        # Create XAI configuration
        # Option 1: Use a preset configuration
        xai_config = NO_PRIVACY_CONFIG  # For initial testing

        # Option 2: Create custom configuration
        custom_xai_config = FederatedXAIConfig(
            explainability=ExplainabilityConfig(
                use_lime=True,
                use_shap=True,
                max_features=10,
                explanations_per_client=5
            ),
            privacy=PrivacyConfig(
                enable_privacy=True,
                epsilon=2.0,
                clip_values=True
            ),
            aggregation=AggregationConfig(
                aggregation_method='weighted_average',
                temporal_decay=0.7
            ),
            collect_explanations=True,
            explanation_rounds=[1, 5, 10],  # Only collect on these rounds
            save_explanations=True,
            explanations_path='./results/explanations'
        )
        
        # Initialize orchestrator
        orchestrator = FederatedOrchestrator(
            input_dim=X_train.shape[1],
            features=feature_names,
            model_config=model_config
        )
        
        # Simulate federated data
        client_data = simulate_federated_data(X_train, y_train, num_clients=3)
        
        # Add clients to orchestrator
        for client_id, (X, y) in client_data.items():
            orchestrator.add_client(client_id, X, y)
        
        # Initialize XAI components (choose one configuration)
        # initialize_xai_components(orchestrator, xai_config)  # Use preset config
        initialize_xai_components(orchestrator, custom_xai_config)  # Use custom config
        
        # Train federated model with explainability
        logger.info("Starting federated training with explainability...")
        history = orchestrator.train_federated(
            num_rounds=10,
            local_epochs=1,
            client_fraction=0.8,
            batch_size=32,
            min_clients=2,
            validation_data=(X_test, y_test)
        )
        
        # Access explanations
        for round_num, round_data in enumerate(history['rounds']):
            global_explanation = orchestrator.explanation_aggregator.get_global_explanation(round_num + 1)
            
            if global_explanation:
                logger.info(f"\nGlobal Explanation for Round {round_num + 1}:")
                
                if 'explanations' in global_explanation and 'lime' in global_explanation['explanations']:
                    lime_exp = global_explanation['explanations']['lime']
                    logger.info("Top LIME features:")
                    for feature, importance in sorted(
                        lime_exp.items(), key=lambda x: abs(x[1]), reverse=True
                    )[:5]:
                        logger.info(f"  {feature}: {importance:.4f}")
        
        # Save visualization plots
        save_training_plots(history)
        save_client_contributions(history)
        
        # Get final training summary
        summary = orchestrator.get_training_summary()
        logger.info("\nTraining Summary:")
        logger.info(f"Number of rounds: {summary['num_rounds']}")
        logger.info(f"Number of clients: {summary['num_clients']}")
        
        # Example of running an ablation study
        logger.info("\nRunning ablation study - LIME only...")
        
        # Create LIME-only configuration
        lime_only_config = FederatedXAIConfig(
            explainability=ExplainabilityConfig(
                use_lime=True,
                use_shap=False
            ),
            privacy=custom_xai_config.privacy,
            aggregation=custom_xai_config.aggregation,
            collect_explanations=True,
            explanation_rounds=[1, 5, 10],
            save_explanations=True,
            explanations_path='./results/lime_only_explanations'
        )
        
        # Reinitialize orchestrator for ablation study
        ablation_orchestrator = FederatedOrchestrator(
            input_dim=X_train.shape[1],
            features=feature_names,
            model_config=model_config
        )
        
        # Add clients and initialize with LIME-only config
        for client_id, (X, y) in client_data.items():
            ablation_orchestrator.add_client(client_id, X, y)
        
        initialize_xai_components(ablation_orchestrator, lime_only_config)
        
        # Run training with LIME-only configuration
        lime_history = ablation_orchestrator.train_federated(
            num_rounds=10,
            local_epochs=1,
            client_fraction=0.8,
            batch_size=32,
            min_clients=2,
            validation_data=(X_test, y_test)
        )
        
        # Compare results between configurations
        logger.info("\nComparison of different configurations:")
        logger.info("Full XAI vs LIME-only accuracy comparison:")
        
        full_accuracy = [m['accuracy'] for m in history['global_metrics']]
        lime_accuracy = [m['accuracy'] for m in lime_history['global_metrics']]
        
        for i in range(len(full_accuracy)):
            logger.info(f"Round {i+1}: Full XAI = {full_accuracy[i]:.4f}, "
                       f"LIME-only = {lime_accuracy[i]:.4f}")
        
        return orchestrator, history, ablation_orchestrator, lime_history
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error("Detailed error: ", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        orchestrator, history, ablation_orchestrator, lime_history = main()
        logger.info("Example completed successfully")
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)