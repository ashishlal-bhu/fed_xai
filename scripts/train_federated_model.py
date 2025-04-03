import sys
import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from scripts.preprocess_data import preprocess_data
from server.orchestrator import FederatedOrchestrator
from utils.fed_visualization import save_training_plots, save_client_contributions
from utils.fed_xai_visualization import create_all_visualizations, save_explanation_dashboard
from configuration.xai_config import FederatedXAIConfig, LIME_ONLY_CONFIG, SHAP_ONLY_CONFIG, STRICT_PRIVACY_CONFIG
from integration.xai_integration import initialize_xai_components

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('federated_training')

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

def main(xai_config=None, rounds=25):
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        X_train, X_test, y_train, y_test, features = preprocess_data(task="mortality", sample_fraction=0.4)
        
        logger.info(f"Full dataset shape: {X_train.shape}")
        logger.info(f"Number of features: {len(features)}")
        
        # Define consistent model architecture
        hidden_sizes = [256, 128, 64]  # Fixed sizes for all layers
        model_config = {
            'input_dim': X_train.shape[1],
            'hidden_sizes': hidden_sizes,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'l2_reg': 0.01
        }
        
        logger.info("Model configuration:")
        for key, value in model_config.items():
            logger.info(f"- {key}: {value}")
        
        # Use provided XAI config or create a custom one
        if xai_config is None:
            # Create a custom XAI configuration
            xai_config = FederatedXAIConfig(
                # Configure explainability components
                explainability={
                    'use_lime': True,          # Enable LIME explanations
                    'use_shap': True,          # Enable SHAP explanations
                    'lime_samples': 1000,      # Number of samples for LIME
                    'shap_samples': 100,       # Number of samples for SHAP background
                    'max_features': 15,        # Show top 15 features in explanations
                    'explanations_per_client': 10,  # Number of examples to explain per client
                    'explanation_batch_size': 10    # Increased batch size for explanations
                },
                # Configure privacy components
                privacy={
                    'enable_privacy': False,   # Disable privacy protections temporarily
                    'epsilon': 1.0,            # Differential privacy budget
                    'clip_values': False,      # Disable value clipping
                    'clip_range': [-1.0, 1.0]  # Range for clipping
                },
                # Configure aggregation components
                aggregation={
                    'aggregation_method': 'weighted_average',  # Use weighted averaging
                    'discard_outliers': True,                 # Remove outlier explanations
                    'min_clients_per_round': 2                # Minimum clients for valid explanation
                },
                # Global settings
                collect_explanations=True,
                explanation_rounds=[5, 10, 15, 20, 25],  # Only collect explanations on these rounds
                save_explanations=True
            )
        
        logger.info("XAI Configuration:")
        logger.info(f"- Explainability: LIME={xai_config.explainability.use_lime}, SHAP={xai_config.explainability.use_shap}")
        logger.info(f"- Privacy: Enabled={xai_config.privacy.enable_privacy}, Epsilon={xai_config.privacy.epsilon}")
        logger.info(f"- Aggregation Method: {xai_config.aggregation.aggregation_method}")
        logger.info(f"- Explanation Rounds: {xai_config.explanation_rounds}")
        
        # Initialize orchestrator with XAI configuration
        orchestrator = FederatedOrchestrator(
            input_dim=X_train.shape[1],
            features=features,
            model_config=model_config,
            xai_config=xai_config  # Pass XAI configuration to orchestrator
        )
        
        # Simulate federated data
        client_data = simulate_federated_data(X_train, y_train, num_clients=10)
        
        # Add clients to orchestrator
        for client_id, (X, y) in client_data.items():
            orchestrator.add_client(client_id, X, y)
        
        # Train federated model
        logger.info("Starting federated training with XAI components...")
        history = orchestrator.train_federated(
            num_rounds=rounds,
            local_epochs=2,
            client_fraction=0.8,
            batch_size=32,
            min_clients=3,
            validation_data=(X_test, y_test)
        )
        
        # Plot training progress
        save_training_plots(history)
        save_client_contributions(history)
        
        # Generate explanation visualizations if explanations were collected
        if orchestrator.explanation_history:
            logger.info("Generating explanation visualizations...")
            visualization_files = create_all_visualizations(
                orchestrator.explanation_history,
                orchestrator.client_explanations,
                feature_names=features
            )
            
            # Create explanation dashboard
            dashboard_path = save_explanation_dashboard(
                orchestrator.explanation_history
            )
            logger.info(f"Explanation dashboard saved to: {dashboard_path}")
        
        # Get final training summary
        summary = orchestrator.get_training_summary()
        logger.info("\nTraining Summary:")
        logger.info(f"Number of rounds: {summary['num_rounds']}")
        logger.info(f"Number of clients: {summary['num_clients']}")
        logger.info(f"Explanations collected: {summary.get('explanations_collected', False)}")
        
        if summary.get('final_metrics'):
            logger.info("\nFinal Metrics:")
            for metric, value in summary['final_metrics'].items():
                if isinstance(value, float):
                    logger.info(f"{metric}: {value:.4f}")
        
        return orchestrator, history
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error("Detailed error: ", exc_info=True)
        raise

def run_xai_experiments():
    """Run a series of XAI experiments with different configurations"""
    logger.info("Starting XAI experiments...")
    
    # Experiment 1: LIME only
    logger.info("\n=== Experiment 1: LIME Only ===")
    orchestrator1, history1 = main(xai_config=LIME_ONLY_CONFIG, rounds=10)
    
    # Experiment 2: SHAP only
    logger.info("\n=== Experiment 2: SHAP Only ===")
    orchestrator2, history2 = main(xai_config=SHAP_ONLY_CONFIG, rounds=10)
    
    # Experiment 3: High Privacy
    logger.info("\n=== Experiment 3: Strict Privacy ===")
    orchestrator3, history3 = main(xai_config=STRICT_PRIVACY_CONFIG, rounds=10)
    
    logger.info("All XAI experiments completed")
    
    # Compare explanations across experiments
    # (Add more analysis code here if needed)

if __name__ == "__main__":
    try:
        # Run single XAI-enabled training
        orchestrator, history = main()
        logger.info("Training with XAI completed successfully")
        
        # Uncomment to run all experiments
        # run_xai_experiments()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1) 
        