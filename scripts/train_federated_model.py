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

def main():
    try:
        # Load and preprocess data
        logger.info("Loading and preprocessing data...")
        X_train, X_test, y_train, y_test, features = preprocess_data(task="mortality",sample_fraction=0.4)
        
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
        
        # Initialize orchestrator
        orchestrator = FederatedOrchestrator(
            input_dim=X_train.shape[1],
            features=features,
            model_config=model_config
        )
        
        # Simulate federated data
        client_data = simulate_federated_data(X_train, y_train, num_clients=10)
        
        # Add clients to orchestrator
        for client_id, (X, y) in client_data.items():
            orchestrator.add_client(client_id, X, y)
        
        # Train federated model
        logger.info("Starting federated training...")
        history = orchestrator.train_federated(
            num_rounds=25,
            local_epochs=2,
            client_fraction=0.8,
            batch_size=32,
            min_clients=3,
            validation_data=(X_test, y_test)
        )
        
        # Plot training progress
        save_training_plots(history)
        save_client_contributions(history)
        
        # Get final training summary
        summary = orchestrator.get_training_summary()
        logger.info("\nTraining Summary:")
        logger.info(f"Number of rounds: {summary['num_rounds']}")
        logger.info(f"Number of clients: {summary['num_clients']}")
        
        if summary['final_metrics']:
            logger.info("\nFinal Metrics:")
            for metric, value in summary['final_metrics'].items():
                if isinstance(value, float):
                    logger.info(f"{metric}: {value:.4f}")
        
        return orchestrator, history
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        logger.error("Detailed error: ", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        orchestrator, history = main()
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)