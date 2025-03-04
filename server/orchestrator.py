import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import sys
import os
import time
import tensorflow as tf
from sklearn.metrics import roc_auc_score

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules with correct paths
from server.fed_server import FederatedServer
from models.fed_client import FederatedClient
from utils.fed_visualization import save_training_plots, save_client_contributions

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('orchestrator')

class FederatedOrchestrator:
    """Orchestrator for federated learning training process"""
    
    def __init__(
        self,
        input_dim: int,
        features: Union[List[str], pd.Index],
        model_config: Optional[Dict] = None,
        aggregation_method: str = 'fedavg'
    ):
        """Initialize orchestrator"""
        self.model_config = model_config or {}
        
        # Initialize server
        self.server = FederatedServer(
            input_dim=input_dim,
            features=features,
            model_config=model_config,
            aggregation_method=aggregation_method
        )
        
        # Store clients
        self.clients: Dict[str, FederatedClient] = {}
        
        # Training configuration
        self.current_round = 0
        self.training_history = []
        
        logger.info("Initialized federated orchestrator")

   
    
    def add_client(
        self,
        client_id: str,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ):
        """Add new client with its data"""
        try:
            # Create client instance
            client = FederatedClient(
                client_id=client_id,
                input_dim=self.server.input_dim,
                features=self.server.features,
                model_config=self.model_config
            )
            
            # Register client with server
            self.server.register_client(client_id, len(X))
            
            # Store client
            self.clients[client_id] = {
                'client': client,
                'data': (X, y)
            }
            
            logger.info(f"Added client {client_id} with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error adding client {client_id}: {str(e)}")
            raise
    
    def train_federated(
        self,
        num_rounds: int,
        local_epochs: int = 1,
        min_clients: int = 2,
        client_fraction: float = 1.0,
        batch_size: int = 32,
        validation_data: Optional[Tuple] = None
    ) -> Dict:
        """Run federated training process"""
        logger.info(f"Starting federated training for {num_rounds} rounds")
        logger.info(f"Number of clients: {len(self.clients)}")
        
        try:
            if len(self.clients) < min_clients:
                raise ValueError(f"Not enough clients. Have {len(self.clients)}, need {min_clients}")
            
            # Initialize history structure
            training_history = {
                'rounds': [],
                'global_metrics': [],
                'client_metrics': {}
            }
            
            # Training rounds
            for round_num in range(num_rounds):
                self.current_round = round_num + 1
                logger.info(f"\nStarting round {self.current_round}/{num_rounds}")
                round_start_time = time.time()
                
                # Select clients for this round
                participating_clients = self._select_clients(client_fraction)
                logger.info(f"Selected {len(participating_clients)} clients for this round")
                
                # Get current global weights
                global_weights = self.server.get_global_weights()
                client_weights = {}
                round_client_metrics = {}
                
                # Train selected clients
                for client_id in participating_clients:
                    client_info = self.clients[client_id]
                    client = client_info['client']  # Get the actual client object
                    X, y = client_info['data']  # Get client's data
                    
                    # Update client with global model
                    client.update_local_model(global_weights)
                    
                    # Train client locally
                    history = client.train_local_model(
                        X, y,  # Use client's own data
                        epochs=local_epochs,
                        batch_size=batch_size
                    )
                    
                    # Get updated weights and metrics
                    client_weights[client_id] = client.get_local_weights()
                    metrics = client.evaluate_local_model(X, y)  # Evaluate on client's data
                    round_client_metrics[client_id] = metrics
                    
                    # Update server's client metadata
                    self.server.update_client_metadata(
                        client_id, metrics, self.current_round
                    )
                
                # Update global model
                self.server.update_global_model(client_weights)
                
                # Evaluate global model
                global_metrics = {}
                if validation_data is not None:
                    X_val, y_val = validation_data
                    global_metrics = self.server.evaluate_global_model(X_val, y_val)
                
                # Store round results
                round_data = {
                    'round': self.current_round,
                    'client_metrics': round_client_metrics,
                    'global_metrics': global_metrics,
                    'duration': time.time() - round_start_time
                }
                
                training_history['rounds'].append(round_data)
                training_history['global_metrics'].append(global_metrics)
                
                # Log round summary
                self._log_round_summary(round_data)
                
                # Save intermediate plots (won't generate during rounds)
                save_training_plots(training_history, is_final=False)
                save_client_contributions(training_history, is_final=False)
            
            # Save final plots
            save_training_plots(training_history, is_final=True)
            save_client_contributions(training_history, is_final=True)
            
            logger.info("Federated training completed")
            return training_history
            
        except Exception as e:
            logger.error(f"Error in federated training: {str(e)}")
            raise
    
    def _select_clients(self, fraction: float) -> List[str]:
        """Select clients for current round"""
        num_clients = max(1, int(len(self.clients) * fraction))
        return np.random.choice(
            list(self.clients.keys()),
            size=num_clients,
            replace=False
        )
    
    def _log_round_summary(self, metrics: Dict):
        """Log summary of training round"""
        logger.info(f"\nRound {metrics['round']} Summary:")
        logger.info(f"Duration: {metrics['duration']:.2f} seconds")
        
        if metrics['global_metrics']:
            logger.info("\nGlobal Metrics:")
            for metric, value in metrics['global_metrics'].items():
                if isinstance(value, float):
                    logger.info(f"- {metric}: {value:.4f}")
        
        logger.info("\nClient Metrics:")
        for client_id, client_metrics in metrics['client_metrics'].items():
            logger.info(f"\nClient {client_id}:")
            for metric, value in client_metrics.items():
                if isinstance(value, float):
                    logger.info(f"- {metric}: {value:.4f}")
    
    def get_global_model(self) -> tf.keras.Model:
        """Get current global model"""
        return self.server.global_model
    
    def get_training_summary(self) -> Dict:
        """Get summary of entire training process"""
        return {
            'num_rounds': self.current_round,
            'num_clients': len(self.clients),
            'server_summary': self.server.get_training_summary(),
            'final_metrics': self.training_history[-1] if self.training_history else None
        }