import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, classification_report
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from models.model_factory import create_model

logger = logging.getLogger('fed_server')

class FederatedServer:
    def __init__(
        self,
        input_dim: int,
        features: Union[List[str], pd.Index],
        model_config: Optional[Dict] = None,
        aggregation_method: str = 'fedavg'
    ):
        self.input_dim = input_dim
        self.features = features
        self.model_config = model_config
        self.aggregation_method = aggregation_method
        
        # Initialize global model using factory
        self.global_model = create_model(input_dim, model_config)
        
        # Store client metadata
        self.clients = {}
        
        logger.info(f"Initialized federated server with {aggregation_method} aggregation")

    def aggregate_weights(self, client_weights: Dict[str, List[np.ndarray]]) -> List[np.ndarray]:
        """
        Aggregate client weights using FedAvg
        
        Args:
            client_weights: Dictionary mapping client IDs to their model weights
            
        Returns:
            Aggregated weights
        """
        logger.info(f"Aggregating weights from {len(client_weights)} clients")
        
        try:
            # Get number of clients
            n_clients = len(client_weights)
            if n_clients == 0:
                raise ValueError("No client weights to aggregate")

            # Initialize aggregated weights with zeros
            sample_weights = next(iter(client_weights.values()))
            aggregated_weights = [np.zeros_like(w) for w in sample_weights]
            
            # Simple averaging of weights
            for client_id, weights in client_weights.items():
                for i, w in enumerate(weights):
                    aggregated_weights[i] += w / n_clients
            
            # Verify aggregated weights have same structure as original
            if len(aggregated_weights) != len(sample_weights):
                raise ValueError(
                    f"Aggregated weights length ({len(aggregated_weights)}) "
                    f"doesn't match original ({len(sample_weights)})"
                )
            
            logger.info("Successfully aggregated weights")
            logger.info(f"Number of weight tensors: {len(aggregated_weights)}")
            
            return aggregated_weights
            
        except Exception as e:
            logger.error(f"Error aggregating weights: {str(e)}")
            raise

    def update_global_model(self, client_weights: Dict[str, List[np.ndarray]]):
        """Update global model with aggregated weights"""
        try:
            # Verify we have weights to aggregate
            if not client_weights:
                raise ValueError("No client weights provided")
            
            # Get aggregated weights
            aggregated_weights = self.aggregate_weights(client_weights)
            
            # Verify weights before updating
            original_weights = self.global_model.get_weights()
            if len(aggregated_weights) != len(original_weights):
                raise ValueError(
                    f"Aggregated weights length ({len(aggregated_weights)}) "
                    f"doesn't match model weights ({len(original_weights)})"
                )
            
            # Update global model
            self.global_model.set_weights(aggregated_weights)
            logger.info("Updated global model with aggregated weights")
            
        except Exception as e:
            logger.error(f"Error updating global model: {str(e)}")
            raise

    def evaluate_global_model(
        self,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series]
    ) -> Dict:
        """
        Evaluate global model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating global model performance...")
        
        try:
            # Get predictions
            y_pred_proba = self.global_model.predict(X_test, verbose=0)
            # Ensure predictions are 1D
            y_pred_proba = y_pred_proba.ravel()
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Convert y_test to numpy array if it's a pandas Series
            if isinstance(y_test, pd.Series):
                y_test = y_test.values
                
            # Calculate metrics
            metrics = {
                'accuracy': float(np.mean(y_pred == y_test)),
                'auc': float(roc_auc_score(y_test, y_pred_proba)),
                'report': classification_report(y_test, y_pred, output_dict=True)
            }
            
            logger.info("Global model evaluation:")
            logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"AUC: {metrics['auc']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating global model: {str(e)}")
            logger.error("Detailed error: ", exc_info=True)
            raise

    def get_global_weights(self) -> List[np.ndarray]:
        """Get current global model weights"""
        return self.global_model.get_weights()

    def register_client(self, client_id: str, data_size: int):
        """Register a new client"""
        if client_id not in self.clients:
            self.clients[client_id] = {
                'data_size': data_size,
                'contribution_score': 1.0,
                'last_update': 0
            }
            logger.info(f"Registered client {client_id} with {data_size} samples")

    def update_client_metadata(self, client_id: str, metrics: Dict, round_number: int):
        """Update client metadata with latest performance"""
        if client_id in self.clients:
            # Update contribution score based on AUC
            auc = metrics.get('auc', 0.5)
            old_score = self.clients[client_id]['contribution_score']
            new_score = 0.7 * old_score + 0.3 * auc
            
            self.clients[client_id]['contribution_score'] = new_score
            self.clients[client_id]['last_update'] = round_number
            
            logger.info(f"Updated metadata for client {client_id}")
            logger.info(f"New contribution score: {new_score:.4f}")
        else:
            logger.warning(f"Tried to update unknown client: {client_id}")

    def get_training_summary(self) -> Dict:
        """Get summary of training process"""
        return {
            'num_clients': len(self.clients),
            'client_scores': {
                client_id: info['contribution_score']
                for client_id, info in self.clients.items()
            },
            'client_updates': {
                client_id: info['last_update']
                for client_id, info in self.clients.items()
            }
        }