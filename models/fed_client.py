import logging
from typing import Dict, List, Union, Optional
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from models.fed_xai_model import FederatedXAIModel

logger = logging.getLogger('fed_client')

class FederatedClient:
    """Client implementation for federated learning"""
    
    def __init__(
        self,
        client_id: str,
        input_dim: int,
        features: Union[List[str], pd.Index],
        model_config: Optional[Dict] = None
    ):
        self.client_id = client_id
        self.input_dim = input_dim
        self.features = features
        
        # Initialize local model
        self.local_model = FederatedXAIModel(
            input_dim=input_dim,
            features=features,
            model_config=model_config
        )
        
        logger.info(f"Initialized client {client_id}")
    
    def update_local_model(self, global_weights: List[np.ndarray]):
        """Update local model with global weights"""
        try:
            self.local_model.set_weights(global_weights)
            logger.info(f"Client {self.client_id} updated with global weights")
        except Exception as e:
            logger.error(f"Error updating local model: {str(e)}")
            raise
    
    def train_local_model(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        **train_kwargs
    ):
        """Train local model on client data"""
        try:
            history = self.local_model.local_train(X, y, **train_kwargs)
            logger.info(f"Client {self.client_id} completed local training")
            return history
        except Exception as e:
            logger.error(f"Error in local training: {str(e)}")
            raise
    
    def get_local_weights(self) -> List[np.ndarray]:
        """Get current local model weights"""
        return self.local_model.get_weights()
    
    def initialize_explainers(self, X_train: Union[np.ndarray, pd.DataFrame]):
        """Initialize local explainers"""
        try:
            self.local_model.initialize_explainers(X_train)
            logger.info(f"Client {self.client_id} initialized explainers")
        except Exception as e:
            logger.error(f"Error initializing explainers: {str(e)}")
            raise
    
    def explain_instance(self, instance: Union[np.ndarray, pd.Series]) -> Dict:
        """Get local explanation for an instance"""
        try:
            explanation = self.local_model.explain_instance(instance)
            return {
                'client_id': self.client_id,
                'explanations': explanation
            }
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            raise
    
    def evaluate_local_model(
        self,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series]
    ) -> Dict:
        """Evaluate local model performance"""
        try:
            y_pred = self.local_model.predict(X_test)
            y_pred_proba = self.local_model.predict_proba(X_test)[:, 1]
            
            metrics = {
                'accuracy': np.mean(y_pred == y_test),
                'auc': roc_auc_score(y_test, y_pred_proba),
                'client_id': self.client_id
            }
            
            logger.info(f"Client {self.client_id} evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating local model: {str(e)}")
            raise