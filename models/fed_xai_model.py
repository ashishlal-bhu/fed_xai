# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Scientific Computing
import numpy as np
import pandas as pd

# Machine Learning Metrics
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin

# Explainability Libraries
import shap
import lime
import lime.lime_tabular

# Utilities
import logging
from typing import Dict, List, Tuple, Optional, Union
import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from models.model_factory import create_model

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fed_xai_model')

class FederatedXAIModel(BaseEstimator, ClassifierMixin):
    """
    Federated XAI Model for mortality prediction with LIME and SHAP explanations.
    Supports local training and model weight updates for federated learning.
    """
    
    def __init__(
        self,
        input_dim: int,
        features: Union[List[str], pd.Index],
        model_config: Optional[Dict] = None
    ):
        """Initialize Federated XAI Model"""
        logger.info("Initializing Federated XAI Model...")
        
        self.input_dim = input_dim
        self.features = features
        self.feature_names = self._validate_features(features)
        self.model_config = model_config
        
        # Initialize model using factory
        self.model = create_model(input_dim, model_config)
        
        # Initialize explainers
        self.lime_explainer = None
        self.shap_explainer = None
        
        logger.info(f"Model initialized with {input_dim} features")

    def _validate_features(self, features: Union[List[str], pd.Index]) -> List[str]:
        """
        Validate and process feature names.
        
        Args:
            features: List or Index of feature names
            
        Returns:
            List of validated feature names
        """
        if isinstance(features, pd.DataFrame):
            feature_names = features.columns.tolist()
        elif isinstance(features, pd.Index):
            feature_names = features.tolist()
        elif isinstance(features, (list, np.ndarray)):
            feature_names = list(features)
        else:
            raise ValueError(f"Unsupported features type: {type(features)}")
            
        if len(feature_names) != self.input_dim:
            raise ValueError(
                f"Number of features ({len(feature_names)}) "
                f"doesn't match input_dim ({self.input_dim})"
            )
            
        return feature_names

    def _validate_input_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Tuple:
        """
        Validate input data format and dimensions.
        
        Args:
            X: Input features
            y: Optional target values
            
        Returns:
            Tuple of validated (X, y) or just X if y is None
        """
        # Validate X
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        elif isinstance(X, pd.Series):
            X_arr = X.values.reshape(1, -1)
        elif isinstance(X, np.ndarray):
            X_arr = X.reshape(-1, self.input_dim) if len(X.shape) == 1 else X
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")
            
        if X_arr.shape[1] != self.input_dim:
            raise ValueError(f"Input has {X_arr.shape[1]} features, expected {self.input_dim}")
            
        # Validate y if provided
        if y is not None:
            if isinstance(y, pd.Series):
                y_arr = y.values
            elif isinstance(y, np.ndarray):
                y_arr = y
            else:
                raise ValueError(f"Unsupported target type: {type(y)}")
                
            if len(y_arr.shape) != 1:
                raise ValueError(f"Target should be 1D, got shape {y_arr.shape}")
                
            if len(y_arr) != len(X_arr):
                raise ValueError(f"Length mismatch: X has {len(X_arr)} samples, y has {len(y_arr)}")
                
            return X_arr, y_arr
            
        return X_arr

    def local_train(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        validation_split: float = 0.2,
        epochs: int = 1,
        batch_size: int = 32
    ) -> tf.keras.callbacks.History:
        """
        Train model on local data.
        
        Args:
            X: Training features
            y: Training targets
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        logger.info("Starting local training...")
        
        try:
            # Validate input data
            X_arr, y_arr = self._validate_input_data(X, y)
            X_arr = X_arr.astype('float32')
            
            # Calculate class weights
            unique, counts = np.unique(y_arr, return_counts=True)
            total = len(y_arr)
            class_weight = {int(val): total/(len(unique) * count) 
                          for val, count in zip(unique, counts)}
            
            logger.info(f"Class distribution: {dict(zip(unique, counts))}")
            logger.info(f"Class weights: {class_weight}")
            
            # Train model
            history = self.model.fit(
                X_arr, y_arr,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                class_weight=class_weight,
                verbose=0
            )
            
            logger.info("Local training completed")
            return history
            
        except Exception as e:
            logger.error(f"Error in local training: {str(e)}")
            raise

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make binary predictions"""
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Get probability predictions"""
        X_arr = self._validate_input_data(X)
        pred = self.model.predict(X_arr, verbose=0)
        return np.column_stack([1 - pred, pred])

    def initialize_explainers(self, X_train: Union[np.ndarray, pd.DataFrame]):
        """Initialize LIME and SHAP explainers"""
        logger.info("Initializing explainers...")
        
        try:
            X_train_arr = self._validate_input_data(X_train)
            
            # Initialize LIME explainer
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train_arr,
                feature_names=self.feature_names,
                class_names=['Not Deceased', 'Deceased'],
                mode='classification'
            )
            
            # Initialize SHAP explainer with smaller background set
            background = shap.sample(X_train_arr, min(100, len(X_train_arr)))
            self.shap_explainer = shap.KernelExplainer(
                self.predict_proba,
                background
            )
            
            logger.info("Explainers initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing explainers: {str(e)}")
            raise

    def explain_instance(self, instance: Union[np.ndarray, pd.Series]) -> Dict:
        """Generate explanations for a single instance"""
        logger.info("Generating explanation...")
        
        if self.lime_explainer is None or self.shap_explainer is None:
            raise ValueError("Explainers not initialized. Call initialize_explainers first.")
        
        try:
            instance_arr = self._validate_input_data(instance)
            explanations = {}
            
            # Get LIME explanation
            lime_exp = self.lime_explainer.explain_instance(
                instance_arr[0],
                self.predict_proba,
                num_features=len(self.feature_names)
            )
            explanations['lime'] = lime_exp
            
            # Get SHAP explanation
            shap_values = self.shap_explainer.shap_values(instance_arr)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            explanations['shap'] = {
                'values': shap_values,
                'expected_value': (
                    self.shap_explainer.expected_value[1]
                    if isinstance(self.shap_explainer.expected_value, list)
                    else self.shap_explainer.expected_value
                )
            }
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            raise

    def get_weights(self) -> List[np.ndarray]:
        """Get current model weights"""
        return self.model.get_weights()
    
    def set_weights(self, weights: List[np.ndarray]):
        """Set model weights"""
        self.model.set_weights(weights)