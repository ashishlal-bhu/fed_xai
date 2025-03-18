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
from typing import Dict, List, Tuple, Optional, Union, Any
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
        
        # Track initialization state
        self.lime_initialized = False
        self.shap_initialized = False
        
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

    def initialize_explainers(
    self, 
    X_train: Union[np.ndarray, pd.DataFrame],
    lime: bool = True,
    shap: bool = True,
    lime_samples: int = 5000,
    shap_samples: int = 100,
    **kwargs
    ):
        """
        Initialize LIME and SHAP explainers with customizable options.
    
        Args:
         X_train: Training data for explainer initialization
            lime: Whether to initialize LIME explainer
            shap: Whether to initialize SHAP explainer
            lime_samples: Number of samples for LIME training
         shap_samples: Number of samples for SHAP background
            **kwargs: Additional explainer parameters
        """
        logger.info("Initializing explainers...")
        try:
            X_train_arr = self._validate_input_data(X_train)
            
            # Initialize LIME explainer if requested
            if lime:
                # Import lime and lime.lime_tabular here to ensure we're using the actual module
                import lime
                import lime.lime_tabular
                
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train_arr,
                    feature_names=self.feature_names,
                    class_names=['Not Deceased', 'Deceased'],
                    sample_around_instance=True,
                    mode='classification',
                    training_labels=None,  # No need for labels for explanation
                    random_state=42,  # For reproducibility
                    verbose=False
                )
                self.lime_initialized = True
                logger.info("LIME explainer initialized")
            
            # Initialize SHAP explainer with smaller background set
            if shap:
                # Import shap here to ensure we're using the actual module
                import shap
                
                # Use specified sample size or default
                background_size = min(shap_samples, len(X_train_arr))
                background = shap.sample(X_train_arr, background_size)
                
                self.shap_explainer = shap.KernelExplainer(
                    self.predict_proba,
                    background
                )
                self.shap_initialized = True
                logger.info(f"SHAP explainer initialized with {background_size} background samples")
            
            logger.info("Explainer initialization completed")
            
        except Exception as e:
            logger.error(f"Error initializing explainers: {str(e)}")
            raise

    def explain_instance(
        self, 
        instance: Union[np.ndarray, pd.Series],
        max_features: Optional[int] = None,
        lime_kwargs: Optional[Dict] = None,
        shap_kwargs: Optional[Dict] = None
    ) -> Dict:
        """
        Generate explanations for a single instance with customizable options.
        
        Args:
            instance: The instance to explain
            max_features: Maximum number of features to include in explanation
            lime_kwargs: Additional parameters for LIME explanation
            shap_kwargs: Additional parameters for SHAP explanation
            
        Returns:
            Dictionary containing explanations
        """
        logger.info("Generating explanation...")
        
        # Apply default values
        if lime_kwargs is None:
            lime_kwargs = {}
        if shap_kwargs is None:
            shap_kwargs = {}
        
        # Set default max_features if not provided
        if max_features is None:
            max_features = 10  # Default to 10 features
        
        try:
            instance_arr = self._validate_input_data(instance)
            explanations = {}
            
            # Get LIME explanation if initialized
            if self.lime_initialized and self.lime_explainer is not None:
                lime_exp = self.lime_explainer.explain_instance(
                    instance_arr[0],
                    self.predict_proba,
                    num_features=max_features,
                    **lime_kwargs
                )
                explanations['lime'] = lime_exp
            
            # Get SHAP explanation if initialized
            if self.shap_initialized and self.shap_explainer is not None:
                # Set reasonable defaults for SHAP
                default_shap_kwargs = {
                    'nsamples': 'auto',  # Auto-determine sample count
                    'l1_reg': 'aic'      # Use AIC for feature selection
                }
                
                # Override defaults with provided kwargs
                for key, value in default_shap_kwargs.items():
                    if key not in shap_kwargs:
                        shap_kwargs[key] = value
                
                # Generate SHAP values
                shap_values = self.shap_explainer.shap_values(
                    instance_arr,
                    **shap_kwargs
                )
                
                # Handle different return formats
                if isinstance(shap_values, list):
                    # For binary classification, sometimes shap_values is a list with
                    # values for each class. We typically want class 1 (positive class)
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

    def get_explanations_summary(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        sample_size: int = 10,
        max_features: int = 10
    ) -> Dict[str, Any]:
        """
        Generate summary of explanations across multiple instances.
        
        Args:
            X: Instances to explain
            sample_size: Number of instances to sample
            max_features: Maximum features per explanation
            
        Returns:
            Dictionary with explanation summaries
        """
        logger.info(f"Generating explanation summary for {sample_size} instances...")
        
        try:
            # Validate input
            X_arr = self._validate_input_data(X)
            
            # Sample instances
            if len(X_arr) > sample_size:
                indices = np.random.choice(len(X_arr), sample_size, replace=False)
                X_sample = X_arr[indices]
            else:
                X_sample = X_arr
            
            # Initialize summary structures
            lime_importances = {feature: [] for feature in self.feature_names}
            shap_importances = {feature: [] for feature in self.feature_names}
            
            # Generate explanations for each instance
            for i in range(len(X_sample)):
                instance = X_sample[i]
                
                # Get explanations
                explanation = self.explain_instance(
                    instance,
                    max_features=max_features
                )
                
                # Extract LIME importances
                if 'lime' in explanation:
                    lime_exp = explanation['lime']
                    for feature, importance in lime_exp.as_list():
                        if feature in lime_importances:
                            lime_importances[feature].append(abs(importance))
                
                # Extract SHAP importances
                if 'shap' in explanation:
                    shap_values = explanation['shap']['values']
                    
                    # Ensure we're working with a flat array
                    if len(np.array(shap_values).shape) > 1:
                        shap_values = shap_values[0]
                    
                    for i, feature in enumerate(self.feature_names):
                        if i < len(shap_values):
                            shap_importances[feature].append(abs(shap_values[i]))
            
            # Compute average importance for each feature
            average_lime = {
                feature: np.mean(values) if values else 0.0
                for feature, values in lime_importances.items()
            }
            
            average_shap = {
                feature: np.mean(values) if values else 0.0
                for feature, values in shap_importances.items()
            }
            
            # Get top features
            top_lime = dict(sorted(
                average_lime.items(),
                key=lambda x: x[1],
                reverse=True
            )[:max_features])
            
            top_shap = dict(sorted(
                average_shap.items(),
                key=lambda x: x[1],
                reverse=True
            )[:max_features])
            
            return {
                'lime': top_lime,
                'shap': top_shap,
                'metadata': {
                    'sample_size': len(X_sample),
                    'max_features': max_features
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating explanation summary: {str(e)}")
            raise
    
    def get_feature_importance(
        self, 
        X: Union[np.ndarray, pd.DataFrame],
        method: str = 'lime',
        sample_size: int = 100
    ) -> Dict[str, float]:
        """
        Get global feature importance using sampling approach.
        
        Args:
            X: Dataset to compute importance from
            method: Explanation method ('lime' or 'shap')
            sample_size: Number of instances to sample
            
        Returns:
            Dictionary mapping features to importance scores
        """
        logger.info(f"Calculating global feature importance using {method}...")
        
        try:
            # Get explanation summary
            summary = self.get_explanations_summary(
                X,
                sample_size=sample_size,
                max_features=len(self.feature_names)  # Get all features
            )
            
            # Return importance for requested method
            if method.lower() == 'lime' and 'lime' in summary:
                return summary['lime']
            
            if method.lower() == 'shap' and 'shap' in summary:
                return summary['shap']
            
            raise ValueError(f"Invalid explanation method: {method}")
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise

    def get_weights(self) -> List[np.ndarray]:
        """Get current model weights"""
        return self.model.get_weights()
    
    def set_weights(self, weights: List[np.ndarray]):
        """Set model weights"""
        self.model.set_weights(weights)
        
    def explainers_initialized(self) -> Dict[str, bool]:
        """
        Check which explainers are initialized.
        
        Returns:
            Dictionary indicating initialization status of explainers
        """
        return {
            'lime': self.lime_initialized,
            'shap': self.shap_initialized
        }
        
    def reset_explainers(self):
        """Reset all explainers to uninitialized state"""
        self.lime_explainer = None
        self.shap_explainer = None
        self.lime_initialized = False
        self.shap_initialized = False
        logger.info("Explainers have been reset")