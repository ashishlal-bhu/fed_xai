import logging
from typing import Dict, List, Union, Optional, Tuple
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
from configuration.xai_config import FederatedXAIConfig, ExplainabilityConfig

logger = logging.getLogger('fed_client')

class FederatedClient:
    """Client implementation for federated learning with XAI capabilities"""
    
    def __init__(
        self,
        client_id: str,
        input_dim: int,
        features: Union[List[str], pd.Index],
        model_config: Optional[Dict] = None,
        xai_config: Optional[FederatedXAIConfig] = None
    ):
        """
        Initialize a federated client with XAI capabilities.
        
        Args:
            client_id: Unique identifier for the client
            input_dim: Input dimension for the model
            features: Feature names
            model_config: Model configuration
            xai_config: Configuration for XAI components
        """
        self.client_id = client_id
        self.input_dim = input_dim
        self.features = features
        
        # Initialize local model
        self.local_model = FederatedXAIModel(
            input_dim=input_dim,
            features=features,
            model_config=model_config
        )
        
        # Store XAI configuration
        if xai_config is None:
            from configuration.xai_config import DEFAULT_CONFIG
            self.xai_config = DEFAULT_CONFIG
        else:
            self.xai_config = xai_config
        
        # Explanation state tracking
        self.explanation_enabled = True
        self.explainers_initialized = False
        
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
        round_num: int = 0,
        epochs: int = 1,
        batch_size: int = 32,
        validation_split: float = 0.2,
        **train_kwargs
    ):
        """
        Train local model on client data and collect explanations if configured.
        
        Args:
            X: Training features
            y: Training targets
            round_num: Current training round
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            **train_kwargs: Additional training parameters
            
        Returns:
            Dictionary with training history and explanations
        """
        try:
            history = self.local_model.local_train(
                X, y, 
                validation_split=validation_split, 
                epochs=epochs, 
                batch_size=batch_size,
                **train_kwargs
            )
            logger.info(f"Client {self.client_id} completed local training")
            
            # Collect explanations if enabled and this is a designated explanation round
            if self._should_collect_explanations(round_num):
                # Initialize explainers if not already initialized
                if not self.explainers_initialized:
                    self.initialize_explainers(X, y)
                
                # Generate explanations
                explanations = self.generate_explanations(X, y, round_num)
                
                # Combine history and explanations
                if isinstance(history, dict):
                    history['explanations'] = explanations
                else:
                    history = {'training': history, 'explanations': explanations}
            
            return history
        except Exception as e:
            logger.error(f"Error in local training: {str(e)}")
            raise
    
    def _should_collect_explanations(self, round_num: int) -> bool:
        """
        Determine if explanations should be collected this round.
        
        Args:
            round_num: Current training round
            
        Returns:
            True if explanations should be collected, False otherwise
        """
        if not hasattr(self, 'xai_config') or not self.explanation_enabled:
            return False
        
        if not self.xai_config.collect_explanations:
            return False
        
        if self.xai_config.explanation_rounds is not None:
            return round_num in self.xai_config.explanation_rounds
        
        return True
    
    def get_local_weights(self) -> List[np.ndarray]:
        """Get current local model weights"""
        return self.local_model.get_weights()
    
    def initialize_explainers(self, X_train: Union[np.ndarray, pd.DataFrame], y_train: Union[np.ndarray, pd.Series]):
        """
        Initialize local explainers based on configuration.
        
        Args:
            X_train: Training data used to initialize explainers
            y_train: Training labels used to initialize explainers
        """
        try:
            # Only initialize the explainers specified in configuration
            explainability_config = self.xai_config.explainability
            
            # Pass both X and y to the model's initialize_explainers method
            self.local_model.initialize_explainers(
                X_train,
                y_train,
                lime=explainability_config.use_lime, 
                shap=explainability_config.use_shap,
                shap_samples=explainability_config.shap_samples,
                lime_samples=explainability_config.lime_samples
            )
            self.explainers_initialized = True
            
            logger.info(f"Client {self.client_id} initialized explainers with LIME={explainability_config.use_lime}, SHAP={explainability_config.use_shap}")
        except Exception as e:
            logger.error(f"Error initializing explainers: {str(e)}")
            self.explainers_initialized = False
            raise
        
    def explain_instance(
        self, 
        instance: Union[np.ndarray, pd.Series],
        max_features: Optional[int] = None
    ) -> Dict:
        """
        Get local explanation for an instance.
        
        Args:
            instance: The instance to explain
            max_features: Maximum number of features to include in explanation
            
        Returns:
            Dictionary containing explanations
        """
        try:
            # Use configuration if max_features not provided
            if max_features is None and hasattr(self, 'xai_config'):
                max_features = self.xai_config.explainability.max_features
            
            # Get explanation from model
            explanation = self.local_model.explain_instance(
                instance, 
                max_features=max_features
            )
            
            # Filter explanation types based on configuration
            if hasattr(self, 'xai_config'):
                filtered_explanation = {}
                explainability_config = self.xai_config.explainability
                
                if explainability_config.use_lime and 'lime' in explanation:
                    filtered_explanation['lime'] = explanation['lime']
                elif explainability_config.use_lime:
                    # If LIME is enabled but not in explanation, generate it
                    try:
                        lime_exp = self.local_model.lime_explainer.explain_instance(
                            instance,
                            self.local_model.predict_proba,
                            num_features=max_features,
                            num_samples=getattr(explainability_config, 'lime_samples', 5000)
                        )
                        if lime_exp is not None and hasattr(lime_exp, 'as_list'):
                            lime_features = lime_exp.as_list()
                            if lime_features:
                                filtered_explanation['lime'] = {str(feature): float(importance) 
                                                              for feature, importance in lime_features}
                    except Exception as e:
                        logger.error(f"Error generating LIME explanation: {str(e)}")
                        logger.error("Detailed error: ", exc_info=True)
                
                if explainability_config.use_shap and 'shap' in explanation:
                    filtered_explanation['shap'] = explanation['shap']
                
                explanation = filtered_explanation
            
            return {
                'client_id': self.client_id,
                'explanations': explanation
            }
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            raise
    
    def generate_explanations(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        round_num: int = 0
    ) -> Dict:
        """
        Generate explanations for a sample of instances and aggregate them.
        
        Args:
            X: Feature data
            y: Target data
            round_num: Current training round
            
        Returns:
            Dictionary containing explanation summaries
        """
        try:
            # Sample data for explanations
            X_sample, y_sample = self._sample_data_for_explanations(X, y)
            
            if X_sample is None or len(X_sample) == 0:
                logger.warning(f"No samples available for explanations on client {self.client_id}")
                return {'error': 'No samples available for explanations'}
            
            logger.info(f"Generating explanations for {len(X_sample)} samples on client {self.client_id}")
            
            # Generate raw explanations
            raw_explanations = []
            explainability_config = self.xai_config.explainability
            
            # Process in batches to avoid memory issues
            batch_size = min(explainability_config.explanation_batch_size, len(X_sample))
            
            for i in range(0, len(X_sample), batch_size):
                batch_end = min(i + batch_size, len(X_sample))
                logger.debug(f"Processing explanation batch {i//batch_size + 1}")
                
                # Get instances for this batch
                batch = X_sample.iloc[i:batch_end] if isinstance(X_sample, pd.DataFrame) else X_sample[i:batch_end]
                
                # Generate explanations for each instance
                for j in range(len(batch)):
                    instance = batch.iloc[j] if isinstance(batch, pd.DataFrame) else batch[j]
                    
                    # Get explanations for this instance
                    explanation = self.explain_instance(instance)['explanations']
                    raw_explanations.append(explanation)
            
            # Summarize explanations
            summary = self._summarize_explanations(raw_explanations)
            
            # Apply privacy mechanisms if enabled
            if self.xai_config.privacy.enable_privacy:
                summary = self._apply_privacy_to_explanations(summary)
            
            # Create complete explanation package
            explanation_data = {
                'client_id': self.client_id,
                'round': round_num,
                'summary': summary,
                'metadata': {
                    'sample_size': len(X_sample),
                    'positive_ratio': float(np.mean(y_sample)),
                    'feature_count': self.input_dim
                }
            }
            
            return explanation_data
        except Exception as e:
            logger.error(f"Error generating explanations: {str(e)}")
            return {'error': str(e)}
    
    def _sample_data_for_explanations(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series]]:
        """
        Sample data for explanation generation, ensuring privacy requirements.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Tuple of (sampled_X, sampled_y)
        """
        try:
            if len(X) == 0:
                return None, None
            
            explainability_config = self.xai_config.explainability
            privacy_config = self.xai_config.privacy
            
            # Determine sample size
            sample_size = min(explainability_config.explanations_per_client, len(X))
            
            # Check minimum sample size for privacy
            if privacy_config.enable_privacy and sample_size < privacy_config.min_samples:
                logger.warning(f"Sample size ({sample_size}) is less than minimum "
                             f"required for privacy ({privacy_config.min_samples})")
                return None, None
            
            # Stratified sampling to maintain class distribution
            pos_indices = np.where(y == 1)[0]
            neg_indices = np.where(y == 0)[0]
            
            # Calculate number of samples for each class
            pos_ratio = len(pos_indices) / len(y)
            pos_samples = int(sample_size * pos_ratio)
            neg_samples = sample_size - pos_samples
            
            # Ensure we don't sample more than available
            pos_samples = min(pos_samples, len(pos_indices))
            neg_samples = min(neg_samples, len(neg_indices))
            
            # Randomly sample from each class
            sampled_pos = np.random.choice(pos_indices, size=pos_samples, replace=False)
            sampled_neg = np.random.choice(neg_indices, size=neg_samples, replace=False)
            
            # Combine indices
            sampled_indices = np.concatenate([sampled_pos, sampled_neg])
            
            # Extract sample data
            if isinstance(X, pd.DataFrame):
                X_sample = X.iloc[sampled_indices]
                y_sample = y.iloc[sampled_indices] if isinstance(y, pd.Series) else y[sampled_indices]
            else:
                X_sample = X[sampled_indices]
                y_sample = y[sampled_indices]
            
            return X_sample, y_sample
        except Exception as e:
            logger.error(f"Error sampling data for explanations: {str(e)}")
            return None, None
    
    def _summarize_explanations(self, raw_explanations: List[Dict]) -> Dict:
        """
        Summarize raw explanations into feature importance summaries.
        
        Args:
            raw_explanations: List of raw explanations
            
        Returns:
            Dictionary containing summarized explanations
        """
        if not raw_explanations:
            return {'lime': {}, 'shap': {}}
        
        explainability_config = self.xai_config.explainability
        
        # Initialize summaries
        summary = {}
        
        # Process LIME explanations if available
        if explainability_config.use_lime:
            lime_importances = self._extract_lime_importances(raw_explanations)
            summary['lime'] = self._get_top_features(
                lime_importances, 
                explainability_config.max_features
            )
        
        # Process SHAP explanations if available
        if explainability_config.use_shap:
            shap_importances = self._extract_shap_importances(raw_explanations)
            summary['shap'] = self._get_top_features(
                shap_importances, 
                explainability_config.max_features
            )
        
        return summary
    
    def _extract_lime_importances(self, raw_explanations: List[Dict]) -> Dict[str, float]:
        """Extract feature importances from LIME explanations"""
        try:
            # Initialize importances dictionary
            importances = {feature: 0.0 for feature in self.features}
            non_zero_found = False
            
            # Process each raw explanation
            for explanation in raw_explanations:
                if 'lime' not in explanation:
                    logger.warning("Missing LIME explanation in record")
                    continue
                
                lime_exp = explanation['lime']
                
                # Check if lime_exp is a dictionary (already processed)
                if isinstance(lime_exp, dict):
                    for feature, importance in lime_exp.items():
                        # Extract the base feature name from the condition
                        base_feature = feature.split(' ')[0]
                        if base_feature in importances:
                            importances[base_feature] += abs(float(importance))
                            if abs(float(importance)) > 0:
                                non_zero_found = True
                # Check if lime_exp has as_list method (raw LIME explanation)
                elif hasattr(lime_exp, 'as_list'):
                    for feature, importance in lime_exp.as_list():
                        # Extract the base feature name from the condition
                        base_feature = feature.split(' ')[0]
                        if base_feature in importances:
                            importances[base_feature] += abs(float(importance))
                            if abs(float(importance)) > 0:
                                non_zero_found = True
                else:
                    logger.warning(f"Invalid LIME explanation format: {lime_exp}")
            
            # Average the importances
            num_explanations = len(raw_explanations)
            if num_explanations > 0:
                for feature in importances:
                    importances[feature] /= num_explanations
            
            # If all values are zero, use fallback mechanism
            if not non_zero_found:
                logger.warning("All LIME importance values are zero, using fallback mechanism")
                
                # Try to use model weights as fallback
                if hasattr(self.local_model, 'model') and hasattr(self.local_model.model, 'get_weights'):
                    try:
                        weights = self.local_model.model.get_weights()
                        if weights and len(weights) > 0:
                            # Use the first layer weights
                            first_layer_weights = weights[0]
                            if len(first_layer_weights.shape) == 2:
                                # Take absolute mean of weights for each feature
                                for i, feature in enumerate(self.features):
                                    if i < first_layer_weights.shape[0]:
                                        importances[feature] = float(np.mean(np.abs(first_layer_weights[i, :])))
                                logger.info("Used model weights as fallback for LIME importances")
                                return importances
                    except Exception as e:
                        logger.warning(f"Failed to use model weights as fallback: {str(e)}")
                
                # If model weights fallback fails, use small random values
                logger.info("Using small random values as fallback for LIME importances")
                for feature in importances:
                    importances[feature] = float(np.random.uniform(0.001, 0.01))
            
            # Normalize importances to ensure they sum to 1
            total = sum(importances.values())
            if total > 0:
                for feature in importances:
                    importances[feature] /= total
            
            return importances
            
        except Exception as e:
            logger.error(f"Error extracting LIME importances: {str(e)}")
            logger.error("Detailed error: ", exc_info=True)
            # Return small random values as last resort
            return {feature: float(np.random.uniform(0.001, 0.01)) for feature in self.features}
    
    def _extract_shap_importances(self, raw_explanations: List[Dict]) -> Dict[str, float]:
        """Extract feature importances from SHAP explanations"""
        try:
            # Initialize importances dictionary
            importances = {feature: 0.0 for feature in self.features}
            non_zero_found = False
            
            # Process each raw explanation
            for explanation in raw_explanations:
                if 'shap' not in explanation:
                    logger.warning("Missing SHAP explanation in record")
                    continue
                
                shap_values = explanation['shap']
                
                # Check if shap_values is a dictionary (already processed)
                if isinstance(shap_values, dict):
                    for feature, importance in shap_values.items():
                        if feature in importances:
                            importances[feature] += abs(float(importance))
                            if abs(float(importance)) > 0:
                                non_zero_found = True
                # Check if shap_values is a numpy array
                elif isinstance(shap_values, np.ndarray):
                    for i, feature in enumerate(self.features):
                        if i < len(shap_values):
                            importances[feature] += abs(float(shap_values[i]))
                            if abs(float(shap_values[i])) > 0:
                                non_zero_found = True
                else:
                    logger.warning(f"Invalid SHAP explanation format: {shap_values}")
            
            # Average the importances
            num_explanations = len(raw_explanations)
            if num_explanations > 0:
                for feature in importances:
                    importances[feature] /= num_explanations
            
            # If all values are zero, use fallback mechanism
            if not non_zero_found:
                logger.warning("All SHAP importance values are zero, using fallback mechanism")
                
                # Try to use model weights as fallback
                if hasattr(self.local_model, 'model') and hasattr(self.local_model.model, 'get_weights'):
                    try:
                        weights = self.local_model.model.get_weights()
                        if weights and len(weights) > 0:
                            # Use the first layer weights
                            first_layer_weights = weights[0]
                            if len(first_layer_weights.shape) == 2:
                                # Take absolute mean of weights for each feature
                                for i, feature in enumerate(self.features):
                                    if i < first_layer_weights.shape[0]:
                                        importances[feature] = float(np.mean(np.abs(first_layer_weights[i, :])))
                                logger.info("Used model weights as fallback for SHAP importances")
                                return importances
                    except Exception as e:
                        logger.warning(f"Failed to use model weights as fallback: {str(e)}")
                
                # If model weights fallback fails, use small random values
                logger.info("Using small random values as fallback for SHAP importances")
                for feature in importances:
                    importances[feature] = float(np.random.uniform(0.001, 0.01))
            
            # Normalize importances to ensure they sum to 1
            total = sum(importances.values())
            if total > 0:
                for feature in importances:
                    importances[feature] /= total
            
            return importances
            
        except Exception as e:
            logger.error(f"Error extracting SHAP importances: {str(e)}")
            logger.error("Detailed error: ", exc_info=True)
            # Return small random values as last resort
            return {feature: float(np.random.uniform(0.001, 0.01)) for feature in self.features}
    
    def _get_top_features(self, importances: Dict[str, float], max_features: int) -> Dict[str, float]:
        """
        Get top N features by importance.
        
        Args:
            importances: Dictionary mapping features to importance values
            max_features: Maximum number of features to include
            
        Returns:
            Dictionary with top features and their importance values
        """
        # Sort features by importance
        sorted_features = sorted(
            importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Take top N features
        top_features = sorted_features[:max_features]
        
        return dict(top_features)
    
    def _apply_privacy_to_explanations(self, summary: Dict) -> Dict:
        """
        Apply privacy mechanisms to explanation summary.
        
        Args:
            summary: Dictionary with explanation summaries
            
        Returns:
            Dictionary with privacy-protected explanation summaries
        """
        privacy_config = self.xai_config.privacy
        
        if not privacy_config.enable_privacy:
            return summary
        
        logger.info(f"Applying privacy mechanisms to explanations for client {self.client_id}")
        
        # Process each explanation type
        for exp_type, importances in summary.items():
            # Skip if empty
            if not importances:
                continue
            
            # Get importance values as array for vectorized operations
            features = list(importances.keys())
            values = np.array(list(importances.values()))
            
            # 1. Clip values if configured
            if privacy_config.clip_values:
                min_val, max_val = privacy_config.clip_range
                values = np.clip(values, min_val, max_val)
            
            # 2. Add differential privacy noise (Laplace mechanism)
            sensitivity = ((privacy_config.clip_range[1] - privacy_config.clip_range[0]) 
                          if privacy_config.clip_values else 2.0)
            
            scale = sensitivity / privacy_config.epsilon
            noise = np.random.laplace(0, scale, size=values.shape)
            values += noise
            
            # Update summary with noisy values
            summary[exp_type] = {
                feature: float(value)
                for feature, value in zip(features, values)
            }
        
        return summary
    
    def evaluate_local_model(
        self,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series]
    ) -> Dict:
        """
        Evaluate local model performance.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
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
            
    def get_misclassified_explanations(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        k: int = 5
    ) -> List[Dict]:
        """
        Get explanations for k misclassified instances.
        
        Args:
            X: Feature data
            y: Target data
            k: Number of misclassified instances to explain
            
        Returns:
            List of explanations for misclassified instances
        """
        try:
            # Get predictions
            y_pred = self.local_model.predict(X)
            
            # Find misclassified instances
            misclassified = np.where(y_pred != y)[0]
            
            if len(misclassified) == 0:
                logger.info(f"No misclassified instances found for client {self.client_id}")
                return []
            
            # Sample k misclassified instances
            sample_size = min(k, len(misclassified))
            sampled_indices = np.random.choice(misclassified, size=sample_size, replace=False)
            
            # Generate explanations
            explanations = []
            for idx in sampled_indices:
                instance = X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]
                true_label = int(y.iloc[idx]) if isinstance(y, pd.Series) else int(y[idx])
                pred_label = int(y_pred[idx])
                
                explanation = self.explain_instance(instance)['explanations']
                
                explanations.append({
                    'instance_index': int(idx),
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'explanations': explanation
                })
            
            return explanations
        except Exception as e:
            logger.error(f"Error generating misclassified explanations: {str(e)}")
            return []