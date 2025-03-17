# integration/xai_integration.py
"""
Integration module for connecting the configuration system with the federated learning framework.

This module provides adapter classes and functions to integrate the XAI configuration
with the existing federated learning components.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple

# Add project root to path (if needed)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from configuration.xai_config import (
    FederatedXAIConfig, 
    ExplainabilityConfig,
    PrivacyConfig, 
    AggregationConfig
)

# Set up logging
logger = logging.getLogger('xai_integration')


class ExplanationManager:
    """
    Manages the explanation process for federated learning clients.
    
    This class serves as a mediator between the FederatedClient and
    the explainability components, applying configurations and privacy controls.
    """
    
    def __init__(
        self,
        client_id: str,
        feature_names: List[str],
        xai_config: FederatedXAIConfig
    ):
        """
        Initialize explanation manager.
        
        Args:
            client_id: Identifier for the client
            feature_names: Names of model features
            xai_config: Configuration for XAI components
        """
        self.client_id = client_id
        self.feature_names = feature_names
        self.feature_count = len(feature_names)
        self.config = xai_config
        
        # Counters for tracking explanations
        self.explanation_counts = {
            'lime': 0,
            'shap': 0,
            'total': 0
        }
        
        logger.info(f"Initialized explanation manager for client {client_id}")
    
    def generate_explanations(
        self,
        client,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        round_num: int = 0
    ) -> Dict[str, Any]:
        """
        Generate explanations for client data.
        
        Args:
            client: FederatedClient instance
            X: Input features
            y: Target values
            round_num: Current training round number
            
        Returns:
            Dictionary containing explanation data
        """
        # Check if we should collect explanations this round
        if not self._should_collect_explanations(round_num):
            logger.info(f"Skipping explanation collection for round {round_num}")
            return None
        
        logger.info(f"Generating explanations for client {self.client_id} (round {round_num})")
        
        # Sample data for explanations
        X_sample, y_sample = self._sample_data_for_explanations(X, y)
        
        # Generate raw explanations
        raw_explanations = self._generate_raw_explanations(client, X_sample)
        
        # Process and summarize explanations
        summary = self._summarize_explanations(raw_explanations)
        
        # Apply privacy mechanisms if enabled
        if self.config.privacy.enable_privacy:
            summary = self._apply_privacy_mechanisms(summary)
        
        # Update counters
        self.explanation_counts['total'] += len(raw_explanations)
        
        # Create explanation package
        explanation_data = {
            'client_id': self.client_id,
            'round': round_num,
            'summary': summary,
            'metadata': {
                'sample_size': len(X_sample),
                'positive_ratio': float(np.mean(y_sample)),
                'explanation_count': len(raw_explanations),
                'feature_count': self.feature_count
            }
        }
        
        # Save explanations if configured
        if self.config.save_explanations:
            self._save_explanations(explanation_data, round_num)
        
        return explanation_data
    
    def _should_collect_explanations(self, round_num: int) -> bool:
        """Determine if explanations should be collected this round."""
        if not self.config.collect_explanations:
            return False
        
        if self.config.explanation_rounds is not None:
            return round_num in self.config.explanation_rounds
        
        return True
    
    def _sample_data_for_explanations(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series]
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.Series]]:
        """Sample data for explanation generation."""
        sample_size = min(
            self.config.explainability.explanations_per_client,
            len(X)
        )
        
        # Check minimum sample size for privacy
        if sample_size < self.config.privacy.min_samples:
            logger.warning(f"Sample size ({sample_size}) is less than minimum "
                         f"required for privacy ({self.config.privacy.min_samples})")
            
            if self.config.privacy.enable_privacy:
                logger.info("Disabling explanations due to privacy constraints")
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
    
    def _generate_raw_explanations(
        self,
        client,
        X_sample: Union[np.ndarray, pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """Generate raw explanations for sampled data."""
        raw_explanations = []
        explainability_config = self.config.explainability
        
        # Process in batches to avoid memory issues
        batch_size = explainability_config.explanation_batch_size
        
        for i in range(0, len(X_sample), batch_size):
            batch_end = min(i + batch_size, len(X_sample))
            logger.debug(f"Processing explanation batch {i//batch_size + 1}")
            
            # Get instances for this batch
            batch = X_sample.iloc[i:batch_end] if isinstance(X_sample, pd.DataFrame) else X_sample[i:batch_end]
            
            # Generate explanations for each instance
            for j in range(len(batch)):
                instance = batch.iloc[j] if isinstance(batch, pd.DataFrame) else batch[j]
                
                # Call client's explain_instance method
                explanation = client.explain_instance(instance)
                
                # Filter explanations based on configuration
                filtered_explanation = {}
                
                if explainability_config.use_lime and 'lime' in explanation:
                    filtered_explanation['lime'] = explanation['lime']
                    self.explanation_counts['lime'] += 1
                
                if explainability_config.use_shap and 'shap' in explanation:
                    filtered_explanation['shap'] = explanation['shap']
                    self.explanation_counts['shap'] += 1
                
                raw_explanations.append(filtered_explanation)
        
        logger.info(f"Generated {len(raw_explanations)} raw explanations")
        return raw_explanations
    
    def _summarize_explanations(self, raw_explanations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize raw explanations into aggregated feature importance."""
        if not raw_explanations:
            return {'lime': {}, 'shap': {}}
        
        explainability_config = self.config.explainability
        max_features = explainability_config.max_features
        
        # Initialize summary structures
        summary = {}
        
        # Summarize LIME explanations if available
        if explainability_config.use_lime and 'lime' in raw_explanations[0]:
            lime_importances = self._extract_lime_importances(raw_explanations)
            summary['lime'] = self._get_top_features(lime_importances, max_features)
        
        # Summarize SHAP explanations if available
        if explainability_config.use_shap and 'shap' in raw_explanations[0]:
            shap_importances = self._extract_shap_importances(raw_explanations)
            summary['shap'] = self._get_top_features(shap_importances, max_features)
        
        return summary
    
    def _extract_lime_importances(self, raw_explanations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract and aggregate LIME feature importances."""
        # Initialize with zeros for all features
        importances = {feature: 0.0 for feature in self.feature_names}
        count = 0
        
        for explanation in raw_explanations:
            if 'lime' not in explanation:
                continue
                
            lime_exp = explanation['lime']
            features = lime_exp.as_list()
            
            for feature, importance in features:
                if feature in importances:
                    importances[feature] += abs(importance)
                    count += 1
        
        # Average the importances
        if count > 0:
            for feature in importances:
                importances[feature] /= count
        
        return importances
    
    def _extract_shap_importances(self, raw_explanations: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract and aggregate SHAP feature importances."""
        # Initialize with zeros for all features
        importances = {feature: 0.0 for feature in self.feature_names}
        count = 0
        
        for explanation in raw_explanations:
            if 'shap' not in explanation:
                continue
                
            shap_values = explanation['shap']['values']
            
            # Ensure we're working with a 2D array
            if len(shap_values.shape) > 2:
                shap_values = shap_values[0]  # Use first class for multi-class
            
            # For binary classification, sometimes values are [n_samples, 2]
            if len(shap_values.shape) == 2 and shap_values.shape[1] > 1:
                # Use positive class (class 1) values
                values = shap_values[:, 1]
            else:
                values = shap_values
            
            for i, feature in enumerate(self.feature_names):
                if i < len(values):
                    importances[feature] += abs(values[i])
                    count += 1
        
        # Average the importances
        if count > 0:
            for feature in importances:
                importances[feature] /= (count / len(self.feature_names))
        
        return importances
    
    def _get_top_features(self, importances: Dict[str, float], max_features: int) -> Dict[str, float]:
        """Get top N features by importance."""
        # Sort features by importance
        sorted_features = sorted(
            importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        # Take top N features
        top_features = sorted_features[:max_features]
        
        return dict(top_features)
    
    def _apply_privacy_mechanisms(self, summary: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Apply privacy mechanisms to explanation summary."""
        privacy_config = self.config.privacy
        
        if not privacy_config.enable_privacy:
            return summary
        
        logger.info("Applying privacy mechanisms to explanations")
        
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
            sensitivity = (privacy_config.clip_range[1] - privacy_config.clip_range[0]) 
            if privacy_config.clip_values else 2.0
            
            scale = sensitivity / privacy_config.epsilon
            noise = np.random.laplace(0, scale, size=values.shape)
            values += noise
            
            # Update summary with noisy values
            summary[exp_type] = {
                feature: float(value)
                for feature, value in zip(features, values)
            }
        
        return summary
    
    def _save_explanations(self, explanation_data: Dict[str, Any], round_num: int):
        """Save explanation data to disk."""
        import json
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(self.config.explanations_path, exist_ok=True)
        
        # Create filename
        filename = f"client_{self.client_id}_round_{round_num}_explanations.json"
        filepath = os.path.join(self.config.explanations_path, filename)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(explanation_data, f, indent=2)
        
        logger.info(f"Saved explanations to {filepath}")


class GlobalExplanationAggregator:
    """
    Aggregates explanations from multiple clients into global explanations.
    
    This class implements the server-side aggregation of explanations
    according to the configured aggregation strategy.
    """
    
    def __init__(
        self,
        feature_names: List[str],
        xai_config: FederatedXAIConfig
    ):
        """
        Initialize global explanation aggregator.
        
        Args:
            feature_names: List of feature names
            xai_config: Configuration for XAI components
        """
        self.feature_names = feature_names
        self.feature_count = len(feature_names)
        self.config = xai_config
        self.aggregation_config = xai_config.aggregation
        
        # Initialize storage for explanations
        self.round_explanations = {}  # Round -> client explanations
        self.global_explanations = {}  # Round -> global explanation
        self.feature_importance_history = {
            'lime': {},
            'shap': {}
        }  # Round -> feature importance
        
        logger.info("Initialized global explanation aggregator")
    
    def add_client_explanations(
        self,
        client_id: str,
        explanation_data: Dict[str, Any],
        round_num: int
    ):
        """
        Add client explanations for the current round.
        
        Args:
            client_id: Client identifier
            explanation_data: Explanation data from client
            round_num: Current training round
        """
        if round_num not in self.round_explanations:
            self.round_explanations[round_num] = {}
        
        self.round_explanations[round_num][client_id] = explanation_data
        
        logger.info(f"Added explanations from client {client_id} for round {round_num}")
    
    def aggregate_round_explanations(self, round_num: int) -> Dict[str, Any]:
        """
        Aggregate explanations for the current round.
        
        Args:
            round_num: Current training round
            
        Returns:
            Dictionary containing aggregated explanations
        """
        if round_num not in self.round_explanations:
            logger.warning(f"No explanations found for round {round_num}")
            return None
        
        client_explanations = self.round_explanations[round_num]
        
        # Check if we have enough clients
        if len(client_explanations) < self.aggregation_config.min_clients_per_round:
            logger.warning(f"Not enough clients for aggregation in round {round_num}. "
                         f"Have {len(client_explanations)}, need "
                         f"{self.aggregation_config.min_clients_per_round}")
            return None
        
        logger.info(f"Aggregating explanations from {len(client_explanations)} clients for round {round_num}")
        
        # Extract summaries from each client
        client_summaries = {}
        for client_id, explanation_data in client_explanations.items():
            if 'summary' in explanation_data:
                client_summaries[client_id] = explanation_data['summary']
        
        # Aggregate explanations by type
        aggregated = {}
        for exp_type in ['lime', 'shap']:
            # Skip if not all clients have this explanation type
            if not all(exp_type in summary for summary in client_summaries.values()):
                logger.info(f"Skipping {exp_type} aggregation - not all clients have this type")
                continue
            
            # Aggregate this explanation type
            aggregated[exp_type] = self._aggregate_explanations(
                {client_id: summary[exp_type] for client_id, summary in client_summaries.items()},
                exp_type
            )
            
            # Store in history
            self.feature_importance_history[exp_type][round_num] = aggregated[exp_type]
        
        # Store global explanation for this round
        self.global_explanations[round_num] = {
            'round': round_num,
            'explanations': aggregated,
            'client_count': len(client_explanations),
            'client_ids': list(client_explanations.keys())
        }
        
        return self.global_explanations[round_num]
    
    def _aggregate_explanations(
        self,
        client_importances: Dict[str, Dict[str, float]],
        explanation_type: str
    ) -> Dict[str, float]:
        """
        Aggregate feature importances from multiple clients.
        
        Args:
            client_importances: Dictionary mapping client IDs to feature importances
            explanation_type: Type of explanation ('lime' or 'shap')
            
        Returns:
            Dictionary mapping features to aggregated importance values
        """
        # Get aggregation method
        method = self.aggregation_config.aggregation_method
        
        # Initialize with zeros for all features
        aggregated = {feature: 0.0 for feature in self.feature_names}
        
        # Count how many clients included each feature
        feature_counts = {feature: 0 for feature in self.feature_names}
        
        # Collect importance values for each feature across clients
        feature_values = {feature: [] for feature in self.feature_names}
        
        # Extract values from clients
        for client_id, importances in client_importances.items():
            for feature, importance in importances.items():
                if feature in aggregated:
                    feature_values[feature].append(importance)
                    feature_counts[feature] += 1
        
        # Apply outlier removal if configured
        if self.aggregation_config.discard_outliers:
            for feature in feature_values:
                if len(feature_values[feature]) > 3:  # Need at least 4 points to detect outliers
                    feature_values[feature] = self._remove_outliers(
                        feature_values[feature],
                        self.aggregation_config.outlier_threshold
                    )
        
        # Apply the selected aggregation method
        if method == 'simple_average':
            # Simple average of available values
            for feature in self.feature_names:
                values = feature_values[feature]
                if values:
                    aggregated[feature] = sum(values) / len(values)
        
        elif method == 'weighted_average':
            # Weighted average (could use client metrics, data size, etc.)
            # Here we use a simple count-based weighting
            for feature in self.feature_names:
                values = feature_values[feature]
                if values:
                    aggregated[feature] = sum(values) / len(values)
        
        elif method == 'median':
            # Median of values (robust to outliers)
            for feature in self.feature_names:
                values = feature_values[feature]
                if values:
                    aggregated[feature] = np.median(values)
        
        elif method == 'robust_average':
            # Trimmed mean (remove top and bottom 10%)
            for feature in self.feature_names:
                values = feature_values[feature]
                if len(values) > 4:  # Need at least 5 points for trimming
                    sorted_values = sorted(values)
                    trim_size = max(1, int(len(values) * 0.1))
                    trimmed = sorted_values[trim_size:-trim_size]
                    aggregated[feature] = sum(trimmed) / len(trimmed)
                elif values:
                    aggregated[feature] = sum(values) / len(values)
        
        # Apply temporal decay to combine with previous rounds
        latest_round = max(self.feature_importance_history[explanation_type].keys()) if self.feature_importance_history[explanation_type] else None
        
        if latest_round is not None:
            previous = self.feature_importance_history[explanation_type][latest_round]
            decay = self.aggregation_config.temporal_decay
            
            for feature in aggregated:
                if feature in previous:
                    aggregated[feature] = (decay * aggregated[feature] + 
                                          (1 - decay) * previous[feature])
        
        return aggregated
    
    def _remove_outliers(self, values: List[float], threshold: float) -> List[float]:
        """Remove outliers based on z-score."""
        values_array = np.array(values)
        mean = np.mean(values_array)
        std = np.std(values_array)
        
        if std == 0:
            return values
        
        z_scores = np.abs((values_array - mean) / std)
        return values_array[z_scores < threshold].tolist()
    
    def get_global_explanation(self, round_num: Optional[int] = None) -> Dict[str, Any]:
        """
        Get global explanation for a specific round.
        
        Args:
            round_num: Round number (None = latest round)
            
        Returns:
            Global explanation data
        """
        if not self.global_explanations:
            return None
        
        if round_num is None:
            round_num = max(self.global_explanations.keys())
        
        return self.global_explanations.get(round_num)
    
    def get_feature_importance_history(self, explanation_type: str = 'lime') -> Dict[int, Dict[str, float]]:
        """
        Get history of feature importance across rounds.
        
        Args:
            explanation_type: Type of explanation ('lime' or 'shap')
            
        Returns:
            Dictionary mapping rounds to feature importance
        """
        return self.feature_importance_history.get(explanation_type, {})
    
    def get_client_agreement(self, round_num: int) -> Dict[str, float]:
        """
        Calculate agreement between clients on feature importance.
        
        Args:
            round_num: Round number
            
        Returns:
            Dictionary mapping client pairs to agreement scores
        """
        if round_num not in self.round_explanations:
            return {}
        
        client_explanations = self.round_explanations[round_num]
        
        # Extract LIME explanations (could also use SHAP)
        client_importances = {}
        for client_id, explanation_data in client_explanations.items():
            if 'summary' in explanation_data and 'lime' in explanation_data['summary']:
                client_importances[client_id] = explanation_data['summary']['lime']
        
        # Calculate agreement between pairs of clients
        agreement = {}
        client_ids = list(client_importances.keys())
        
        for i in range(len(client_ids)):
            for j in range(i+1, len(client_ids)):
                client_a = client_ids[i]
                client_b = client_ids[j]
                
                agreement[f"{client_a}_vs_{client_b}"] = self._calculate_agreement(
                    client_importances[client_a],
                    client_importances[client_b]
                )
        
        return agreement
    
    def _calculate_agreement(
        self,
        importance_a: Dict[str, float],
        importance_b: Dict[str, float]
    ) -> float:
        """
        Calculate agreement score between two sets of feature importances.
        
        Uses Rank Biased Overlap (RBO) to measure similarity.
        
        Args:
            importance_a: Feature importances from first client
            importance_b: Feature importances from second client
            
        Returns:
            Agreement score (0-1)
        """
        # Get feature rankings
        ranked_a = sorted(importance_a.keys(), 
                         key=lambda f: abs(importance_a.get(f, 0)), 
                         reverse=True)
        ranked_b = sorted(importance_b.keys(), 
                         key=lambda f: abs(importance_b.get(f, 0)), 
                         reverse=True)
        
        # Calculate Rank-Biased Overlap (p=0.9)
        p = 0.9
        depth = min(len(ranked_a), len(ranked_b))
        
        # Short circuit for edge cases
        if depth == 0:
            return 0.0
        if depth == 1:
            return 1.0 if ranked_a[0] == ranked_b[0] else 0.0
        
        # Calculate overlap at each depth
        score = 0.0
        cumulative_weight = 0.0
        
        for d in range(1, depth + 1):
            # Get top d elements from each ranking
            set_a = set(ranked_a[:d])
            set_b = set(ranked_b[:d])
            
            # Calculate overlap
            overlap = len(set_a.intersection(set_b)) / d
            
            # Calculate weight for this depth
            weight = (1 - p) * (p ** (d - 1))
            cumulative_weight += weight
            
            # Update score
            score += weight * overlap
        
        # Normalize by cumulative weight
        if cumulative_weight > 0:
            score /= cumulative_weight
        
        return score


# Helper functions to integrate with existing code

def update_orchestrator_for_xai(federated_orchestrator, xai_config=None):
    """
    Update FederatedOrchestrator for XAI capabilities.
    
    This function modifies an existing FederatedOrchestrator instance
    to incorporate explainability features.
    
    Args:
        federated_orchestrator: Existing FederatedOrchestrator instance
        xai_config: XAI configuration (None = use defaults)
        
    Returns:
        Updated FederatedOrchestrator instance
    """
    # Use default config if none provided
    if xai_config is None:
        from configuration.xai_config import DEFAULT_CONFIG
        xai_config = DEFAULT_CONFIG
    
    # Add XAI configuration to orchestrator
    federated_orchestrator.xai_config = xai_config
    
    # Initialize global explanation aggregator
    federated_orchestrator.explanation_aggregator = GlobalExplanationAggregator(
        feature_names=federated_orchestrator.server.features,
        xai_config=xai_config
    )
    
    # Initialize explanation managers for clients
    for client_id, client_info in federated_orchestrator.clients.items():
        client = client_info['client']
        
        # Add explanation manager to client
        client.explanation_manager = ExplanationManager(
            client_id=client_id,
            feature_names=federated_orchestrator.server.features,
            xai_config=xai_config
        )
    
    # Extend orchestrator's train_federated method with XAI capabilities
    original_train_method = federated_orchestrator.train_federated
    
    def extended_train_federated(*args, **kwargs):
        """Extended training method with XAI capabilities."""
        # Call original method
        training_history = original_train_method(*args, **kwargs)
        
        # Add explanations field if not present
        if 'explanations' not in training_history:
            training_history['explanations'] = {}
        
        return training_history
    
    # Replace train_federated method
    federated_orchestrator.original_train_federated = original_train_method
    federated_orchestrator.train_federated = extended_train_federated
    
    logger.info("Updated FederatedOrchestrator for XAI capabilities")
    
    return federated_orchestrator


def patch_client_for_xai(client, xai_config=None):
    """
    Patch FederatedClient for XAI capabilities.
    
    This function extends an existing FederatedClient with methods
    to generate and manage explanations.
    
    Args:
        client: Existing FederatedClient instance
        xai_config: XAI configuration (None = use defaults)
        
    Returns:
        Updated FederatedClient instance
    """
    # Use default config if none provided
    if xai_config is None:
        from configuration.xai_config import DEFAULT_CONFIG
        xai_config = DEFAULT_CONFIG
    
    # Add XAI configuration to client
    client.xai_config = xai_config
    
    # Add explanation manager if not already present
    if not hasattr(client, 'explanation_manager'):
        client.explanation_manager = ExplanationManager(
            client_id=client.client_id,
            feature_names=client.features,
            xai_config=xai_config
        )
    
    # Extend client's train_local_model method to collect explanations
    original_train_method = client.train_local_model
    
    def extended_train_local_model(X, y, **train_kwargs):
        """Extended training method with explanation collection."""
        # Get round number from kwargs if available
        round_num = train_kwargs.pop('round_num', 0)
        
        # Call original method
        history = original_train_method(X, y, **train_kwargs)
        
        # Generate explanations if configured
        if hasattr(client, 'explanation_manager') and client.xai_config.collect_explanations:
            explanations = client.explanation_manager.generate_explanations(
                client, X, y, round_num
            )
            
            # Add explanations to history
            if history is not None and explanations is not None:
                if not isinstance(history, dict):
                    history = {'original': history}
                history['explanations'] = explanations
        
        return history
    
    # Replace train_local_model method
    client.original_train_local_model = original_train_method
    client.train_local_model = extended_train_local_model
    
    logger.info(f"Patched FederatedClient {client.client_id} for XAI capabilities")
    
    return client


def initialize_xai_components(orchestrator, xai_config=None):
    """
    Initialize all XAI components for a federated learning setup.
    
    This function updates all necessary components to enable explainability
    features in the federated learning system.
    
    Args:
        orchestrator: FederatedOrchestrator instance
        xai_config: XAI configuration (None = use defaults)
        
    Returns:
        Updated orchestrator
    """
    # Use default config if none provided
    if xai_config is None:
        from configuration.xai_config import DEFAULT_CONFIG
        xai_config = DEFAULT_CONFIG
    
    # Update orchestrator
    update_orchestrator_for_xai(orchestrator, xai_config)
    
    # Patch all clients
    for client_id, client_info in orchestrator.clients.items():
        client = client_info['client']
        patch_client_for_xai(client, xai_config)
    
    logger.info("Initialized all XAI components")
    
    return orchestrator