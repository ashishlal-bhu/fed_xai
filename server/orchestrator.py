import logging
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
import sys
import os
import time
import json
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules with correct paths
from server.fed_server import FederatedServer
from models.fed_client import FederatedClient
from utils.fed_visualization import save_training_plots, save_client_contributions
from utils.fed_xai_visualization import (
    save_feature_importance_plot,
    save_client_feature_importance_comparison,
    save_feature_importance_evolution,
    save_client_agreement_heatmap,
    save_explanation_dashboard,
    create_all_visualizations
)
from configuration.xai_config import FederatedXAIConfig, DEFAULT_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('orchestrator')

class FederatedOrchestrator:
    """Orchestrator for federated learning training process with XAI capabilities"""
    
    def __init__(
        self,
        input_dim: int,
        features: Union[List[str], pd.Index],
        model_config: Optional[Dict] = None,
        aggregation_method: str = 'fedavg',
        xai_config: Optional[FederatedXAIConfig] = None
    ):
        """
        Initialize orchestrator with XAI capabilities.
        
        Args:
            input_dim: Input dimension for models
            features: Feature names
            model_config: Model configuration
            aggregation_method: Method for aggregating model weights
            xai_config: Configuration for XAI components
        """
        self.model_config = model_config or {}
        
        # Initialize server
        self.server = FederatedServer(
            input_dim=input_dim,
            features=features,
            model_config=model_config,
            aggregation_method=aggregation_method
        )
        
        # Store clients
        self.clients: Dict[str, Dict] = {}
        
        # Training configuration
        self.current_round = 0
        self.training_history = []
        
        # XAI configuration and components
        self.xai_config = xai_config if xai_config is not None else DEFAULT_CONFIG
        self.explanation_history = {}
        self.client_explanations = {}
        self.feature_importance_history = {'lime': {}, 'shap': {}}
        
        logger.info("Initialized federated orchestrator with XAI capabilities")
    
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
                model_config=self.model_config,
                xai_config=self.xai_config  # Pass XAI configuration to client
            )
            
            # Initialize explainers if XAI is enabled
            if self.xai_config.collect_explanations:
                client.initialize_explainers(X, y)  # Pass both X and y for proper initialization
            
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
        """
        Run federated training process with XAI components.
        
        Args:
            num_rounds: Number of training rounds
            local_epochs: Number of local training epochs per round
            min_clients: Minimum number of clients per round
            client_fraction: Fraction of clients to select each round
            batch_size: Batch size for training
            validation_data: Validation data for evaluating global model
            
        Returns:
            Dictionary containing training history and explanations
        """
        logger.info(f"Starting federated training for {num_rounds} rounds")
        logger.info(f"Number of clients: {len(self.clients)}")
        
        try:
            if len(self.clients) < min_clients:
                raise ValueError(f"Not enough clients. Have {len(self.clients)}, need {min_clients}")
            
            # Initialize history structure
            training_history = {
                'rounds': [],
                'global_metrics': [],
                'client_metrics': {},
                'explanations': {}  # Add explanations field
            }
            
            # Clear previous explanation history
            self.explanation_history = {}
            self.client_explanations = {}
            
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
                round_client_explanations = {}
                
                # Train selected clients
                for client_id in participating_clients:
                    client_info = self.clients[client_id]
                    client = client_info['client']  # Get the actual client object
                    X, y = client_info['data']  # Get client's data
                    
                    # Update client with global model
                    client.update_local_model(global_weights)
                    
                    # Train client locally with current round number
                    history = client.train_local_model(
                        X, y,  # Use client's own data
                        round_num=self.current_round,
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
                    
                    # Extract explanations if available
                    if isinstance(history, dict) and 'explanations' in history:
                        round_client_explanations[client_id] = history['explanations']
                        # Store client explanations for later use
                        self.client_explanations[client_id] = history['explanations']
                
                # Update global model
                self.server.update_global_model(client_weights)
                
                # Evaluate global model
                global_metrics = {}
                if validation_data is not None:
                    X_val, y_val = validation_data
                    global_metrics = self.server.evaluate_global_model(X_val, y_val)
                
                # Aggregate explanations if available
                round_explanations = None
                if round_client_explanations and self._should_collect_explanations(self.current_round):
                    round_explanations = self._aggregate_explanations(round_client_explanations)
                    # Store in explanation history
                    self.explanation_history[self.current_round] = round_explanations
                    # Add to training history
                    training_history['explanations'][self.current_round] = round_explanations
                
                # Store round results
                round_data = {
                    'round': self.current_round,
                    'client_metrics': round_client_metrics,
                    'global_metrics': global_metrics,
                    'duration': time.time() - round_start_time,
                    'explanations': round_explanations
                }
                
                training_history['rounds'].append(round_data)
                training_history['global_metrics'].append(global_metrics)
                
                # Log round summary
                self._log_round_summary(round_data)
                
                # Save intermediate plots (only for regular metrics)
                save_training_plots(training_history, is_final=False)
                save_client_contributions(training_history, is_final=False)
            
            # Save final plots for regular metrics
            save_training_plots(training_history, is_final=True)
            save_client_contributions(training_history, is_final=True)
            
            # If explanations were collected, save visualizations
            if self.explanation_history and self.xai_config.collect_explanations:
                self._save_explanation_visualizations()
            
            logger.info("Federated training completed")
            return training_history
            
        except Exception as e:
            logger.error(f"Error in federated training: {str(e)}")
            raise
    
    def _should_collect_explanations(self, round_num: int) -> bool:
        """
        Determine if explanations should be collected this round.
        
        Args:
            round_num: Current training round
            
        Returns:
            True if explanations should be collected, False otherwise
        """
        if not self.xai_config.collect_explanations:
            return False
        
        if self.xai_config.explanation_rounds is not None:
            return round_num in self.xai_config.explanation_rounds
        
        return True
    
    def _aggregate_explanations(self, client_explanations: Dict[str, Dict]) -> Dict:
        """
        Aggregate explanations from multiple clients.
        
        Args:
            client_explanations: Dictionary mapping client IDs to explanation data
            
        Returns:
            Dictionary containing aggregated explanations
        """
        logger.info(f"Aggregating explanations from {len(client_explanations)} clients")
        
        # Extract summaries from client explanations
        client_summaries = {}
        for client_id, explanation in client_explanations.items():
            if 'summary' in explanation:
                client_summaries[client_id] = explanation['summary']
        
        # Check if we have enough summaries
        if len(client_summaries) < self.xai_config.aggregation.min_clients_per_round:
            logger.warning(f"Not enough clients with explanations. Have {len(client_summaries)}, "
                         f"need {self.xai_config.aggregation.min_clients_per_round}")
            return {
                'error': 'Not enough clients with explanations',
                'client_count': len(client_summaries)
            }
        
        # Get aggregation method from configuration
        aggregation_method = self.xai_config.aggregation.aggregation_method
        
        # Initialize aggregated explanations
        aggregated = {}
        
        # Aggregate each explanation type (lime, shap)
        for exp_type in ['lime', 'shap']:
            # Skip if not all clients have this explanation type
            if not all(exp_type in summary for summary in client_summaries.values()):
                logger.info(f"Skipping {exp_type} aggregation - not all clients have this type")
                continue
            
            # Collect importance values for each feature across clients
            feature_values = {}
            for client_id, summary in client_summaries.items():
                for feature, importance in summary[exp_type].items():
                    if feature not in feature_values:
                        feature_values[feature] = []
                    
                    feature_values[feature].append(importance)
            
            # Apply outlier detection if configured
            if self.xai_config.aggregation.discard_outliers:
                for feature in feature_values:
                    if len(feature_values[feature]) > 3:  # Need at least 4 points to detect outliers
                        feature_values[feature] = self._remove_outliers(
                            feature_values[feature],
                            self.xai_config.aggregation.outlier_threshold
                        )
            
            # Apply aggregation method
            aggregated_importance = {}
            
            if aggregation_method == 'simple_average':
                # Simple average of available values
                for feature, values in feature_values.items():
                    if values:
                        aggregated_importance[feature] = sum(values) / len(values)
            
            elif aggregation_method == 'weighted_average':
                # Weighted by client contribution scores
                client_scores = {
                    client_id: self.server.clients.get(client_id, {}).get('contribution_score', 1.0)
                    for client_id in client_summaries.keys()
                }
                
                for feature, values in feature_values.items():
                    if values:
                        total_weight = sum(client_scores.values())
                        weighted_sum = sum(
                            value * client_scores[client_id] 
                            for value, client_id in zip(values, client_summaries.keys())
                        )
                        aggregated_importance[feature] = weighted_sum / total_weight
            
            elif aggregation_method == 'median':
                # Median of values (robust to outliers)
                for feature, values in feature_values.items():
                    if values:
                        aggregated_importance[feature] = np.median(values)
            
            else:  # Default to simple average
                for feature, values in feature_values.items():
                    if values:
                        aggregated_importance[feature] = sum(values) / len(values)
            
            # Store aggregated importance for this explanation type
            aggregated[exp_type] = aggregated_importance
            
            # Update feature importance history
            self.feature_importance_history[exp_type][self.current_round] = aggregated_importance
        
        # Calculate client agreement if we have multiple clients
        if len(client_summaries) > 1:
            client_agreement = self._calculate_client_agreement(client_summaries)
            aggregated['client_agreement'] = client_agreement
        
        # Add metadata
        aggregated['metadata'] = {
            'round': self.current_round,
            'client_count': len(client_summaries),
            'client_ids': list(client_summaries.keys()),
            'aggregation_method': aggregation_method
        }
        
        return aggregated
    
    def _remove_outliers(self, values: List[float], threshold: float) -> List[float]:
        """
        Remove outliers based on z-score.
        
        Args:
            values: List of values to process
            threshold: Z-score threshold for outlier detection
            
        Returns:
            List of values with outliers removed
        """
        values_array = np.array(values)
        mean = np.mean(values_array)
        std = np.std(values_array)
        
        if std == 0:
            return values
        
        z_scores = np.abs((values_array - mean) / std)
        return values_array[z_scores < threshold].tolist()
    
    def _calculate_client_agreement(self, client_summaries: Dict[str, Dict]) -> Dict[str, float]:
        """
        Calculate agreement between clients on feature importance.
        
        Args:
            client_summaries: Dictionary mapping client IDs to explanation summaries
            
        Returns:
            Dictionary mapping client pairs to agreement scores
        """
        # Use LIME explanations for agreement calculation
        client_importances = {}
        for client_id, summary in client_summaries.items():
            if 'lime' in summary:
                client_importances[client_id] = summary['lime']
        
        # Calculate agreement between pairs of clients
        agreement = {}
        client_ids = list(client_importances.keys())
        
        for i in range(len(client_ids)):
            for j in range(i+1, len(client_ids)):
                client_a = client_ids[i]
                client_b = client_ids[j]
                
                agreement[f"{client_a}_vs_{client_b}"] = self._calculate_rank_similarity(
                    client_importances[client_a],
                    client_importances[client_b]
                )
        
        return agreement
    
    def _calculate_rank_similarity(
        self, 
        importance_a: Dict[str, float],
        importance_b: Dict[str, float]
    ) -> float:
        """
        Calculate rank-based similarity between two feature importance sets.
        
        Uses Rank-Biased Overlap (RBO) to measure similarity.
        
        Args:
            importance_a: Feature importances from first client
            importance_b: Feature importances from second client
            
        Returns:
            Similarity score (0-1)
        """
        # Get feature rankings
        ranked_a = sorted(importance_a.keys(), 
                        key=lambda f: abs(importance_a.get(f, 0)), 
                        reverse=True)
        ranked_b = sorted(importance_b.keys(), 
                        key=lambda f: abs(importance_b.get(f, 0)), 
                        reverse=True)
        
        # Get common features
        common_features = set(ranked_a).intersection(set(ranked_b))
        
        # Calculate Rank-Biased Overlap (p=0.9)
        p = 0.9
        
        # Short circuit for edge cases
        if not common_features:
            return 0.0
        
        # Find ranks of common features in each ranking
        ranks_a = {feature: ranked_a.index(feature) for feature in common_features}
        ranks_b = {feature: ranked_b.index(feature) for feature in common_features}
        
        # Calculate rank correlation (Spearman's rho)
        if len(common_features) < 2:
            return 1.0 if len(common_features) == 1 else 0.0
        
        # Extract ranks as lists
        features = list(common_features)
        a_ranks = [ranks_a[f] for f in features]
        b_ranks = [ranks_b[f] for f in features]
        
        # Calculate correlation
        from scipy.stats import spearmanr
        correlation, _ = spearmanr(a_ranks, b_ranks)
        
        # Convert to similarity score (0-1)
        similarity = (correlation + 1) / 2
        
        return max(0.0, similarity)  # Ensure non-negative
    
    def _save_explanation_visualizations(self):
        """Save visualizations of explanations"""
        logger.info("Saving explanation visualizations...")
        
        # Ensure results directory exists
        results_dir = os.path.join(os.getcwd(), 'results', 'explanations')
        os.makedirs(results_dir, exist_ok=True)
        
        try:
            # Generate all visualizations
            filepaths = create_all_visualizations(
                self.explanation_history,
                self.client_explanations,
                feature_names=self.server.features,
                output_dir=results_dir
            )
            
            logger.info(f"Saved {len(filepaths)} explanation visualizations to {results_dir}")
            
            # Save explanation data as JSON for later use
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            json_filepath = os.path.join(results_dir, f"explanation_history_{timestamp}.json")
            
            with open(json_filepath, 'w') as f:
                # Convert numpy values to Python types for JSON serialization
                serializable_history = {}
                for round_num, data in self.explanation_history.items():
                    serializable_history[str(round_num)] = json.loads(
                        json.dumps(data, default=lambda o: float(o) if isinstance(o, np.float32) else o)
                    )
                
                json.dump(serializable_history, f, indent=2)
                
            logger.info(f"Saved explanation data to {json_filepath}")
            
        except Exception as e:
            logger.error(f"Error saving explanation visualizations: {str(e)}")
    
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
        
        # Log explanation summary if available
        if metrics['explanations']:
            logger.info("\nExplanation Summary:")
            
            if 'metadata' in metrics['explanations']:
                metadata = metrics['explanations']['metadata']
                logger.info(f"- Clients with explanations: {metadata['client_count']}")
                logger.info(f"- Aggregation method: {metadata['aggregation_method']}")
            
            if 'lime' in metrics['explanations']:
                lime_imp = metrics['explanations']['lime']
                top_lime = sorted(
                    lime_imp.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )[:5]  # Top 5 features
                
                logger.info("\nTop 5 features by LIME importance:")
                for feature, importance in top_lime:
                    logger.info(f"- {feature}: {importance:.4f}")
    
    def get_global_model(self) -> tf.keras.Model:
        """Get current global model"""
        return self.server.global_model
    
    def get_training_summary(self) -> Dict:
        """Get summary of entire training process"""
        return {
            'num_rounds': self.current_round,
            'num_clients': len(self.clients),
            'server_summary': self.server.get_training_summary(),
            'final_metrics': self.training_history[-1] if self.training_history else None,
            'explanations_collected': len(self.explanation_history) > 0
        }
    
    def get_explanations(self, round_num: Optional[int] = None) -> Dict:
        """
        Get explanations for a specific round.
        
        Args:
            round_num: Round number (None = latest round)
            
        Returns:
            Dictionary containing explanations
        """
        if not self.explanation_history:
            return {}
        
        if round_num is None:
            round_num = max(self.explanation_history.keys())
        
        return self.explanation_history.get(round_num, {})
    
    def get_feature_importance_history(self, explanation_type: str = 'lime') -> Dict:
        """
        Get history of feature importance across rounds.
        
        Args:
            explanation_type: Type of explanation ('lime' or 'shap')
            
        Returns:
            Dictionary mapping rounds to feature importance
        """
        return self.feature_importance_history.get(explanation_type, {})
    
    def get_client_explanations(self, client_id: Optional[str] = None) -> Dict:
        """
        Get explanations from clients.
        
        Args:
            client_id: Client ID (None = all clients)
            
        Returns:
            Dictionary containing client explanations
        """
        if client_id is None:
            return self.client_explanations
        
        return self.client_explanations.get(client_id, {})
    
    def generate_explanation_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive explanation report.
        
        Args:
            output_path: Path to save report
            
        Returns:
            Path to saved report
        """
        if not self.explanation_history:
            logger.warning("No explanations available to generate report")
            return ""
        
        # Create dashboard with all visualizations
        from datetime import datetime
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_dir = os.path.join(os.getcwd(), 'results', 'reports')
            os.makedirs(results_dir, exist_ok=True)
            output_path = os.path.join(results_dir, f"explanation_report_{timestamp}")
        
        # Save dashboard
        dashboard_path = save_explanation_dashboard(
            self.explanation_history,
            output_dir=output_path
        )
        
        logger.info(f"Generated explanation report at {dashboard_path}")
        return dashboard_path