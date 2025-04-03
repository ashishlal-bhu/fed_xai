# configuration/xai_config.py
"""
Configuration module for Federated XAI components.

This module provides configuration classes for the explainability, privacy, and 
aggregation components of the federated XAI system. These configurations 
enable flexible ablation studies and experiments.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Tuple
import json
import os

logger = logging.getLogger('fed_xai_config')

class ExplainabilityConfig:
    """Configuration for explainability methods."""
    
    def __init__(
        self,
        use_lime: bool = True,
        use_shap: bool = True,
        lime_samples: int = 1000,
        shap_samples: int = 100,
        max_features: int = 15,
        explanations_per_client: int = 10,
        explanation_batch_size: int = 100,
        **kwargs
    ):
        """
        Initialize explainability configuration.
        
        Args:
            use_lime: Whether to use LIME explanations
            use_shap: Whether to use SHAP explanations
            lime_samples: Number of samples for LIME explanations
            shap_samples: Number of samples for SHAP background dataset
            max_features: Maximum number of features to include in explanations
            explanations_per_client: Number of explanations to collect per client
            explanation_batch_size: Batch size for explanation generation
            **kwargs: Additional configuration parameters
        """
        self.use_lime = use_lime
        self.use_shap = use_shap
        self.lime_samples = lime_samples
        self.shap_samples = shap_samples
        self.max_features = max_features
        self.explanations_per_client = explanations_per_client
        self.explanation_batch_size = explanation_batch_size
        
        # Store any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Validate configuration
        self._validate()
        
        logger.info(f"Initialized explainability configuration: {self.__dict__}")
    
    def _validate(self):
        """Validate configuration parameters."""
        if not (self.use_lime or self.use_shap):
            raise ValueError("At least one explanation method must be enabled")
        
        if self.lime_samples < 100:
            logger.warning(f"Low LIME sample count ({self.lime_samples}) may affect explanation quality")
        
        if self.shap_samples < 10:
            logger.warning(f"Low SHAP sample count ({self.shap_samples}) may affect explanation quality")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExplainabilityConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


class PrivacyConfig:
    """Configuration for privacy preservation in explanations."""
    
    def __init__(
        self,
        enable_privacy: bool = False,
        epsilon: float = 1.0,
        delta: float = 0.01,
        clip_values: bool = False,
        clip_range: Tuple[float, float] = (-1.0, 1.0),
        noise_scale: float = 0.1,
        min_samples: int = 100,
        max_samples: int = 5000,
        use_secure_aggregation: bool = False,
        **kwargs
    ):
        """
        Initialize privacy configuration.
        
        Args:
            enable_privacy: Whether to enable privacy protections
            epsilon: Privacy budget (higher = less privacy)
            delta: Failure probability
            clip_values: Whether to clip explanation values
            clip_range: Range for value clipping
            noise_scale: Scale of noise to add (0.0 to 1.0)
            min_samples: Minimum number of samples for explanations
            max_samples: Maximum number of samples for explanations
            use_secure_aggregation: Whether to use secure multiparty computation
            **kwargs: Additional configuration parameters
        """
        self.enable_privacy = enable_privacy
        self.epsilon = epsilon
        self.delta = delta
        self.clip_values = clip_values
        self.clip_range = clip_range
        self.noise_scale = noise_scale
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.use_secure_aggregation = use_secure_aggregation
        
        # Validate parameters
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if delta <= 0 or delta >= 1:
            raise ValueError("Delta must be between 0 and 1")
        if noise_scale < 0 or noise_scale > 1:
            raise ValueError("Noise scale must be between 0 and 1")
        if min_samples < 10:
            raise ValueError("Minimum samples must be at least 10")
        if max_samples < min_samples:
            raise ValueError("Maximum samples must be greater than minimum samples")
        
        # Store any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Validate configuration
        self._validate()
        
        logger.info(f"Initialized privacy configuration: {self.__dict__}")
    
    def _validate(self):
        """Validate configuration parameters."""
        if self.epsilon > 10:
            logger.warning(f"High epsilon value ({self.epsilon}) provides minimal privacy protection")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PrivacyConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


class AggregationConfig:
    """Configuration for explanation aggregation."""
    
    def __init__(
        self,
        aggregation_method: str = 'weighted_average', 
        temporal_decay: float = 0.8,
        discard_outliers: bool = True,
        outlier_threshold: float = 2.0,
        consistency_threshold: float = 0.5,
        min_clients_per_round: int = 2,
        **kwargs
    ):
        """
        Initialize aggregation configuration.
        
        Args:
            aggregation_method: Method for aggregating explanations 
                               ('simple_average', 'weighted_average', 'median')
            temporal_decay: Weight for new explanations vs historical (0-1)
            discard_outliers: Whether to remove outlier explanations
            outlier_threshold: Z-score threshold for outlier detection
            consistency_threshold: Minimum agreement required between clients
            min_clients_per_round: Minimum clients required for aggregation
            **kwargs: Additional configuration parameters
        """
        self.aggregation_method = aggregation_method
        self.temporal_decay = temporal_decay
        self.discard_outliers = discard_outliers
        self.outlier_threshold = outlier_threshold
        self.consistency_threshold = consistency_threshold
        self.min_clients_per_round = min_clients_per_round
        
        # Store any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Validate configuration
        self._validate()
        
        logger.info(f"Initialized aggregation configuration: {self.__dict__}")
    
    def _validate(self):
        """Validate configuration parameters."""
        valid_methods = ['simple_average', 'weighted_average', 'median', 'robust_average']
        if self.aggregation_method not in valid_methods:
            raise ValueError(f"Invalid aggregation method: {self.aggregation_method}. "
                           f"Valid options are {valid_methods}")
        
        if not 0 <= self.temporal_decay <= 1:
            raise ValueError(f"Temporal decay must be in [0,1], got {self.temporal_decay}")
        
        if self.outlier_threshold <= 0:
            raise ValueError(f"Outlier threshold must be positive, got {self.outlier_threshold}")
        
        if not 0 <= self.consistency_threshold <= 1:
            raise ValueError(f"Consistency threshold must be in [0,1], got {self.consistency_threshold}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AggregationConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


class FederatedXAIConfig:
    """Master configuration for federated XAI system."""
    
    def __init__(
        self,
        explainability: Union[Dict[str, Any], ExplainabilityConfig] = None,
        privacy: Union[Dict[str, Any], PrivacyConfig] = None,
        aggregation: Union[Dict[str, Any], AggregationConfig] = None,
        collect_explanations: bool = True,
        explanation_rounds: Optional[List[int]] = None,
        save_explanations: bool = True,
        explanations_path: str = './results/explanations',
        **kwargs
    ):
        """
        Initialize federated XAI configuration.
        
        Args:
            explainability: Explainability configuration or dict
            privacy: Privacy configuration or dict
            aggregation: Aggregation configuration or dict
            collect_explanations: Whether to collect explanations during training
            explanation_rounds: Specific rounds to collect explanations (None = all)
            save_explanations: Whether to save explanations to disk
            explanations_path: Path to save explanations
            **kwargs: Additional configuration parameters
        """
        # Initialize sub-configurations
        self.explainability = self._init_subconfig(explainability, ExplainabilityConfig)
        self.privacy = self._init_subconfig(privacy, PrivacyConfig)
        self.aggregation = self._init_subconfig(aggregation, AggregationConfig)
        
        # Global settings
        self.collect_explanations = collect_explanations
        self.explanation_rounds = explanation_rounds
        self.save_explanations = save_explanations
        self.explanations_path = explanations_path
        
        # Store any additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Validate configuration
        self._validate()
        
        logger.info(f"Initialized federated XAI configuration")
    
    def _init_subconfig(self, config, config_class):
        """Initialize a sub-configuration from dict or existing config."""
        if config is None:
            return config_class()
        elif isinstance(config, dict):
            return config_class.from_dict(config)
        elif isinstance(config, config_class):
            return config
        else:
            raise TypeError(f"Expected dict or {config_class.__name__}, got {type(config)}")
    
    def _validate(self):
        """Validate the overall configuration."""
        if not self.collect_explanations:
            logger.warning("Explanation collection is disabled")
        
        if self.save_explanations and not os.path.exists(self.explanations_path):
            logger.info(f"Creating explanations directory: {self.explanations_path}")
            os.makedirs(self.explanations_path, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to nested dictionary."""
        config_dict = {
            'explainability': self.explainability.to_dict(),
            'privacy': self.privacy.to_dict(),
            'aggregation': self.aggregation.to_dict(),
            'collect_explanations': self.collect_explanations,
            'explanation_rounds': self.explanation_rounds,
            'save_explanations': self.save_explanations,
            'explanations_path': self.explanations_path
        }
        
        # Add any additional attributes
        for key, value in self.__dict__.items():
            if key not in config_dict and not key.startswith('_'):
                config_dict[key] = value
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FederatedXAIConfig':
        """Create configuration from nested dictionary."""
        # Extract sub-configurations
        explainability = config_dict.pop('explainability', {})
        privacy = config_dict.pop('privacy', {})
        aggregation = config_dict.pop('aggregation', {})
        
        return cls(
            explainability=explainability,
            privacy=privacy,
            aggregation=aggregation,
            **config_dict
        )
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved configuration to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'FederatedXAIConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        logger.info(f"Loaded configuration from {filepath}")
        return cls.from_dict(config_dict)


# Preset configurations for common scenarios
DEFAULT_CONFIG = FederatedXAIConfig()

MINIMAL_PRIVACY_CONFIG = FederatedXAIConfig(
    privacy=PrivacyConfig(
        enable_privacy=True,
        epsilon=10.0,  # Very relaxed privacy
        clip_values=True
    )
)

STRICT_PRIVACY_CONFIG = FederatedXAIConfig(
    privacy=PrivacyConfig(
        enable_privacy=True,
        epsilon=0.1,  # Very strict privacy
        delta=1e-6,
        clip_values=True,
        use_secure_aggregation=True
    )
)

LIME_ONLY_CONFIG = FederatedXAIConfig(
    explainability=ExplainabilityConfig(
        use_lime=True,
        use_shap=False
    )
)

SHAP_ONLY_CONFIG = FederatedXAIConfig(
    explainability=ExplainabilityConfig(
        use_lime=False,
        use_shap=True,
        shap_samples=200  # Increase samples since this is the only method
    )
)

NO_PRIVACY_CONFIG = FederatedXAIConfig(
    privacy=PrivacyConfig(
        enable_privacy=False
    )
)