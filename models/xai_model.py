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
from typing import Union, List, Dict, Tuple, Optional
from dataclasses import dataclass, field

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('xai_model')

@dataclass
class ExplanationMetrics:
    """Store and track explanation metrics"""
    global_feature_importance: Dict[str, float] = field(default_factory=dict)
    feature_stability: Dict[str, float] = field(default_factory=dict)
    explanation_confidence: float = 0.0

class XAIModel(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        input_dim: int,
        features: Union[List[str], pd.Index],
        units: int = 128,
        dropout: float = 0.3,
        learning_rate: float = 0.001
    ):
        """Initialize XAI Model"""
        logger.info("Initializing XAI Model...")
        
        # Store all attributes
        self.input_dim = input_dim
        self.features = features  # Store original features
        self.feature_names = self._validate_features(features)
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # Initialize model and explainers
        self.model = self._create_model()
        self.lime_explainer = None
        self.shap_explainer = None
        
        logger.info(f"Model initialized with:")
        logger.info(f"- Input dimensions: {input_dim}")
        logger.info(f"- Hidden units: {units}")
        logger.info(f"- Dropout rate: {dropout}")
        logger.info(f"- Learning rate: {learning_rate}")

    def _validate_features(self, features: Union[List[str], pd.Index]) -> List[str]:
        """Validate and process feature names"""
        logger.info("Validating features...")
        
        # Handle different input types
        if isinstance(features, pd.DataFrame):
            feature_names = features.columns.tolist()
        elif isinstance(features, pd.Index):
            feature_names = features.tolist()
        elif isinstance(features, (list, np.ndarray)):
            feature_names = list(features)
        else:
            raise ValueError(f"Unsupported features type: {type(features)}")
            
        # Validate feature count matches input dimension
        if len(feature_names) != self.input_dim:
            raise ValueError(
                f"Number of features ({len(feature_names)}) "
                f"doesn't match input_dim ({self.input_dim})"
            )
            
        # Ensure all feature names are strings
        feature_names = [str(f) for f in feature_names]
        
        # Check for duplicate feature names
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("Duplicate feature names found")
            
        logger.info(f"Validated {len(feature_names)} features")
        return feature_names

    def _create_model(self) -> tf.keras.Model:
        """Create model architecture"""
        # Create proper input layer
        inputs = tf.keras.Input(shape=(self.input_dim,))
        
        # First dense block
        x = Dense(self.units * 2, activation='relu',
                 kernel_initializer='he_normal',
                 kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x)
        
        # Second dense block
        x = Dense(self.units, activation='relu',
                 kernel_initializer='he_normal',
                 kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout/2)(x)
        
        # Third dense block
        x = Dense(self.units//2, activation='relu',
                 kernel_initializer='he_normal',
                 kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout/4)(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(x)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        return model

    def _validate_input_data(self, X: Union[np.ndarray, pd.DataFrame],
                           y: Optional[Union[np.ndarray, pd.Series]] = None) -> Tuple:
        """Validate input data format and dimensions"""
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
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame],
            y: Union[np.ndarray, pd.Series],
            **kwargs) -> 'XAIModel':
        """Scikit-learn compatible fit method"""
        self.train(X, y, **kwargs)
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make binary predictions"""
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Get probability predictions"""
        X_arr = self._validate_input_data(X)
        pred = self.model.predict(X_arr, verbose=0)
        return np.column_stack([1 - pred, pred])

    def train(self, X: Union[np.ndarray, pd.DataFrame],
              y: Union[np.ndarray, pd.Series],
              validation_split: float = 0.2,
              epochs: int = 100,
              batch_size: int = 32) -> tf.keras.callbacks.History:
        """Train the model with input validation"""
        logger.info("Starting model training...")
        
        try:
            # Validate input data
            X_arr, y_arr = self._validate_input_data(X, y)
            
            # Convert to float32 if not already
            X_arr = X_arr.astype('float32')
            
            # Verify no invalid values
            if np.any(np.isnan(X_arr)) or np.any(np.isinf(X_arr)):
                raise ValueError("Input contains NaN or Inf values")
            
            # Calculate class weights
            unique, counts = np.unique(y_arr, return_counts=True)
            total = len(y_arr)
            class_weight = {int(val): np.float32(total/(len(unique) * count)) 
                          for val, count in zip(unique, counts)}
            
            logger.info(f"Class distribution: {dict(zip(unique, counts))}")
            logger.info(f"Class weights: {class_weight}")
            
            # Define callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    verbose=1
                )
            ]
            
            # Store training data for explainers
            self.X_train = X_arr
            
            # Train model
            history = self.model.fit(
                X_arr, y_arr,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1
            )
            
            logger.info("Model training completed")
            return history
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            logger.error("Detailed error: ", exc_info=True)
            raise

    def evaluate(self, X_test: Union[np.ndarray, pd.DataFrame],
                y_test: Union[np.ndarray, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate model performance"""
        logger.info("Evaluating model performance...")
        
        try:
            # Validate input data
            X_test_arr, y_test_arr = self._validate_input_data(X_test, y_test)
            
            # Get predictions
            y_pred_proba = self.predict_proba(X_test_arr)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            logger.info("\nClassification Report:")
            logger.info("\n" + classification_report(y_test_arr, y_pred))
            
            logger.info("\nConfusion Matrix:")
            logger.info("\n" + str(confusion_matrix(y_test_arr, y_pred)))
            
            auc_score = roc_auc_score(y_test_arr, y_pred_proba)
            logger.info(f"\nROC-AUC Score: {auc_score:.3f}")
            
            return y_pred, y_pred_proba
            
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            raise

    def initialize_explainers(self, X_train: Union[np.ndarray, pd.DataFrame],
                            y_train: Union[np.ndarray, pd.Series]) -> None:
        """Initialize LIME and SHAP explainers"""
        logger.info("Initializing explainers...")
        
        try:
            # Validate training data
            X_train_arr, y_train_arr = self._validate_input_data(X_train, y_train)
            
            # Initialize LIME explainer
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train_arr,
                feature_names=self.feature_names,
                class_names=['Not Deceased', 'Deceased'],
                mode='classification',
                training_labels=y_train_arr
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

    def explain_instance(
        self,
        instance: Union[np.ndarray, pd.Series],
        method: str = 'both'
    ) -> Dict:
        """Generate explanations for a single instance"""
        logger.info(f"Generating {method} explanation...")
        
        if self.lime_explainer is None or self.shap_explainer is None:
            raise ValueError("Explainers not initialized. Call initialize_explainers first.")
        
        try:
            instance_arr = self._validate_input_data(instance)
            explanations = {}
            
            if method in ['lime', 'both']:
                lime_exp = self.lime_explainer.explain_instance(
                    instance_arr[0],
                    self.predict_proba,
                    num_features=len(self.feature_names)
                )
                explanations['lime'] = lime_exp
            
            if method in ['shap', 'both']:
                # Get SHAP values
                shap_values = self.shap_explainer.shap_values(instance_arr)
                expected_value = self.shap_explainer.expected_value
                
                # Debug logging
                logger.info(f"Raw SHAP values type: {type(shap_values)}")
                logger.info(f"Raw expected_value type: {type(expected_value)}")
                
                # Handle binary classification case
                if isinstance(shap_values, list):
                    # For binary classification, take values for positive class
                    values = np.array(shap_values[1])
                    if isinstance(expected_value, list):
                        exp_val = expected_value[1]
                    elif isinstance(expected_value, np.ndarray):
                        exp_val = expected_value[1] if expected_value.size > 1 else expected_value[0]
                    else:
                        exp_val = expected_value
                else:
                    # For single output, reshape if needed
                    values = np.array(shap_values)
                    if values.shape[-1] == 2 * len(self.feature_names):
                        # If we have values for both classes, take positive class
                        values = values[..., len(self.feature_names):]
                    if isinstance(expected_value, np.ndarray):
                        exp_val = expected_value[0]
                    else:
                        exp_val = expected_value

                # Debug logging
                logger.info(f"Processed values shape: {values.shape}")
                logger.info(f"Number of features: {len(self.feature_names)}")
                logger.info(f"Final expected_value: {exp_val}")

                explanations['shap'] = {
                    'values': values,
                    'expected_value': float(exp_val)
                }
            
            return explanations
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            logger.error("Detailed error: ", exc_info=True)
            raise