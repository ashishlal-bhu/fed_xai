import sys
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, roc_auc_score

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
from scripts.preprocess_data import preprocess_data
from models.xai_model import XAIModel
from utils.visualization import (
    plot_training_history,
    plot_confusion_matrix,
    visualize_lime_explanation,
    visualize_shap_values,
    visualize_feature_importance
)

# Set up logging with timestamp for file
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_log_{timestamp}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('training')

def validate_data_split(X_train, X_test, y_train, y_test):
    """Validate data splits and class distribution"""
    logger.info("Validating data splits...")
    
    # Check shapes
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    
    # Check class distribution
    train_dist = pd.Series(y_train).value_counts(normalize=True)
    test_dist = pd.Series(y_test).value_counts(normalize=True)
    
    logger.info("\nClass distribution:")
    logger.info(f"Training set:\n{train_dist}")
    logger.info(f"Test set:\n{test_dist}")
    
    # Check for data leakage
    train_idx = set(X_train.index) if hasattr(X_train, 'index') else set()
    test_idx = set(X_test.index) if hasattr(X_test, 'index') else set()
    overlap = train_idx.intersection(test_idx)
    
    if overlap:
        logger.warning(f"Found {len(overlap)} overlapping indices between train and test!")
        
    return True

def find_best_hyperparameters(X_train, y_train):
    """Find best hyperparameters using a simplified search"""
    logger.info("Finding best hyperparameters...")
    
    # Reduced parameter combinations
    params = [
        {'units': 128, 'dropout': 0.3, 'learning_rate': 0.001},  # Default params
        {'units': 64, 'dropout': 0.2, 'learning_rate': 0.001},   # Lighter model
    ]
    
    best_score = 0
    best_params = None
    
    for p in params:
        # Initialize model with these parameters
        model = XAIModel(
            input_dim=X_train.shape[1],
            features=X_train.columns if hasattr(X_train, 'columns') else None,
            **p
        )
        
        # Perform quick validation (no cross-validation)
        history = model.train(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=20,  # Reduced epochs
            batch_size=32
        )
        
        # Get the best validation AUC score
        val_auc = max(history.history['val_auc'])
        logger.info(f"Parameters: {p}, Validation AUC: {val_auc:.3f}")
        
        if val_auc > best_score:
            best_score = val_auc
            best_params = p
    
    logger.info(f"Best parameters found: {best_params}")
    return best_params

def train_and_evaluate(task="mortality",sample_fraction=0.1):
    """Main training and evaluation function"""
    try:
        # Step 1: Load and preprocess data
        logger.info(f"Starting preprocessing for task: {task}")
        X_train, X_test, y_train, y_test, features = preprocess_data(task,sample_fraction)
        
        # Validate data splits
        validate_data_split(X_train, X_test, y_train, y_test)
        
        # Find best hyperparameters
        best_params = find_best_hyperparameters(X_train, y_train)
        
        # Initialize model with best parameters
        logger.info("Initializing model with best parameters")
        model = XAIModel(
            input_dim=X_train.shape[1],
            features=features,
            **best_params
        )
        
        # Train model
        logger.info("Starting model training")
        history = model.train(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=100,  # Increased epochs with early stopping
            batch_size=32
        )
        
        # Evaluate model
        logger.info("Evaluating model")
        y_pred, y_pred_proba = model.evaluate(X_test, y_test)
        
        # Initialize explainers
        logger.info("Initializing explainers")
        model.initialize_explainers(X_train, y_train)
        
        return model, history, (X_test, y_test, y_pred, y_pred_proba)
        
    except Exception as e:
        logger.error(f"Error in training and evaluation: {str(e)}")
        raise

def generate_explanations(model, X_test, y_test, n_samples=3):
    """Generate and visualize explanations for sample instances"""
    try:
        logger.info(f"Generating explanations for {n_samples} samples")
        
        # Select balanced samples if possible
        pos_indices = np.where(y_test == 1)[0]
        neg_indices = np.where(y_test == 0)[0]
        
        # Try to get equal number of positive and negative samples
        n_each = min(n_samples // 2, min(len(pos_indices), len(neg_indices)))
        selected_pos = np.random.choice(pos_indices, size=n_each, replace=False)
        selected_neg = np.random.choice(neg_indices, size=n_each, replace=False)
        
        indices = np.concatenate([selected_pos, selected_neg])
        np.random.shuffle(indices)
        
        for idx in indices:
            instance = X_test.iloc[idx:idx+1]
            true_label = y_test.iloc[idx]
            
            logger.info(f"\nExplaining instance {idx}")
            logger.info(f"True label: {'Deceased' if true_label == 1 else 'Not Deceased'}")
            
            # Get predictions
            pred_proba = model.predict_proba(instance)[0]
            pred_class = 1 if pred_proba[1] > 0.5 else 0
            
            logger.info(f"Predicted probability of mortality: {pred_proba[1]:.3f}")
            logger.info(f"Predicted class: {'Deceased' if pred_class == 1 else 'Not Deceased'}")
            
            # Generate explanations
            explanations = model.explain_instance(instance)
            
            # Visualize explanations
            if 'lime' in explanations:
                visualize_lime_explanation(explanations['lime'])
            
            if 'shap' in explanations:
                visualize_shap_values(explanations['shap'], model.feature_names)
                
    except Exception as e:
        logger.error(f"Error generating explanations: {str(e)}")
        raise

def main():
    """Main execution function"""
    try:
        logger.info("Starting XAI model training pipeline")
        
        # Train and evaluate model
        model, history, eval_results = train_and_evaluate("mortality",sample_fraction=0.4)
        X_test, y_test, y_pred, y_pred_proba = eval_results
        
        # Plot training history
        logger.info("Plotting training history")
        plot_training_history(history)
        
        # Plot confusion matrix
        logger.info("Plotting confusion matrix")
        plot_confusion_matrix(y_test, y_pred)
        
        # Generate explanations for sample instances
        generate_explanations(model, X_test, y_test)
        
        logger.info("Pipeline completed successfully")
        return model, history, eval_results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Run the pipeline
        model, history, eval_results = main()
        
    except Exception as e:
        logger.error("Script failed with error: %s", str(e))
        sys.exit(1)