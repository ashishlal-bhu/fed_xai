import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('visualization')

def plot_training_history(history):
    """
    Plot training history metrics
    
    Parameters:
    -----------
    history : keras.callbacks.History
        Training history from model.fit()
    """
    logger.info("Plotting training history...")
    
    try:
        metrics = history.history
        n_metrics = len(metrics.keys()) // 2  # Divide by 2 for train/val pairs
        
        plt.figure(figsize=(15, 5))
        
        for i, metric in enumerate(list(metrics.keys())[:n_metrics], 1):
            plt.subplot(1, n_metrics, i)
            
            plt.plot(metrics[metric], label=f'Training {metric}')
            if f'val_{metric}' in metrics:
                plt.plot(metrics[f'val_{metric}'], label=f'Validation {metric}')
                
            plt.title(f'Model {metric}')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.legend()
            
        plt.tight_layout()
        plt.savefig('train_hist.png')
        
    except Exception as e:
        logger.error(f"Error plotting training history: {str(e)}")
        raise

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix with annotations
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    """
    logger.info("Plotting confusion matrix...")
    
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.savefig('confusion_matrix.png')
        
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")
        raise

def visualize_lime_explanation(explanation, top_features=10):
    """
    Visualize LIME explanation
    
    Parameters:
    -----------
    explanation : LimeTabularExplanation
        LIME explanation object
    top_features : int
        Number of top features to display
    """
    logger.info("Visualizing LIME explanation...")
    
    try:
        # Get the explanation for the predicted class
        exp_list = explanation.as_list()
        
        # Sort by absolute values and get top features
        sorted_exp = sorted(exp_list, key=lambda x: abs(x[1]), reverse=True)
        top_exp = sorted_exp[:top_features]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        features, values = zip(*top_exp)
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(values)), values)
        
        # Color bars based on positive/negative contribution
        for i, bar in enumerate(bars):
            if values[i] < 0:
                bar.set_color('red')
            else:
                bar.set_color('blue')
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Contribution')
        plt.title('LIME Feature Importance')
        
        plt.tight_layout()
        plt.savefig('lime_exp.png')
        
    except Exception as e:
        logger.error(f"Error visualizing LIME explanation: {str(e)}")
        raise

def visualize_shap_values(shap_values, feature_names, instance_idx=0):
    """
    Visualize SHAP values for a single instance
    """
    logger.info("Visualizing SHAP values...")
    
    try:
        # Extract values from the explanation dictionary
        values = shap_values['values']
        expected_value = shap_values['expected_value']
        
        # Debug logging
        logger.info(f"Values shape: {values.shape if hasattr(values, 'shape') else 'no shape'}")
        logger.info(f"Number of features: {len(feature_names)}")
        
        # Convert values to numpy array if needed
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        
        # If we have more values than features, we're probably getting both classes
        # Take only the values corresponding to the positive class
        if len(values.shape) > 1 and values.shape[1] > len(feature_names):
            # Take only the values for positive class
            instance_values = values[instance_idx, :len(feature_names)]
        else:
            instance_values = values[instance_idx] if len(values.shape) > 1 else values[:len(feature_names)]
        
        # Ensure we have 1D array of values
        instance_values = instance_values.ravel()
        
        # Debug logging
        logger.info(f"Processed instance values shape: {instance_values.shape}")
        logger.info(f"First few values: {instance_values[:3]}")
        
        # Create feature importance pairs and sort
        importance_pairs = [(name, float(value)) for name, value in zip(feature_names, instance_values)]
        sorted_pairs = sorted(importance_pairs, key=lambda x: abs(x[1]), reverse=True)
        
        # Split into separate lists for plotting
        features, values = zip(*sorted_pairs)
        values = np.array(values)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Create horizontal bar plot
        bars = plt.barh(range(len(values)), values)
        
        # Color bars based on positive/negative contribution
        for i, bar in enumerate(bars):
            bar.set_color('blue' if values[i] >= 0 else 'red')
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('SHAP value')
        plt.title(f'SHAP Feature Importance\nBase value: {float(expected_value):.3f}')
        
        plt.tight_layout()
        plt.savefig('shap_fig.png')
        
    except Exception as e:
        logger.error(f"Error visualizing SHAP values: {str(e)}")
        logger.error("Detailed error: ", exc_info=True)
        raise

def visualize_feature_importance(feature_names, importance_values, title="Feature Importance"):
    """
    Create a bar plot of overall feature importance
    
    Parameters:
    -----------
    feature_names : list
        Names of the features
    importance_values : array-like
        Importance values for each feature
    title : str
        Plot title
    """
    logger.info("Visualizing feature importance...")
    
    try:
        # Ensure values are numpy array
        importance_values = np.array(importance_values)
        
        # Sort by importance
        sorted_idx = np.argsort(np.abs(importance_values))
        pos = np.arange(sorted_idx.shape[0]) + .5
        
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Create horizontal bar plot
        bars = plt.barh(pos, importance_values[sorted_idx])
        
        # Color bars based on positive/negative contribution
        for i, bar in enumerate(bars):
            if importance_values[sorted_idx[i]] < 0:
                bar.set_color('red')
            else:
                bar.set_color('blue')
        
        plt.yticks(pos, np.array(feature_names)[sorted_idx])
        plt.xlabel('Importance')
        plt.title(title)
        
        plt.tight_layout()
        plt.savefig('feature_imo.png')
        
    except Exception as e:
        logger.error(f"Error visualizing feature importance: {str(e)}")
        raise