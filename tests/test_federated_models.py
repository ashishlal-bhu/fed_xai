# test_federated_models.py

import sys
import os
import json
import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import project modules
from scripts.preprocess_data import preprocess_data
from server.orchestrator import FederatedOrchestrator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_federated_models')

def load_model(experiment_dir):
    """
    Load a trained model from experiment directory
    
    Args:
        experiment_dir (str): Path to experiment directory
    
    Returns:
        FederatedOrchestrator: Loaded orchestrator with trained model
    """
    # Load configuration
    config_path = os.path.join(experiment_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load data features (needed to initialize the model)
    _, _, _, _, features = preprocess_data(
        task=config.get('task', 'mortality'),
        sample_fraction=config.get('sample_fraction', 0.4)
    )
    
    # Define model architecture
    hidden_sizes = config.get('hidden_sizes', [256, 128, 64])
    model_config = {
        'input_dim': len(features),
        'hidden_sizes': hidden_sizes,
        'dropout': config.get('dropout', 0.3),
        'learning_rate': config.get('learning_rate', 0.001),
        'l2_reg': config.get('l2_reg', 0.01)
    }
    
    # Create orchestrator
    orchestrator = FederatedOrchestrator(
        input_dim=len(features),
        features=features,
        model_config=model_config
    )
    
    # Load model weights
    model_path = os.path.join(experiment_dir, 'final_model.h5')
    orchestrator.server.model.load_weights(model_path)
    
    return orchestrator

def evaluate_model(orchestrator, X_test, y_test, threshold=0.5):
    """
    Evaluate a trained model on test data
    
    Args:
        orchestrator (FederatedOrchestrator): Orchestrator with trained model
        X_test (pandas.DataFrame): Test features
        y_test (pandas.Series): Test labels
        threshold (float): Decision threshold for binary classification
    
    Returns:
        dict: Evaluation metrics
    """
    # Get predictions
    y_pred_proba = orchestrator.server.model.predict(X_test)
    y_pred = (y_pred_proba > threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_pred_proba)),
        'test_samples': len(y_test),
        'positive_samples': int(sum(y_test)),
        'negative_samples': int(len(y_test) - sum(y_test)),
        'threshold': threshold
    }
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    return metrics

def plot_confusion_matrix(metrics, output_dir):
    """
    Plot confusion matrix
    
    Args:
        metrics (dict): Evaluation metrics with confusion matrix values
        output_dir (str): Directory to save the plot
    """
    cm = np.array([
        [metrics['true_negatives'], metrics['false_positives']],
        [metrics['false_negatives'], metrics['true_positives']]
    ])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    output_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    return output_path

def test_experiment(experiment_dir, test_data=None):
    """
    Test a trained model from an experiment
    
    Args:
        experiment_dir (str): Path to experiment directory
        test_data (tuple, optional): Tuple of (X_test, y_test). If None, loads test data.
    
    Returns:
        dict: Test results
    """
    logger.info(f"Testing model from: {experiment_dir}")
    
    # Create test results directory
    test_dir = os.path.join(experiment_dir, 'test_results')
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Load model
        orchestrator = load_model(experiment_dir)
        
        # Load test data if not provided
        if test_data is None:
            _, X_test, _, y_test, _ = preprocess_data(
                task="mortality", 
                sample_fraction=0.4
            )
        else:
            X_test, y_test = test_data
        
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Positive samples: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.2f}%)")
        
        # Evaluate model with default threshold
        metrics = evaluate_model(orchestrator, X_test, y_test)
        
        # Plot confusion matrix
        cm_path = plot_confusion_matrix(metrics, test_dir)
        logger.info(f"Confusion matrix saved to: {cm_path}")
        
        # Print and save metrics
        logger.info("Test Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"- {key}: {value:.4f}")
            else:
                logger.info(f"- {key}: {value}")
        
        metrics_path = os.path.join(test_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Evaluate model with different thresholds
        thresholds = np.arange(0.1, 1.0, 0.1)
        threshold_metrics = {}
        
        for threshold in thresholds:
            threshold_metrics[f"{threshold:.1f}"] = evaluate_model(
                orchestrator, X_test, y_test, threshold=threshold
            )
        
        # Save threshold metrics
        thresholds_path = os.path.join(test_dir, 'threshold_metrics.json')
        with open(thresholds_path, 'w') as f:
            json.dump(threshold_metrics, f, indent=2)
        
        # Plot threshold vs metrics
        plt.figure(figsize=(10, 6))
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for metric in metrics_to_plot:
            values = [m[metric] for m in threshold_metrics.values()]
            plt.plot(thresholds, values, marker='o', label=metric)
        
        plt.xlabel('Threshold')
        plt.ylabel('Metric Value')
        plt.title('Metrics vs. Threshold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        threshold_plot_path = os.path.join(test_dir, 'threshold_metrics.png')
        plt.savefig(threshold_plot_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Threshold analysis saved to: {threshold_plot_path}")
        
        return {
            'success': True,
            'metrics': metrics,
            'threshold_metrics': threshold_metrics
        }
    
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        logger.error("Detailed error: ", exc_info=True)
        
        # Save error information
        error_path = os.path.join(test_dir, 'error.log')
        with open(error_path, 'w') as f:
            f.write(f"Error: {str(e)}\n")
        
        return {
            'success': False,
            'error': str(e)
        }

def test_all_experiments(experiments_dir):
    """
    Test all experiments in a directory
    
    Args:
        experiments_dir (str): Path to experiments directory
    
    Returns:
        dict: Summary of all test results
    """
    logger.info(f"Testing all experiments in: {experiments_dir}")
    
    # Find all experiment directories
    experiment_dirs = []
    for item in os.listdir(experiments_dir):
        item_path = os.path.join(experiments_dir, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'config.json')):
            experiment_dirs.append(item_path)
    
    # Load test data once to use for all experiments
    _, X_test, _, y_test, _ = preprocess_data(task="mortality", sample_fraction=0.4)
    test_data = (X_test, y_test)
    
    # Test each experiment
    all_results = {}
    for exp_dir in experiment_dirs:
        exp_name = os.path.basename(exp_dir)
        logger.info(f"Testing experiment: {exp_name}")
        
        results = test_experiment(exp_dir, test_data)
        all_results[exp_name] = results
    
    # Save summary of all results
    summary_path = os.path.join(experiments_dir, 'test_summary.json')
    with open(summary_path, 'w') as f:
        summary = {
            'experiments': len(experiment_dirs),
            'successful_tests': sum(1 for r in all_results.values() if r.get('success', False)),
            'failed_tests': sum(1 for r in all_results.values() if not r.get('success', False)),
            'results': all_results
        }
        json.dump(summary, f, indent=2)
    
    # Create comparison table
    comparison_data = []
    for exp_name, results in all_results.items():
        if results.get('success', False):
            metrics = results.get('metrics', {})
            comparison_data.append({
                'experiment': exp_name,
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'roc_auc': metrics.get('roc_auc', 0)
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # Sort by F1 score
        df = df.sort_values('f1_score', ascending=False)
        
        # Save as CSV
        csv_path = os.path.join(experiments_dir, 'comparison.csv')
        df.to_csv(csv_path, index=False)
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        
        # Plot for each experiment
        bar_width = 0.15
        positions = np.arange(len(df))
        
        for i, metric in enumerate(metrics_to_plot):
            offset = bar_width * (i - len(metrics_to_plot)/2 + 0.5)
            plt.bar(positions + offset, df[metric], width=bar_width, label=metric)
        
        plt.xlabel('Experiments')
        plt.ylabel('Metric Value')
        plt.title('Comparison of Experiment Results')
        plt.xticks(positions, df['experiment'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.grid(True, axis='y', alpha=0.3)
        
        comparison_plot_path = os.path.join(experiments_dir, 'comparison.png')
        plt.savefig(comparison_plot_path, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comparison saved to: {comparison_plot_path}")
    
    logger.info(f"All tests completed. Summary saved to {summary_path}")
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test federated learning models")
    parser.add_argument('--dir', '-d', help='Path to experiment directory')
    parser.add_argument('--all', '-a', action='store_true', help='Test all experiments in directory')
    args = parser.parse_args()
    
    if args.all and args.dir:
        test_all_experiments(args.dir)
    elif args.dir:
        test_experiment(args.dir)
    else:
        parser.print_help()
    