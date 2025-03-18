import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger('fed_visualization')

def ensure_results_dir():
    """Create results directory if it doesn't exist"""
    results_dir = os.path.join(os.getcwd(), 'results')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    return results_dir

def save_training_plots(training_history: dict, is_final: bool = False):
    """
    Save plots of training metrics.
    
    Args:
        training_history: Dictionary containing training history
        is_final: Whether this is the final plot
    """
    logger.info("Saving training plots...")
    
    try:
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.getcwd(), 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Extract metrics
        global_metrics = training_history.get('global_metrics', [])
        
        if not global_metrics:
            logger.warning("No global metrics found in training history")
            return
        
        # Get round numbers
        rounds = list(range(1, len(global_metrics) + 1))
        
        # Safely extract metrics, using 0.0 as default if key doesn't exist
        accuracies = [m.get('accuracy', 0.0) if m else 0.0 for m in global_metrics]
        aucs = [m.get('auc', 0.0) if m else 0.0 for m in global_metrics]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(rounds, accuracies, 'b-o', linewidth=2, markersize=6)
        plt.title('Global Model Accuracy')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        # Plot AUC
        plt.subplot(1, 2, 2)
        plt.plot(rounds, aucs, 'r-o', linewidth=2, markersize=6)
        plt.title('Global Model AUC')
        plt.xlabel('Round')
        plt.ylabel('AUC')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"training_metrics{'_final' if is_final else ''}_{timestamp}.png"
        filepath = os.path.join(results_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training plot to {filepath}")
        
        return filepath
    except Exception as e:
        logger.error(f"Error saving training plots: {str(e)}")
        # Don't re-raise the exception to prevent halting the training process
        return None

def save_client_contributions(training_history: dict, is_final: bool = False):
    """
    Save plots of client contributions.
    
    Args:
        training_history: Dictionary containing training history
        is_final: Whether this is the final plot
    """
    logger.info("Saving client contribution plots...")
    
    try:
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.getcwd(), 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Extract client metrics
        rounds = training_history.get('rounds', [])
        
        if not rounds:
            logger.warning("No round data found in training history")
            return
        
        # Get client IDs
        client_ids = set()
        for round_data in rounds:
            client_metrics = round_data.get('client_metrics', {})
            client_ids.update(client_metrics.keys())
        
        client_ids = sorted(list(client_ids))
        
        if not client_ids:
            logger.warning("No client data found in training history")
            return
        
        # Extract client accuracies and AUCs
        client_accuracies = {client_id: [] for client_id in client_ids}
        client_aucs = {client_id: [] for client_id in client_ids}
        round_numbers = []
        
        for round_data in rounds:
            round_numbers.append(round_data.get('round', 0))
            client_metrics = round_data.get('client_metrics', {})
            
            for client_id in client_ids:
                metrics = client_metrics.get(client_id, {})
                # Use default value of 0.0 if metric is missing
                client_accuracies[client_id].append(metrics.get('accuracy', 0.0) if metrics else 0.0)
                client_aucs[client_id].append(metrics.get('auc', 0.0) if metrics else 0.0)
        
        # Create plots
        plt.figure(figsize=(12, 8))
        
        # Plot accuracies
        plt.subplot(2, 1, 1)
        for client_id in client_ids:
            plt.plot(round_numbers, client_accuracies[client_id], 'o-', linewidth=2, label=client_id)
        
        plt.title('Client Accuracies')
        plt.xlabel('Round')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        
        # Plot AUCs
        plt.subplot(2, 1, 2)
        for client_id in client_ids:
            plt.plot(round_numbers, client_aucs[client_id], 'o-', linewidth=2, label=client_id)
        
        plt.title('Client AUCs')
        plt.xlabel('Round')
        plt.ylabel('AUC')
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"client_contributions{'_final' if is_final else ''}_{timestamp}.png"
        filepath = os.path.join(results_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved client contribution plot to {filepath}")
        
        return filepath
    except Exception as e:
        logger.error(f"Error saving client contribution plots: {str(e)}")
        # Don't re-raise the exception to prevent halting the training process
        return None