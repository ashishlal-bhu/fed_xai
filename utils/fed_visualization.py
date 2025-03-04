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

def save_training_plots(history: dict, is_final: bool = False):
    """
    Create and save training progress plots
    
    Args:
        history: Training history dictionary
        is_final: Whether this is the final plot after all rounds
    """
    if not is_final:
        return  # Only save plots at the end of training
        
    logger.info("Creating and saving training plots...")
    results_dir = ensure_results_dir()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # Extract metrics
        rounds = [m['round'] for m in history['rounds']]
        global_metrics = history['global_metrics']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Federated Learning Performance', fontsize=16)
        
        # Plot global metrics
        accuracies = [m['accuracy'] for m in global_metrics]
        aucs = [m['auc'] for m in global_metrics]
        
        axes[0, 0].plot(rounds, accuracies, marker='o', linewidth=2)
        axes[0, 0].set_title('Global Model Accuracy')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(rounds, aucs, marker='o', linewidth=2, color='orange')
        axes[0, 1].set_title('Global Model AUC')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].grid(True)
        
        # Extract client metrics
        client_metrics = {}
        for round_data in history['rounds']:
            for client_id, metrics in round_data['client_metrics'].items():
                if client_id not in client_metrics:
                    client_metrics[client_id] = {'accuracy': [], 'auc': []}
                client_metrics[client_id]['accuracy'].append(metrics['accuracy'])
                client_metrics[client_id]['auc'].append(metrics['auc'])
        
        # Plot client metrics
        for client_id, metrics in client_metrics.items():
            axes[1, 0].plot(rounds[:len(metrics['accuracy'])], 
                          metrics['accuracy'],
                          marker='.',
                          label=f'Client {client_id}')
            
            axes[1, 1].plot(rounds[:len(metrics['auc'])],
                          metrics['auc'],
                          marker='.',
                          label=f'Client {client_id}')
        
        axes[1, 0].set_title('Client Accuracies')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].set_title('Client AUCs')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(results_dir, f'training_progress_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to {plot_path}")
        
    except Exception as e:
        logger.error(f"Error creating training plots: {str(e)}")
        raise

def save_client_contributions(history: dict, is_final: bool = False):
    """
    Create and save client contribution analysis plots
    
    Args:
        history: Training history dictionary
        is_final: Whether this is the final plot after all rounds
    """
    if not is_final:
        return  # Only save plots at the end of training
        
    logger.info("Creating and saving client contribution plots...")
    results_dir = ensure_results_dir()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        # Prepare client data
        client_data = []
        for round_data in history['rounds']:
            round_num = round_data['round']
            for client_id, metrics in round_data['client_metrics'].items():
                client_data.append({
                    'round': round_num,
                    'client_id': client_id,
                    'accuracy': metrics['accuracy'],
                    'auc': metrics['auc']
                })
        
        df = pd.DataFrame(client_data)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
        fig.suptitle('Client Contribution Analysis', fontsize=16)
        
        # Box plots
        sns.boxplot(data=df, x='client_id', y='accuracy', ax=axes[0])
        axes[0].set_title('Distribution of Client Accuracies')
        axes[0].set_xlabel('Client ID')
        axes[0].set_ylabel('Accuracy')
        
        sns.boxplot(data=df, x='client_id', y='auc', ax=axes[1])
        axes[1].set_title('Distribution of Client AUCs')
        axes[1].set_xlabel('Client ID')
        axes[1].set_ylabel('AUC')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(results_dir, f'client_contributions_{timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Client contribution plots saved to {plot_path}")
        
    except Exception as e:
        logger.error(f"Error creating client contribution plots: {str(e)}")
        raise