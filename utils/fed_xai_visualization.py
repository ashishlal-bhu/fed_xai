# utils/fed_xai_visualization.py
"""
Visualization module for Federated Explainable AI.

This module provides functions to visualize explanations generated
in the federated learning process, including feature importance,
client agreement, and explanation evolution over time.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import json

# Set up logging
logger = logging.getLogger('fed_xai_visualization')

# Define custom color palettes for consistency
LIME_COLORS = sns.color_palette("Blues_r", 10)
SHAP_COLORS = sns.color_palette("Oranges_r", 10)
CLIENT_COLORS = sns.color_palette("husl", 10)

def ensure_results_dir(subdir=None):
    """Create results directory if it doesn't exist"""
    results_dir = os.path.join(os.getcwd(), 'results')
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    if subdir:
        subdir_path = os.path.join(results_dir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
        return subdir_path
    
    return results_dir

def save_feature_importance_plot(
    explanation_data: Dict,
    round_num: int,
    explanation_type: str = 'lime',
    max_features: int = 15,
    output_dir: Optional[str] = None
):
    """
    Create and save a horizontal bar plot of feature importance.
    
    Args:
        explanation_data: Dictionary with explanation data
        round_num: Training round number
        explanation_type: Type of explanation ('lime' or 'shap')
        max_features: Maximum number of features to display
        output_dir: Directory to save plot
    """
    logger.info(f"Creating feature importance plot for round {round_num} ({explanation_type})")
    
    # Get the explanations
    if 'explanations' not in explanation_data:
        logger.warning("No explanations found in data")
        return
    
    explanations = explanation_data['explanations']
    if explanation_type not in explanations:
        logger.warning(f"No {explanation_type} explanations found")
        return
    
    # Get feature importance
    feature_importance = explanations[explanation_type]
    
    # Sort features by importance and take top N
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:max_features]
    
    # Extract feature names and values
    features = [item[0] for item in sorted_features]
    importance = [item[1] for item in sorted_features]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Select colors based on explanation type
    colors = LIME_COLORS if explanation_type == 'lime' else SHAP_COLORS
    
    # Create bar plot
    bars = plt.barh(
        range(len(features)),
        [abs(x) for x in importance],  # Use absolute values for visualization
        color=colors[0],
        alpha=0.8
    )
    
    # Color bars based on whether the feature has positive or negative impact
    for i, value in enumerate(importance):
        if value < 0:
            bars[i].set_color(colors[-1])
    
    # Add feature names as y-tick labels
    plt.yticks(range(len(features)), features)
    plt.gca().invert_yaxis()  # Invert y-axis to have most important feature at the top
    
    # Add title and labels
    plt.title(f"Top {len(features)} Features by {explanation_type.upper()} Importance - Round {round_num}")
    plt.xlabel("Feature Importance")
    
    # Add values to the end of each bar
    for i, value in enumerate(importance):
        plt.text(
            abs(value) + 0.01 * max(abs(x) for x in importance),
            i,
            f"{value:.3f}",
            va='center'
        )
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = ensure_results_dir('explanations')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{explanation_type}_importance_round_{round_num}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved feature importance plot to {filepath}")
    
    return filepath

def save_client_feature_importance_comparison(
    client_explanations: Dict[str, Dict],
    round_num: int,
    explanation_type: str = 'lime',
    max_features: int = 10,
    max_clients: int = 5,
    output_dir: Optional[str] = None
):
    """
    Create and save a comparison plot of feature importance across clients.
    
    Args:
        client_explanations: Dictionary mapping client IDs to explanation data
        round_num: Training round number
        explanation_type: Type of explanation ('lime' or 'shap')
        max_features: Maximum number of features to display
        max_clients: Maximum number of clients to include
        output_dir: Directory to save plot
    """
    logger.info(f"Creating client feature importance comparison for round {round_num}")
    
    # Limit number of clients if needed
    if len(client_explanations) > max_clients:
        # Take a sample of clients
        client_ids = list(client_explanations.keys())
        selected_clients = np.random.choice(client_ids, max_clients, replace=False)
        client_explanations = {
            client_id: client_explanations[client_id]
            for client_id in selected_clients
        }
    
    # Extract feature importance for each client
    client_features = {}
    all_features = set()
    
    for client_id, explanation_data in client_explanations.items():
        if ('summary' in explanation_data and 
            explanation_type in explanation_data['summary']):
            
            feature_importance = explanation_data['summary'][explanation_type]
            
            # Sort features by importance and take top N
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:max_features]
            
            client_features[client_id] = sorted_features
            all_features.update([item[0] for item in sorted_features])
    
    # If no valid explanations found
    if not client_features:
        logger.warning(f"No valid {explanation_type} explanations found for clients")
        return
    
    # Get union of top features across all clients
    all_features = list(all_features)
    all_features.sort()  # Sort alphabetically for consistency
    
    # Limit to max_features if needed
    if len(all_features) > max_features:
        # Count feature occurrence across clients
        feature_counts = {feature: 0 for feature in all_features}
        for client_id, features in client_features.items():
            for feature, _ in features:
                feature_counts[feature] += 1
        
        # Take most common features
        all_features = sorted(
            feature_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:max_features]
        all_features = [item[0] for item in all_features]
    
    # Create DataFrame for visualization
    data = []
    for client_id, features in client_features.items():
        # Convert to dict for easier lookup
        feature_dict = dict(features)
        
        for feature in all_features:
            importance = feature_dict.get(feature, 0.0)
            data.append({
                'client_id': client_id,
                'feature': feature,
                'importance': importance
            })
    
    df = pd.DataFrame(data)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create grouped bar plot
    sns.barplot(
        x='feature',
        y='importance',
        hue='client_id',
        data=df,
        palette=CLIENT_COLORS
    )
    
    # Add title and labels
    plt.title(f"Top Features by {explanation_type.upper()} Importance Across Clients - Round {round_num}")
    plt.xlabel("Feature")
    plt.ylabel("Feature Importance")
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.legend(title='Client ID')
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = ensure_results_dir('explanations')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"client_comparison_{explanation_type}_round_{round_num}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved client feature importance comparison to {filepath}")
    
    return filepath

def save_feature_importance_evolution(
    explanation_history: Dict[int, Dict],
    feature_names: List[str],
    explanation_type: str = 'lime',
    top_k: int = 5,
    output_dir: Optional[str] = None
):
    """
    Create and save a plot of how feature importance evolves over training rounds.
    
    Args:
        explanation_history: Dictionary mapping round numbers to explanation data
        feature_names: List of feature names to track
        explanation_type: Type of explanation ('lime' or 'shap')
        top_k: Number of top features to track if feature_names is not provided
        output_dir: Directory to save plot
    """
    logger.info("Creating feature importance evolution plot")
    
    # If specific features not provided, find top-k features from the last round
    if not feature_names:
        # Get latest round
        latest_round = max(explanation_history.keys())
        latest_explanation = explanation_history[latest_round]
        
        if ('explanations' in latest_explanation and 
            explanation_type in latest_explanation['explanations']):
            
            feature_importance = latest_explanation['explanations'][explanation_type]
            
            # Sort features by importance and take top N
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:top_k]
            
            feature_names = [item[0] for item in sorted_features]
    
    # Extract importance values for each feature over rounds
    rounds = sorted(explanation_history.keys())
    feature_values = {feature: [] for feature in feature_names}
    
    for round_num in rounds:
        explanation_data = explanation_history[round_num]
        
        if ('explanations' in explanation_data and 
            explanation_type in explanation_data['explanations']):
            
            feature_importance = explanation_data['explanations'][explanation_type]
            
            for feature in feature_names:
                value = feature_importance.get(feature, 0.0)
                feature_values[feature].append(value)
        else:
            # Fill with zeros if no data for this round
            for feature in feature_names:
                feature_values[feature].append(0.0)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot importance for each feature over rounds
    for i, feature in enumerate(feature_names):
        plt.plot(
            rounds,
            feature_values[feature],
            marker='o',
            linewidth=2,
            label=feature,
            color=plt.cm.tab10(i % 10)  # Use tab10 colormap for distinct colors
        )
    
    # Add title and labels
    plt.title(f"Evolution of {explanation_type.upper()} Feature Importance Over Training Rounds")
    plt.xlabel("Training Round")
    plt.ylabel("Feature Importance")
    
    # Add legend
    plt.legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set x-axis ticks to round numbers
    plt.xticks(rounds)
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = ensure_results_dir('explanations')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{explanation_type}_evolution_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved feature importance evolution plot to {filepath}")
    
    return filepath

def save_client_agreement_heatmap(
    client_agreement: Dict[str, float],
    round_num: int,
    output_dir: Optional[str] = None
):
    """
    Create and save a heatmap of client agreement on feature importance.
    
    Args:
        client_agreement: Dictionary mapping client pairs to agreement scores
        round_num: Training round number
        output_dir: Directory to save plot
    """
    logger.info(f"Creating client agreement heatmap for round {round_num}")
    
    # Extract client IDs from pair keys
    client_pairs = list(client_agreement.keys())
    agreement_values = list(client_agreement.values())
    
    # Extract unique client IDs
    unique_clients = set()
    for pair in client_pairs:
        clients = pair.split('_vs_')
        unique_clients.update(clients)
    
    unique_clients = sorted(list(unique_clients))
    
    # Create agreement matrix
    agreement_matrix = np.ones((len(unique_clients), len(unique_clients)))
    
    # Fill matrix with agreement values
    for pair, value in client_agreement.items():
        client_a, client_b = pair.split('_vs_')
        i = unique_clients.index(client_a)
        j = unique_clients.index(client_b)
        agreement_matrix[i, j] = value
        agreement_matrix[j, i] = value  # Matrix is symmetric
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(
        agreement_matrix,
        annot=True,
        cmap='YlGnBu',
        xticklabels=unique_clients,
        yticklabels=unique_clients,
        vmin=0.0,
        vmax=1.0
    )
    
    # Add title
    plt.title(f"Client Agreement on Feature Importance - Round {round_num}")
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = ensure_results_dir('explanations')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"client_agreement_round_{round_num}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved client agreement heatmap to {filepath}")
    
    return filepath

def save_shap_summary_plot(
    global_explanation: Dict,
    feature_names: List[str],
    round_num: int,
    max_features: int = 15,
    output_dir: Optional[str] = None
):
    """
    Create and save a SHAP summary plot for global explanations.
    
    Args:
        global_explanation: Global explanation data
        feature_names: List of feature names
        round_num: Training round number
        max_features: Maximum number of features to display
        output_dir: Directory to save plot
    """
    try:
        # Check if shap is available
        import shap
    except ImportError:
        logger.error("SHAP package not found, cannot create SHAP summary plot")
        return
    
    logger.info(f"Creating SHAP summary plot for round {round_num}")
    
    # Check if SHAP explanations are available
    if ('explanations' not in global_explanation or 
        'shap' not in global_explanation['explanations']):
        logger.warning("No SHAP explanations found")
        return
    
    # Extract SHAP importance values
    shap_importance = global_explanation['explanations']['shap']
    
    # Sort features by importance and take top N
    sorted_features = sorted(
        shap_importance.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:max_features]
    
    # Extract feature names and values
    features = [item[0] for item in sorted_features]
    importance = [item[1] for item in sorted_features]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create violin plot (similar to SHAP summary plot)
    pos = plt.violinplot(
        [importance],
        showmeans=True,
        showmedians=False,
        showextrema=True
    )
    
    # Add feature names as y-tick labels
    plt.yticks(range(1, len(features) + 1), features)
    
    # Add title and labels
    plt.title(f"SHAP Feature Importance Summary - Round {round_num}")
    plt.xlabel("SHAP Value (Impact on Model Output)")
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = ensure_results_dir('explanations')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"shap_summary_round_{round_num}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved SHAP summary plot to {filepath}")
    
    return filepath

def save_privacy_impact_plot(
    privacy_comparison: Dict[str, Dict],
    metric_name: str = 'accuracy',
    output_dir: Optional[str] = None
):
    """
    Create and save a plot showing the impact of privacy settings on model performance.
    
    Args:
        privacy_comparison: Dictionary mapping privacy levels to performance metrics
        metric_name: Performance metric to compare
        output_dir: Directory to save plot
    """
    logger.info(f"Creating privacy impact plot for {metric_name}")
    
    # Extract privacy levels and corresponding metric values
    privacy_levels = []
    metric_values = []
    
    for level, metrics in privacy_comparison.items():
        if metric_name in metrics:
            privacy_levels.append(level)
            metric_values.append(metrics[metric_name])
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    bars = plt.bar(
        privacy_levels,
        metric_values,
        color='skyblue',
        alpha=0.8
    )
    
    # Add metric values above each bar
    for i, value in enumerate(metric_values):
        plt.text(
            i,
            value + 0.01,
            f"{value:.4f}",
            ha='center'
        )
    
    # Add title and labels
    plt.title(f"Impact of Privacy Settings on {metric_name.capitalize()}")
    plt.xlabel("Privacy Level")
    plt.ylabel(metric_name.capitalize())
    
    plt.ylim(0, max(metric_values) * 1.1)  # Set y limit to leave space for text
    
    plt.tight_layout()
    
    # Save plot
    if output_dir is None:
        output_dir = ensure_results_dir('privacy')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"privacy_impact_{metric_name}_{timestamp}.png"
    filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved privacy impact plot to {filepath}")
    
    return filepath

def save_explanation_dashboard(
    explanation_history: Dict[int, Dict],
    output_dir: Optional[str] = None
):
    """
    Create and save a comprehensive dashboard of explanations.
    
    Args:
        explanation_history: Dictionary mapping round numbers to explanation data
        output_dir: Directory to save dashboard
    """
    logger.info("Creating explanation dashboard")
    
    # Ensure output directory exists
    if output_dir is None:
        output_dir = ensure_results_dir('dashboard')
    
    # Save json data for later use
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_filename = f"explanation_history_{timestamp}.json"
    json_filepath = os.path.join(output_dir, json_filename)
    
    with open(json_filepath, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for round_num, data in explanation_history.items():
            serializable_history[str(round_num)] = json.loads(
                json.dumps(data, default=lambda o: float(o) if isinstance(o, np.float32) else o)
            )
        
        json.dump(serializable_history, f, indent=2)
    
    # Get latest round number
    latest_round = max(explanation_history.keys())
    
    # Create visualizations for the latest round
    filepaths = []
    
    # Feature importance plots
    lime_plot = save_feature_importance_plot(
        explanation_history[latest_round],
        latest_round,
        explanation_type='lime',
        output_dir=output_dir
    )
    filepaths.append(lime_plot)
    
    shap_plot = save_feature_importance_plot(
        explanation_history[latest_round],
        latest_round,
        explanation_type='shap',
        output_dir=output_dir
    )
    filepaths.append(shap_plot)
    
    # Evolution plot for a few top features
    evolution_plot = save_feature_importance_evolution(
        explanation_history,
        feature_names=[],  # Auto-detect top features
        explanation_type='lime',
        output_dir=output_dir
    )
    filepaths.append(evolution_plot)
    
    # Create HTML dashboard
    html_filename = f"explanation_dashboard_{timestamp}.html"
    html_filepath = os.path.join(output_dir, html_filename)
    
    with open(html_filepath, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Federated XAI Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                h1, h2 {{ color: #333; }}
                .dashboard {{ display: flex; flex-direction: column; gap: 20px; }}
                .row {{ display: flex; flex-wrap: wrap; gap: 20px; }}
                .card {{ border: 1px solid #ddd; border-radius: 5px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                .card-full {{ width: 100%; }}
                .card-half {{ width: calc(50% - 20px); }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Federated XAI Dashboard</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="dashboard">
                <div class="row">
                    <div class="card card-half">
                        <h2>LIME Feature Importance (Round {latest_round})</h2>
                        <img src="{os.path.basename(lime_plot)}" alt="LIME Feature Importance">
                    </div>
                    <div class="card card-half">
                        <h2>SHAP Feature Importance (Round {latest_round})</h2>
                        <img src="{os.path.basename(shap_plot)}" alt="SHAP Feature Importance">
                    </div>
                </div>
                
                <div class="row">
                    <div class="card card-full">
                        <h2>Feature Importance Evolution</h2>
                        <img src="{os.path.basename(evolution_plot)}" alt="Feature Importance Evolution">
                    </div>
                </div>
                
                <div class="row">
                    <div class="card card-full">
                        <h2>Summary</h2>
                        <p>Training rounds: {len(explanation_history)}</p>
                        <p>Latest round: {latest_round}</p>
                        <p>Data exported to: {os.path.basename(json_filepath)}</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """)
    
    logger.info(f"Saved explanation dashboard to {html_filepath}")
    
    return html_filepath

def create_all_visualizations(
    explanation_history: Dict[int, Dict],
    client_explanations: Optional[Dict[str, Dict]] = None,
    feature_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None
):
    """
    Create all available visualizations for the explanation data.
    
    Args:
        explanation_history: Dictionary mapping round numbers to explanation data
        client_explanations: Dictionary mapping client IDs to explanation data
        feature_names: List of feature names
        output_dir: Directory to save visualizations
    """
    logger.info("Creating all explanation visualizations")
    
    # Ensure output directory exists
    if output_dir is None:
        output_dir = ensure_results_dir('explanations')
    
    filepaths = []
    
    # Get latest round number
    latest_round = max(explanation_history.keys())
    
    # Feature importance plots
    lime_plot = save_feature_importance_plot(
        explanation_history[latest_round],
        latest_round,
        explanation_type='lime',
        output_dir=output_dir
    )
    filepaths.append(lime_plot)
    
    shap_plot = save_feature_importance_plot(
        explanation_history[latest_round],
        latest_round,
        explanation_type='shap',
        output_dir=output_dir
    )
    filepaths.append(shap_plot)
    
    # Evolution plot
    evolution_plot = save_feature_importance_evolution(
        explanation_history,
        feature_names=feature_names,
        explanation_type='lime',
        output_dir=output_dir
    )
    filepaths.append(evolution_plot)
    
    # Client comparison if client data is available
    if client_explanations:
        client_plot = save_client_feature_importance_comparison(
            client_explanations,
            latest_round,
            explanation_type='lime',
            output_dir=output_dir
        )
        filepaths.append(client_plot)
    
    # Dashboard
    dashboard = save_explanation_dashboard(
        explanation_history,
        output_dir=output_dir
    )
    filepaths.append(dashboard)
    
    logger.info(f"Created {len(filepaths)} visualization files")
    
    return filepaths