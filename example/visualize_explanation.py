# examples/visualize_explanations.py
"""
Example script demonstrating how to use the visualization module
to create explanation visualizations.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import visualization module
from utils.fed_xai_visualization import (
    save_feature_importance_plot,
    save_client_feature_importance_comparison,
    save_feature_importance_evolution,
    save_client_agreement_heatmap,
    save_explanation_dashboard,
    create_all_visualizations
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('visualize_explanations')

def load_explanation_data():
    """
    Load explanation data from previous run or create dummy data.
    
    Returns:
        Tuple of (explanation_history, client_explanations)
    """
    # Check if explanation data exists from previous run
    results_dir = os.path.join(os.getcwd(), 'results')
    explanation_dir = os.path.join(results_dir, 'explanations')
    
    if os.path.exists(explanation_dir):
        # Look for JSON files
        json_files = [f for f in os.listdir(explanation_dir) if f.endswith('.json')]
        
        if json_files:
            import json
            latest_file = sorted(json_files)[-1]
            file_path = os.path.join(explanation_dir, latest_file)
            
            logger.info(f"Loading explanation data from {file_path}")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Convert string keys back to integers
                explanation_history = {
                    int(k): v for k, v in data.items()
                }
                
                return explanation_history, None
    
    # Create dummy data if no previous data found
    logger.info("Creating dummy explanation data")
    
    # Create list of feature names
    feature_names = [f'feature_{i}' for i in range(20)]
    
    # Create dummy explanation history
    explanation_history = {}
    for round_num in range(1, 11):  # 10 rounds
        # Create random feature importance
        lime_importance = {
            feature: np.random.normal(0, 1) for feature in feature_names
        }
        
        shap_importance = {
            feature: np.random.normal(0, 1) for feature in feature_names
        }
        
        explanation_history[round_num] = {
            'round': round_num,
            'explanations': {
                'lime': lime_importance,
                'shap': shap_importance
            },
            'client_count': 3,
            'client_ids': ['client_1', 'client_2', 'client_3']
        }
    
    # Create dummy client explanations
    client_explanations = {}
    for client_id in ['client_1', 'client_2', 'client_3']:
        # Create random feature importance
        lime_importance = {
            feature: np.random.normal(0, 1) for feature in feature_names[:10]
        }
        
        shap_importance = {
            feature: np.random.normal(0, 1) for feature in feature_names[:10]
        }
        
        client_explanations[client_id] = {
            'client_id': client_id,
            'round': 10,  # Latest round
            'summary': {
                'lime': lime_importance,
                'shap': shap_importance
            },
            'metadata': {
                'sample_size': 50,
                'positive_ratio': 0.3,
                'feature_count': 20
            }
        }
    
    # Create dummy client agreement
    client_agreement = {
        'client_1_vs_client_2': 0.75,
        'client_1_vs_client_3': 0.62,
        'client_2_vs_client_3': 0.81
    }
    
    # Add client agreement to latest round
    explanation_history[10]['client_agreement'] = client_agreement
    
    return explanation_history, client_explanations

def main():
    """Run visualization examples"""
    try:
        # Load or create explanation data
        explanation_history, client_explanations = load_explanation_data()
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(os.getcwd(), 'results', f'viz_examples_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Example 1: Create feature importance plot for the latest round
        latest_round = max(explanation_history.keys())
        logger.info(f"Latest round: {latest_round}")
        
        # Create LIME importance plot
        lime_plot = save_feature_importance_plot(
            explanation_history[latest_round],
            latest_round,
            explanation_type='lime',
            output_dir=output_dir
        )
        logger.info(f"Created LIME importance plot: {lime_plot}")
        
        # Example 2: Create client feature importance comparison
        if client_explanations:
            client_plot = save_client_feature_importance_comparison(
                client_explanations,
                latest_round,
                explanation_type='lime',
                output_dir=output_dir
            )
            logger.info(f"Created client comparison plot: {client_plot}")
        
        # Example 3: Create feature importance evolution plot
        # Track the top 5 features from the latest round
        if 'explanations' in explanation_history[latest_round]:
            latest_lime = explanation_history[latest_round]['explanations']['lime']
            top_features = sorted(
                latest_lime.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            top_feature_names = [item[0] for item in top_features]
            
            evolution_plot = save_feature_importance_evolution(
                explanation_history,
                feature_names=top_feature_names,
                explanation_type='lime',
                output_dir=output_dir
            )
            logger.info(f"Created evolution plot: {evolution_plot}")
        
        # Example 4: Create client agreement heatmap
        if ('client_agreement' in explanation_history[latest_round]):
            agreement_plot = save_client_agreement_heatmap(
                explanation_history[latest_round]['client_agreement'],
                latest_round,
                output_dir=output_dir
            )
            logger.info(f"Created client agreement heatmap: {agreement_plot}")
        
        # Example 5: Create comprehensive dashboard
        dashboard = save_explanation_dashboard(
            explanation_history,
            output_dir=output_dir
        )
        logger.info(f"Created explanation dashboard: {dashboard}")
        
        # Example 6: Create all visualizations at once
        all_plots = create_all_visualizations(
            explanation_history,
            client_explanations,
            feature_names=None,  # Auto-detect top features
            output_dir=output_dir
        )
        logger.info(f"Created {len(all_plots)} visualizations")
        
        logger.info(f"All visualizations saved to {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {str(e)}")
        logger.error("Detailed error: ", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("Visualization examples completed successfully")
    else:
        logger.error("Visualization examples failed")
        sys.exit(1)