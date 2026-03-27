"""
Advanced Comparison Visualizations for Supplementary Information
================================================================

Generates supplementary figures:
- Figure S4: Residual plots
- Additional comparison plots

Author: [Your Name]
Date: March 2026
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

CLASSES = ["Carbohydrate", "Lignin", "Lipid", "Protein", "Others"]

COLORS = {
    "QuantumKernel": "#FF6B6B",
    "VQR": "#4ECDC4",
    "RandomForest": "#45B7D1"
}

MARKERS = {
    "QuantumKernel": "o",
    "VQR": "s",
    "RandomForest": "^"
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_prediction_data(results_dir, class_name, model_name, train_size):
    """Load prediction data from the correct directory."""
    
    if model_name == "QuantumKernel":
        path = f"{results_dir}/pred_vs_obs_quantum/pred_vs_obs_quantumkernel_{class_name}_{train_size}.csv"
    elif model_name == "VQR":
        path = f"{results_dir}/pred_vs_obs_vqr/pred_vs_obs_vqr_{class_name}_{train_size}.csv"
    else:
        path = f"{results_dir}/pred_vs_obs/pred_vs_obs_{class_name}_{model_name}_{train_size}.csv"
    
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ============================================================================
# FIGURE S4: RESIDUAL PLOTS
# ============================================================================

def create_figure_s4(results_dir="results", output_dir="figures/supplementary", train_size=600):
    """
    Figure S4: Residual analysis of test set predictions.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    models_to_plot = ["QuantumKernel", "VQR", "RandomForest"]
    
    for idx, class_name in enumerate(CLASSES):
        ax = axes[idx]
        
        for model_name in models_to_plot:
            df = load_prediction_data(results_dir, class_name, model_name, train_size)
            
            if df is not None:
                obs = df["Observed_CCS"].values
                pred = df["Predicted_CCS"].values
                residuals = pred - obs
                
                ax.scatter(
                    obs, residuals,
                    c=COLORS[model_name],
                    marker=MARKERS[model_name],
                    alpha=0.6,
                    s=50,
                    label=model_name,
                    edgecolors='white',
                    linewidth=0.5
                )
        
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=2)
        ax.set_xlabel('Observed CCS (Ų)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Residual (Predicted - Observed, Ų)', fontsize=12, fontweight='bold')
        ax.set_title(f'{class_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=9, framealpha=0.9)
    
    fig.delaxes(axes[5])
    
    plt.suptitle('Figure S4: Residual Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/Figure_S4_residual_plots_n{train_size}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/Figure_S4_residual_plots_n{train_size}.pdf', bbox_inches='tight')
    print(f"✅ Saved Figure S4: {output_dir}/Figure_S4_residual_plots_n{train_size}.pdf")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_supplementary_figures(results_dir="results", output_dir="figures/supplementary"):
    """
    Generate all supplementary figures.
    """
    
    print("\n" + "="*80)
    print("GENERATING SUPPLEMENTARY FIGURES")
    print("="*80 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        create_figure_s4(results_dir, output_dir)
    except Exception as e:
        print(f"⚠️  Error creating Figure S4: {e}")
    
    print("\n" + "="*80)
    print("✅ SUPPLEMENTARY FIGURES COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {output_dir}/")
    print("="*80 + "\n")

if __name__ == "__main__":
    generate_supplementary_figures()
