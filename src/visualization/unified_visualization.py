"""
Unified Visualization System for QML-CML Manuscript
===================================================

Generates main manuscript figures and tables:
- Figure 4: Test set predictions (observed vs predicted)
- Figure 5: Experimental sample predictions
- Table 1: Model performance summary
- Tables S4-S6: Detailed metrics

Author: [Your Name]
Date: March 2026
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

CLASSES = ["Carbohydrate", "Lignin", "Lipid", "Protein", "Others"]
TRAIN_SIZES = [200, 400, 600]

COLORS = {
    "QuantumKernel": "#FF6B6B",
    "VQR": "#4ECDC4",
    "RandomForest": "#45B7D1",
    "SVR": "#FFA07A",
    "Lasso": "#98D8C8",
    "VotingRegressor": "#6C5CE7"
}

MARKERS = {
    "QuantumKernel": "o",
    "VQR": "s",
    "RandomForest": "^",
    "SVR": "D",
    "Lasso": "v",
    "VotingRegressor": "P"
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
    else:
        print(f"⚠️  File not found: {path}")
        return None

# ============================================================================
# FIGURE 4: TEST SET PREDICTIONS
# ============================================================================

def create_figure_4(results_dir="results", output_dir="figures/main", train_size=600):
    """
    Figure 4: Observed vs Predicted CCS for test set.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    models_to_plot = ["QuantumKernel", "VQR", "RandomForest", "SVR", "Lasso", "VotingRegressor"]
    
    for idx, class_name in enumerate(CLASSES):
        ax = axes[idx]
        
        all_obs = []
        all_pred = []
        
        for model_name in models_to_plot:
            df = load_prediction_data(results_dir, class_name, model_name, train_size)
            
            if df is not None:
                obs = df["Observed_CCS"].values
                pred = df["Predicted_CCS"].values
                
                ax.scatter(
                    obs, pred,
                    c=COLORS.get(model_name, "#999999"),
                    marker=MARKERS.get(model_name, "o"),
                    alpha=0.6,
                    s=60,
                    label=model_name,
                    edgecolors='white',
                    linewidth=0.7
                )
                
                all_obs.extend(obs)
                all_pred.extend(pred)
        
        # Perfect prediction line
        if all_obs:
            lims = [
                min(min(all_obs), min(all_pred)) * 0.95,
                max(max(all_obs), max(all_pred)) * 1.05
            ]
            ax.plot(lims, lims, 'k--', alpha=0.5, zorder=0, linewidth=2, label='y = x')
        
        ax.set_xlabel('Observed CCS (Ų)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted CCS (Ų)', fontsize=12, fontweight='bold')
        ax.set_title(f'{class_name}', fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.legend(loc='upper left', fontsize=9, framealpha=0.95, ncol=2)
        
        ax.text(0.98, 0.02, f'n_train = {train_size}', 
                transform=ax.transAxes, 
                ha='right', va='bottom',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.suptitle('Figure 4: Test Set Predictions - Observed vs Predicted CCS', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save
    plt.savefig(f'{output_dir}/Figure_4_test_predictions_n{train_size}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/Figure_4_test_predictions_n{train_size}.pdf', bbox_inches='tight')
    print(f"✅ Saved Figure 4: {output_dir}/Figure_4_test_predictions_n{train_size}.pdf")
    plt.close()

# ============================================================================
# FIGURE 5: EXPERIMENTAL PREDICTIONS
# ============================================================================

def create_figure_5(results_dir="results", output_dir="figures/main"):
    """
    Figure 5: Predictions for experimental samples.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, class_name in enumerate(CLASSES):
        ax = axes[idx]
        
        predictions_dict = {}
        
        # Quantum Kernel
        qk_file = f"{results_dir}/experimental_predictions_quantum/{class_name}_experimental_with_qkernel_preds_600.csv"
        if os.path.exists(qk_file):
            df = pd.read_csv(qk_file)
            if "Predicted_CCS_QuantumKernel_train600" in df.columns:
                predictions_dict["QuantumKernel"] = df["Predicted_CCS_QuantumKernel_train600"].values
        
        # VQR
        vqr_file = f"{results_dir}/experimental_predictions_vqr/{class_name}_experimental_with_vqr_preds_600.csv"
        if os.path.exists(vqr_file):
            df = pd.read_csv(vqr_file)
            if "Predicted_CCS_VQR_train600" in df.columns:
                predictions_dict["VQR"] = df["Predicted_CCS_VQR_train600"].values
        
        # Classical models
        for model_name in ["RandomForest", "SVR", "Lasso", "VotingRegressor"]:
            classical_file = f"{results_dir}/experimental_predictions/{class_name}_experimental_with_preds_{model_name}_600.csv"
            if os.path.exists(classical_file):
                df = pd.read_csv(classical_file)
                col_name = f"Predicted_CCS_{model_name}_train600"
                if col_name in df.columns:
                    predictions_dict[model_name] = df[col_name].values
        
        # Plot
        if predictions_dict:
            n_samples = len(next(iter(predictions_dict.values())))
            sample_ids = np.arange(n_samples)
            
            for model_name, preds in predictions_dict.items():
                ax.scatter(
                    sample_ids, preds,
                    c=COLORS.get(model_name, "#999999"),
                    marker=MARKERS.get(model_name, "o"),
                    alpha=0.7,
                    s=80,
                    label=model_name,
                    edgecolors='white',
                    linewidth=0.8
                )
            
            ax.set_xlabel('Experimental Sample Index', fontsize=12, fontweight='bold')
            ax.set_ylabel('Predicted CCS (Ų)', fontsize=12, fontweight='bold')
            ax.set_title(f'{class_name} (n={n_samples} samples)', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=9, framealpha=0.95, ncol=2)
            ax.grid(True, alpha=0.3, linestyle='--')
        else:
            ax.text(0.5, 0.5, 'No experimental data available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, style='italic')
            ax.set_xticks([])
            ax.set_yticks([])
    
    fig.delaxes(axes[5])
    
    plt.suptitle('Figure 5: Experimental Sample Predictions', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/Figure_5_experimental_predictions.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/Figure_5_experimental_predictions.pdf', bbox_inches='tight')
    print(f"✅ Saved Figure 5: {output_dir}/Figure_5_experimental_predictions.pdf")
    plt.close()

# ============================================================================
# TABLE 1: MODEL PERFORMANCE SUMMARY
# ============================================================================

def create_table_1(results_dir="results", output_dir="tables"):
    """
    Table 1: Average model performance across classes.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results
    all_results = []
    
    # Quantum Kernel
    qk_file = f"{results_dir}/quantum_kernel_results_all_classes.csv"
    if os.path.exists(qk_file):
        df = pd.read_csv(qk_file)
        df['Model'] = 'QuantumKernel'
        all_results.append(df)
    
    # VQR
    vqr_file = f"{results_dir}/vqr_results_all_classes.csv"
    if os.path.exists(vqr_file):
        df = pd.read_csv(vqr_file)
        df['Model'] = 'VQR'
        all_results.append(df)
    
    # Classical
    classical_file = f"{results_dir}/classical_ml_size_scaling_results.csv"
    if os.path.exists(classical_file):
        all_results.append(pd.read_csv(classical_file))
    
    if not all_results:
        print("⚠️  No results files found!")
        return
    
    # Combine
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Calculate summary statistics
    summary = combined_df.groupby('Model').agg({
        'R2': 'mean',
        'MAE': 'mean',
        'RMSE': 'mean'
    }).round(3).reset_index()
    
    summary = summary.sort_values('R2', ascending=False)
    
    # Add model type column
    summary['Model_Type'] = summary['Model'].apply(
        lambda x: 'Quantum' if x in ['QuantumKernel', 'VQR'] else 'Classical'
    )
    
    # Reorder columns
    summary = summary[['Model_Type', 'Model', 'R2', 'MAE', 'RMSE']]
    
    # Save
    summary.to_csv(f'{output_dir}/Table_1_model_performance_summary.csv', index=False)
    print(f"✅ Saved Table 1: {output_dir}/Table_1_model_performance_summary.csv")
    
    print("\nTable 1 Preview:")
    print(summary.to_string(index=False))
    
    return summary

# ============================================================================
# TABLE S4: COMPREHENSIVE METRICS
# ============================================================================

def create_table_s4(results_dir="results", output_dir="tables"):
    """
    Table S4: Comprehensive prediction metrics by class and model.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for class_name in CLASSES:
        for model_name in ["QuantumKernel", "VQR", "RandomForest", "SVR", "Lasso", "VotingRegressor"]:
            df = load_prediction_data(results_dir, class_name, model_name, 600)
            
            if df is not None:
                obs = df["Observed_CCS"].values
                pred = df["Predicted_CCS"].values
                residuals = pred - obs
                
                # Calculate metrics
                mae = np.mean(np.abs(residuals))
                rmse = np.sqrt(np.mean(residuals**2))
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((obs - np.mean(obs))**2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
                
                mape = np.mean(np.abs(residuals / obs)) * 100
                max_error = np.max(np.abs(residuals))
                
                results.append({
                    "Class": class_name,
                    "Model": model_name,
                    "Model_Type": "Quantum" if model_name in ["QuantumKernel", "VQR"] else "Classical",
                    "N_Test": len(obs),
                    "R2": round(r2, 4),
                    "MAE": round(mae, 2),
                    "RMSE": round(rmse, 2),
                    "MAPE_%": round(mape, 1),
                    "Max_Error": round(max_error, 2),
                })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/Table_S4_comprehensive_prediction_metrics.csv", index=False)
    print(f"✅ Saved Table S4: {output_dir}/Table_S4_comprehensive_prediction_metrics.csv")
    
    return results_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def generate_all_figures_and_tables(results_dir="results", 
                                     figures_dir="figures", 
                                     tables_dir="tables"):
    """
    Generate all main figures and tables.
    """
    
    print("\n" + "="*80)
    print("GENERATING FIGURES AND TABLES")
    print("="*80 + "\n")
    
    # Create directories
    os.makedirs(f"{figures_dir}/main", exist_ok=True)
    os.makedirs(f"{figures_dir}/supplementary", exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)
    
    # Main figures
    print("Generating main figures...")
    try:
        create_figure_4(results_dir, f"{figures_dir}/main")
    except Exception as e:
        print(f"⚠️  Error creating Figure 4: {e}")
    
    try:
        create_figure_5(results_dir, f"{figures_dir}/main")
    except Exception as e:
        print(f"⚠️  Error creating Figure 5: {e}")
    
    # Tables
    print("\nGenerating tables...")
    try:
        create_table_1(results_dir, tables_dir)
    except Exception as e:
        print(f"⚠️  Error creating Table 1: {e}")
    
    try:
        create_table_s4(results_dir, tables_dir)
    except Exception as e:
        print(f"⚠️  Error creating Table S4: {e}")
    
    print("\n" + "="*80)
    print("✅ FIGURE AND TABLE GENERATION COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to:")
    print(f"  📊 {figures_dir}/main/")
    print(f"  📋 {tables_dir}/")
    print("="*80 + "\n")

if __name__ == "__main__":
    generate_all_figures_and_tables()
