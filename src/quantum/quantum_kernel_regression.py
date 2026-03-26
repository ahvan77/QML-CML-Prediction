"""
Quantum Kernel Regression for CCS Prediction
============================================

Main implementation for manuscript Section 2.3.

Configuration (Optimized):
- Feature map: ZZFeatureMap with 5 qubits, reps=1
- Kernel: FidelityQuantumKernel
- Regressor: KernelRidge with class-specific α (CV-tuned)
- Training sizes: 200, 400, 600 samples per class
- Classes: Carbohydrate, Lignin, Lipid, Protein, Others

Author: Sadollah Ebrahimi
Date: March 2026
"""

import pandas as pd
import numpy as np
import os
import argparse
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

np.random.seed(42)

# Molecular descriptors (Z-score normalized)
FEATURE_COLS = ["A_scaled", "B_scaled", "C_scaled", "D_scaled", "E_scaled"]

# Training set sizes to evaluate
TRAIN_SIZES = [200, 400, 600]

# Molecular classes
CLASSES = ["Carbohydrate", "Lignin", "Lipid", "Protein", "Others"]

# Quantum circuit configuration (optimized from Section 3.1)
REPS = 1  # ZZFeatureMap repetitions (1 is optimal)

# Class-specific initial α values for CV tuning (from Table S2)
ALPHA_INIT = {
    "Carbohydrate": 0.03,
    "Lignin": 1.0,
    "Lipid": 0.01,
    "Protein": 0.32,
    "Others": 0.1,
}

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def run_quantum_kernel_regression(
    data_dir="Data/features",
    experimental_dir="Data/experimental",
    output_dir="results",
    classes=None,
    train_sizes=None,
    verbose=True
):
    """
    Run quantum kernel regression experiments for CCS prediction.
    
    Parameters
    ----------
    data_dir : str
        Path to directory containing train/test CSV files organized by class
    experimental_dir : str
        Path to experimental validation data
    output_dir : str
        Where to save results
    classes : list, optional
        List of molecular classes to process (default: all 5 classes)
    train_sizes : list, optional
        Training set sizes to evaluate (default: [200, 400, 600])
    verbose : bool
        Print progress information
        
    Returns
    -------
    results_df : DataFrame
        Summary results with MAE, RMSE, R² for each class/size combination
    """
    
    if classes is None:
        classes = CLASSES
    if train_sizes is None:
        train_sizes = TRAIN_SIZES
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/pred_vs_obs_quantum", exist_ok=True)
    os.makedirs(f"{output_dir}/experimental_predictions_quantum", exist_ok=True)
    
    results = []
    
    if verbose:
        print("\n" + "="*80)
        print("QUANTUM KERNEL REGRESSION - CCS PREDICTION")
        print("="*80)
        print(f"Configuration:")
        print(f"  Feature map: ZZFeatureMap (5 qubits, reps={REPS})")
        print(f"  Kernel: FidelityQuantumKernel")
        print(f"  Regressor: KernelRidge (α tuned via 5-fold CV)")
        print(f"  Classes: {len(classes)}")
        print(f"  Training sizes: {train_sizes}")
        print("="*80 + "\n")
    
    # Process each molecular class
    for class_name in classes:
        if verbose:
            print(f"\n{'='*80}")
            print(f"CLASS: {class_name}")
            print(f"{'='*80}")
        
        # Load training and test data
        class_dir = os.path.join(data_dir, class_name)
        train_file = os.path.join(class_dir, "train.csv")
        test_file = os.path.join(class_dir, "test.csv")
        
        if not (os.path.exists(train_file) and os.path.exists(test_file)):
            print(f"⚠️  Skipping {class_name}: Missing train or test file")
            continue
        
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        
        X_train_full = df_train[FEATURE_COLS].values
        y_train_full = df_train["CCS"].values
        X_test = df_test[FEATURE_COLS].values
        y_test = df_test["CCS"].values
        
        if verbose:
            print(f"Data loaded: {len(X_train_full)} training, {len(X_test)} test samples")
        
        # Get class-specific initial α for CV
        alpha_init = ALPHA_INIT.get(class_name, 0.1)
        
        # Process each training size
        for n_samples in train_sizes:
            if len(X_train_full) < n_samples:
                print(f"⚠️  {class_name}: Insufficient samples for n={n_samples} (have {len(X_train_full)})")
                continue
            
            if verbose:
                print(f"\n--- Training size: {n_samples} ---")
            
            # Sample training subset with fixed random state
            X_train, _, y_train, _ = train_test_split(
                X_train_full, y_train_full,
                train_size=n_samples,
                random_state=42
            )
            
            # Normalize targets (z-score normalization for stability)
            y_mean = y_train.mean()
            y_std = y_train.std() if y_train.std() > 0 else 1.0
            y_train_norm = (y_train - y_mean) / y_std
            
            # Create quantum feature map and kernel
           # n_qubits = len(FEATURE_COLS) 
            feature_map = ZZFeatureMap(
               feature_dimension=5,
               reps=REPS,           # or your existing args
               entanglement="full"  # etc., same signature as before
            )
            quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)
            
            try:
                # Compute kernel matrices
                if verbose:
                    print(f"Computing quantum kernel matrices...")
                
                K_train = quantum_kernel.evaluate(X_train)
                K_test = quantum_kernel.evaluate(X_test, X_train)
                
                if verbose:
                    print(f"✓ Kernel matrices computed: K_train {K_train.shape}, K_test {K_test.shape}")
                
            except (MemoryError, Exception) as e:
                print(f"❌ Kernel computation failed for {class_name} (n={n_samples}): {e}")
                continue
            
            # Tune regularization parameter α via cross-validation
            # Search around class-specific initial value
            param_grid = {
                "alpha": np.logspace(
                    np.log10(alpha_init) - 1,  # 10x smaller
                    np.log10(alpha_init) + 1,  # 10x larger
                    5  # 5 values logarithmically spaced
                )
            }
            
            if verbose:
                print(f"Tuning α via 5-fold CV (initial α={alpha_init:.3g})...")
            
            kr = KernelRidge(kernel="precomputed")
            grid_search = GridSearchCV(
                kr, param_grid,
                cv=5,
                scoring="neg_mean_squared_error",
                n_jobs=-1
            )
            grid_search.fit(K_train, y_train_norm)
            
            best_alpha = grid_search.best_params_["alpha"]
            
            if verbose:
                print(f"✓ Best α: {best_alpha:.3g}")
            
            # Train final model with best α
            regressor = KernelRidge(alpha=best_alpha, kernel="precomputed")
            regressor.fit(K_train, y_train_norm)
            
            # Predict on test set and denormalize
            preds_norm = regressor.predict(K_test)
            preds = preds_norm * y_std + y_mean
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, preds)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            r2 = r2_score(y_test, preds)
            
            if verbose:
                print(f"✓ Test set performance:")
                print(f"  MAE:  {mae:.2f} Ų")
                print(f"  RMSE: {rmse:.2f} Ų")
                print(f"  R²:   {r2:.3f}")
            
            # Store results
            results.append({
                "Class": class_name,
                "TrainSize": n_samples,
                "Model": "QuantumKernel",
                "Reps": REPS,
                "BestAlpha": best_alpha,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "N_Train": len(y_train),
                "N_Test": len(y_test)
            })
            
            # Save per-sample predictions
            pred_df = pd.DataFrame({
                "Class": class_name,
                "TrainSize": n_samples,
                "Model": "QuantumKernel",
                "Reps": REPS,
                "BestAlpha": best_alpha,
                "Observed_CCS": y_test,
                "Predicted_CCS": preds
            })
            pred_file = os.path.join(
                output_dir,
                "pred_vs_obs_quantum",
                f"pred_vs_obs_quantumkernel_{class_name}_{n_samples}.csv"
            )
            pred_df.to_csv(pred_file, index=False)
            
            # Predict on experimental samples if available
            exp_file = os.path.join(experimental_dir, f"{class_name}_experimental.csv")
            if os.path.exists(exp_file):
                try:
                    exp_df = pd.read_csv(exp_file)
                    
                    if all(col in exp_df.columns for col in FEATURE_COLS):
                        X_exp = exp_df[FEATURE_COLS].values
                        
                        # Compute kernel between experimental and training samples
                        K_exp = quantum_kernel.evaluate(X_exp, X_train)
                        
                        # Predict and denormalize
                        exp_preds_norm = regressor.predict(K_exp)
                        exp_preds = exp_preds_norm * y_std + y_mean
                        
                        # Add predictions to experimental dataframe
                        exp_df[f"Predicted_CCS_QuantumKernel_train{n_samples}"] = exp_preds
                        
                        # Save
                        exp_out = os.path.join(
                            output_dir,
                            "experimental_predictions_quantum",
                            f"{class_name}_experimental_with_qkernel_preds_{n_samples}.csv"
                        )
                        exp_df.to_csv(exp_out, index=False)
                        
                        if verbose:
                            print(f"✓ Experimental predictions saved ({len(X_exp)} samples)")
                    else:
                        print(f"⚠️  Experimental file missing required feature columns")
                        
                except Exception as e:
                    print(f"⚠️  Experimental prediction failed: {e}")
    
    # Save overall summary
    results_df = pd.DataFrame(results)
    summary_file = os.path.join(output_dir, "quantum_kernel_results_all_classes.csv")
    results_df.to_csv(summary_file, index=False)
    
    if verbose:
        print(f"\n{'='*80}")
        print("✅ QUANTUM KERNEL REGRESSION COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {output_dir}/")
        print(f"  - Summary: quantum_kernel_results_all_classes.csv")
        print(f"  - Per-sample predictions: pred_vs_obs_quantum/")
        print(f"  - Experimental predictions: experimental_predictions_quantum/")
        print(f"\nOverall performance (average R² by class):")
        print(results_df.groupby('Class')['R2'].mean().to_string())
        print("="*80 + "\n")
    
    return results_df


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Quantum Kernel Regression for CCS Prediction"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="Data/features",
        help="Directory containing train/test CSV files by class"
    )
    parser.add_argument(
        "--experimental-dir",
        type=str,
        default="Data/experimental",
        help="Directory containing experimental validation data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Molecular classes to process (default: all)"
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Training set sizes (default: 200 400 600)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Run quantum kernel regression
    results = run_quantum_kernel_regression(
        data_dir=args.data_dir,
        experimental_dir=args.experimental_dir,
        output_dir=args.output_dir,
        classes=args.classes,
        train_sizes=args.train_sizes,
        verbose=not args.quiet
    )
    
    return results


if __name__ == "__main__":
    results = main()
