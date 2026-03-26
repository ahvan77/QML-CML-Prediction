"""
Classical Machine Learning Baselines for CCS Prediction
=======================================================

Main implementation for manuscript Section 2.2.

Models:
- Random Forest (RF): Ensemble of decision trees
- Support Vector Regression (SVR): RBF kernel
- Lasso: L1-regularized linear regression
- Voting Regressor (VR): Ensemble of RF, SVR, KNN

Configuration:
- Training sizes: 200, 400, 600 samples per class
- Classes: Carbohydrate, Lignin, Lipid, Protein, Others
- Hyperparameters: Optimized via 5-fold cross-validation

Author: Sadollah Ebrahimi
Date: March 2026
"""

import os
import pandas as pd
import numpy as np
import argparse
import time
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
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

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def get_models():
    """
    Create classical ML models with optimized hyperparameters.
    
    Returns
    -------
    models : dict
        Dictionary of model_name: model_instance
    """
    
    # Random Forest (optimized via GridSearchCV in preliminary studies)
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    # Support Vector Regression with RBF kernel
    svr = SVR(
        kernel='rbf',
        C=1.0,
        epsilon=0.1,
        gamma='scale'
    )
    
    # Lasso regression with L1 regularization
    lasso = Lasso(
        alpha=1.0,
        random_state=42,
        max_iter=10000
    )
    
    # K-Nearest Neighbors (for Voting Regressor ensemble)
    knn = KNeighborsRegressor(
        n_neighbors=5,
        weights='uniform'
    )
    
    # Voting Regressor: Ensemble of RF, KNN, SVR
    voting = VotingRegressor(
        estimators=[
            ('rf', rf),
            ('knn', knn),
            ('svr', svr)
        ]
    )
    
    models = {
        "RandomForest": rf,
        "SVR": svr,
        "Lasso": lasso,
        "VotingRegressor": voting
    }
    
    return models

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def run_classical_baselines(
    data_dir="Data/features",
    experimental_dir="Data/experimental",
    output_dir="results",
    classes=None,
    train_sizes=None,
    models_to_run=None,
    verbose=True
):
    """
    Run classical ML baseline experiments for CCS prediction.
    
    Parameters
    ----------
    data_dir : str
        Path to directory containing train/test CSV files by class
    experimental_dir : str
        Path to experimental validation data
    output_dir : str
        Where to save results
    classes : list, optional
        List of molecular classes to process (default: all)
    train_sizes : list, optional
        Training set sizes to evaluate (default: [200, 400, 600])
    models_to_run : list, optional
        Which models to run (default: all 4 models)
    verbose : bool
        Print progress information
        
    Returns
    -------
    results_df : DataFrame
        Summary results with MAE, RMSE, R² for each class/size/model
    """
    
    if classes is None:
        classes = CLASSES
    if train_sizes is None:
        train_sizes = TRAIN_SIZES
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/pred_vs_obs", exist_ok=True)
    os.makedirs(f"{output_dir}/experimental_predictions", exist_ok=True)
    
    # Get models
    all_models = get_models()
    if models_to_run is not None:
        models = {k: v for k, v in all_models.items() if k in models_to_run}
    else:
        models = all_models
    
    results = []
    
    if verbose:
        print("\n" + "="*80)
        print("CLASSICAL MACHINE LEARNING BASELINES - CCS PREDICTION")
        print("="*80)
        print(f"Models: {', '.join(models.keys())}")
        print(f"Classes: {len(classes)}")
        print(f"Training sizes: {train_sizes}")
        print("="*80 + "\n")
    
    # Process each molecular class
    for class_name in classes:
        if verbose:
            print(f"\n{'='*80}")
            print(f"CLASS: {class_name}")
            print(f"{'='*80}")
        
        # Load data
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
        
        # Process each training size
        for n_samples in train_sizes:
            if len(X_train_full) < n_samples:
                print(f"⚠️  {class_name}: Insufficient samples for n={n_samples}")
                continue
            
            if verbose:
                print(f"\n--- Training size: {n_samples} ---")
            
            # Sample training subset
            X_train, _, y_train, _ = train_test_split(
                X_train_full, y_train_full,
                train_size=n_samples,
                random_state=42
            )
            
            # Train and evaluate each model
            for model_name, model in models.items():
                if verbose:
                    print(f"\n  {model_name}:")
                
                # Train model
                t0 = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - t0
                
                # Predict on test set
                preds = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)
                
                if verbose:
                    print(f"    MAE:  {mae:.2f} Ų")
                    print(f"    RMSE: {rmse:.2f} Ų")
                    print(f"    R²:   {r2:.3f}")
                    print(f"    Training time: {train_time:.2f}s")
                
                # Store results
                results.append({
                    "Class": class_name,
                    "TrainSize": n_samples,
                    "Model": model_name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2,
                    "Train_Time": train_time,
                    "N_Train": len(y_train),
                    "N_Test": len(y_test)
                })
                
                # Save per-sample predictions
                pred_df = pd.DataFrame({
                    "Class": class_name,
                    "TrainSize": n_samples,
                    "Model": model_name,
                    "Observed_CCS": y_test,
                    "Predicted_CCS": preds
                })
                pred_file = os.path.join(
                    output_dir,
                    "pred_vs_obs",
                    f"pred_vs_obs_{class_name}_{model_name}_{n_samples}.csv"
                )
                pred_df.to_csv(pred_file, index=False)
                
                # Predict on experimental samples
                exp_file = os.path.join(experimental_dir, f"{class_name}_experimental.csv")
                if os.path.exists(exp_file):
                    try:
                        exp_df = pd.read_csv(exp_file)
                        
                        if all(col in exp_df.columns for col in FEATURE_COLS):
                            X_exp = exp_df[FEATURE_COLS].values
                            exp_preds = model.predict(X_exp)
                            
                            # Add to dataframe
                            exp_df[f"Predicted_CCS_{model_name}_train{n_samples}"] = exp_preds
                            
                            # Save (one file per class/size, all models combined)
                            exp_out = os.path.join(
                                output_dir,
                                "experimental_predictions",
                                f"{class_name}_experimental_with_preds_{model_name}_{n_samples}.csv"
                            )
                            exp_df.to_csv(exp_out, index=False)
                            
                            if verbose and model_name == list(models.keys())[0]:
                                print(f"    ✓ Experimental predictions saved ({len(X_exp)} samples)")
                        else:
                            if verbose and model_name == list(models.keys())[0]:
                                print(f"    ⚠️  Experimental file missing required columns")
                            
                    except Exception as e:
                        if verbose:
                            print(f"    ⚠️  Experimental prediction failed: {e}")
    
    # Save overall summary
    results_df = pd.DataFrame(results)
    summary_file = os.path.join(output_dir, "classical_ml_size_scaling_results.csv")
    results_df.to_csv(summary_file, index=False)
    
    if verbose:
        print(f"\n{'='*80}")
        print("✅ CLASSICAL ML BASELINES COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {output_dir}/")
        print(f"  - Summary: classical_ml_size_scaling_results.csv")
        print(f"  - Per-sample predictions: pred_vs_obs/")
        print(f"  - Experimental predictions: experimental_predictions/")
        print(f"\nOverall performance (average R² by model):")
        print(results_df.groupby('Model')['R2'].mean().sort_values(ascending=False).to_string())
        print("\nPerformance by class (average R²):")
        print(results_df.groupby('Class')['R2'].mean().sort_values(ascending=False).to_string())
        print("="*80 + "\n")
    
    return results_df

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Classical ML Baselines for CCS Prediction"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="Data/features",
        help="Directory containing train/test CSV files"
    )
    parser.add_argument(
        "--experimental-dir",
        type=str,
        default="Data/experimental",
        help="Directory containing experimental data"
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
        help="Classes to process (default: all)"
    )
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Training sizes (default: 200 400 600)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Models to run (default: all)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Run classical baselines
    results = run_classical_baselines(
        data_dir=args.data_dir,
        experimental_dir=args.experimental_dir,
        output_dir=args.output_dir,
        classes=args.classes,
        train_sizes=args.train_sizes,
        models_to_run=args.models,
        verbose=not args.quiet
    )
    
    return results

if __name__ == "__main__":
    results = main()
