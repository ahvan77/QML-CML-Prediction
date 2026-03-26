"""
Variational Quantum Regression for CCS Prediction
=================================================

Windows-compatible version using autograd interface.

Main implementation for manuscript Section 2.4.

Configuration (Optimized):
- Circuit: 5 qubits, AngleEmbedding + 2 BasicEntanglerLayers
- Parameters: 10 trainable (2 layers × 5 qubits)
- Training: 30 steps, batch size 32, gradient descent
- Optimizer: Class-specific (from Table S5)
- Output: ⟨Z₀⟩ expectation value

Author: Sadollah Ebrahimi
Date: March 2026
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse

# Check for PennyLane and try to import with error handling
try:
    import pennylane as qml
    from pennylane import numpy as pnp
except ImportError as e:
    print(f"Error importing PennyLane: {e}")
    print("\nPlease install PennyLane:")
    print("  pip uninstall pennylane jax jaxlib -y")
    print("  pip install pennylane==0.37.0 autograd")
    sys.exit(1)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Set random seeds
np.random.seed(42)
try:
    pnp.random.seed(42)
except:
    pass

# Molecular descriptors (Z-score normalized)
FEATURE_COLS = ["A_scaled", "B_scaled", "C_scaled", "D_scaled", "E_scaled"]

# Training set sizes to evaluate
TRAIN_SIZES = [200, 400, 600]

# Molecular classes
CLASSES = ["Carbohydrate", "Lignin", "Lipid", "Protein", "Others"]

# VQR circuit configuration (optimized from Section 3.3)
N_QUBITS = len(FEATURE_COLS)
N_LAYERS = 2  # BasicEntanglerLayers depth (optimized)

# Training configuration (optimized)
BATCH_SIZE = 32
STEPS = 30
LEARNING_RATE = 0.1

# Class-specific optimizers (from Table S5)
CLASS_OPTIMIZERS = {
    "Carbohydrate": "momentum",
    "Lignin": "rmsprop",
    "Lipid": "adam",
    "Protein": "adam",
    "Others": "rmsprop",
}

# ============================================================================
# QUANTUM CIRCUIT
# ============================================================================

# Initialize quantum device with autograd interface
try:
    dev = qml.device("default.qubit", wires=N_QUBITS)
except Exception as e:
    print(f"Error initializing quantum device: {e}")
    print("Trying legacy device initialization...")
    dev = qml.device("default.qubit.autograd", wires=N_QUBITS)

def create_circuit(weights, x):
    """
    VQR circuit as described in manuscript Figure 3.
    
    Architecture:
    1. AngleEmbedding: Encode features as RX rotations
    2. BasicEntanglerLayers: Trainable RY rotations + CNOT chain
    3. Measurement: ⟨Z₀⟩ on first qubit
    
    Parameters
    ----------
    weights : array_like
        Trainable parameters, shape (n_layers, n_qubits)
    x : array_like
        Input features, shape (n_qubits,)
        
    Returns
    -------
    expectation : float
        ⟨Z₀⟩ expectation value in [-1, 1]
    """
    # Encode input features
    qml.AngleEmbedding(x, wires=range(N_QUBITS))
    
    # Variational layers
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    
    # Measure first qubit
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev, interface="autograd")
def qnode(weights, x):
    """Quantum node for circuit evaluation."""
    return create_circuit(weights, x)

def cost_batch(weights, X_batch, y_batch):
    """
    Batch cost function (MSE on normalized targets).
    
    Parameters
    ----------
    weights : array_like
        Circuit parameters
    X_batch : array_like
        Batch of input features
    y_batch : array_like
        Batch of normalized target values
        
    Returns
    -------
    cost : float
        Mean squared error
    """
    try:
        predictions = pnp.array([qnode(weights, x) for x in X_batch])
    except:
        # Fallback to regular numpy if pnp fails
        predictions = np.array([qnode(weights, x) for x in X_batch])
    
    try:
        return pnp.mean((predictions - y_batch) ** 2)
    except:
        return np.mean((predictions - y_batch) ** 2)

# ============================================================================
# OPTIMIZER FACTORY
# ============================================================================

def get_optimizer(name, learning_rate=LEARNING_RATE):
    """
    Create optimizer based on class-specific configuration.
    
    Parameters
    ----------
    name : str
        Optimizer name: 'gd', 'adam', 'rmsprop', 'momentum'
    learning_rate : float
        Learning rate / step size
        
    Returns
    -------
    optimizer : pennylane.Optimizer
    """
    if name == "gd":
        return qml.GradientDescentOptimizer(stepsize=learning_rate)
    elif name == "adam":
        return qml.AdamOptimizer(stepsize=learning_rate)
    elif name == "rmsprop":
        return qml.RMSPropOptimizer(stepsize=learning_rate)
    elif name == "momentum":
        return qml.MomentumOptimizer(stepsize=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {name}")

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_vqr(X_train, y_train, optimizer_name, n_layers=N_LAYERS, 
              batch_size=BATCH_SIZE, steps=STEPS, verbose=False):
    """
    Train VQR model.
    
    Parameters
    ----------
    X_train : array_like
        Training features, shape (n_samples, n_features)
    y_train : array_like
        Normalized training targets, shape (n_samples,)
    optimizer_name : str
        Optimizer to use
    n_layers : int
        Number of variational layers
    batch_size : int
        Mini-batch size
    steps : int
        Number of training steps
    verbose : bool
        Print training progress
        
    Returns
    -------
    weights : array_like
        Trained circuit parameters
    """
    n_samples = len(X_train)
    
    # Initialize weights
    try:
        weights = pnp.array(
            pnp.random.randn(n_layers, N_QUBITS),
            requires_grad=True
        )
    except:
        # Fallback for older PennyLane versions
        weights = np.random.randn(n_layers, N_QUBITS)
        weights = pnp.array(weights, requires_grad=True)
    
    # Create optimizer
    optimizer = get_optimizer(optimizer_name)
    
    # Training loop
    for step in range(steps):
        # Sample mini-batch
        batch_size_actual = min(batch_size, n_samples)
        indices = np.random.choice(n_samples, size=batch_size_actual, replace=False)
        X_batch = X_train[indices]
        y_batch = y_train[indices]
        
        # Update weights
        try:
            weights, cost = optimizer.step_and_cost(
                lambda w: cost_batch(w, X_batch, y_batch),
                weights
            )
        except Exception as e:
            if verbose:
                print(f"  Warning: Optimization step {step} had issues: {e}")
            # Continue with current weights
            cost = cost_batch(weights, X_batch, y_batch)
        
        if verbose and step % 10 == 0:
            print(f"  Step {step:3d}: Batch cost = {cost:.4f}")
    
    return weights

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def run_vqr_experiments(
    data_dir="Data/features",
    experimental_dir="Data/experimental",
    output_dir="results",
    classes=None,
    train_sizes=None,
    verbose=True
):
    """
    Run VQR experiments for CCS prediction.
    
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
    verbose : bool
        Print progress information
        
    Returns
    -------
    results_df : DataFrame
        Summary results with MAE, RMSE, R² for each class/size
    """
    
    if classes is None:
        classes = CLASSES
    if train_sizes is None:
        train_sizes = TRAIN_SIZES
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "pred_vs_obs_vqr"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "experimental_predictions_vqr"), exist_ok=True)
    
    results = []
    
    if verbose:
        print("\n" + "="*80)
        print("VARIATIONAL QUANTUM REGRESSION - CCS PREDICTION")
        print("="*80)
        print(f"Configuration:")
        print(f"  Circuit: {N_QUBITS} qubits, {N_LAYERS} BasicEntanglerLayers")
        print(f"  Parameters: {N_LAYERS * N_QUBITS} trainable")
        print(f"  Training: {STEPS} steps, batch size {BATCH_SIZE}")
        print(f"  Classes: {len(classes)}")
        print(f"  Training sizes: {train_sizes}")
        print(f"  Device: {dev}")
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
        
        # Get class-specific optimizer
        optimizer_name = CLASS_OPTIMIZERS.get(class_name, "momentum")
        if verbose:
            print(f"Optimizer: {optimizer_name}")
        
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
            
            # Normalize targets (z-score)
            y_mean = y_train.mean()
            y_std = y_train.std() if y_train.std() > 0 else 1.0
            y_train_norm = (y_train - y_mean) / y_std
            
            # Train VQR
            if verbose:
                print(f"Training VQR ({STEPS} steps, batch={BATCH_SIZE})...")
            
            try:
                weights = train_vqr(
                    X_train, y_train_norm,
                    optimizer_name=optimizer_name,
                    n_layers=N_LAYERS,
                    batch_size=BATCH_SIZE,
                    steps=STEPS,
                    verbose=verbose
                )
            except Exception as e:
                print(f"❌ Training failed for {class_name} (n={n_samples}): {e}")
                continue
            
            # Predict on test set
            try:
                preds_norm = np.array([qnode(weights, x) for x in X_test])
            except Exception as e:
                print(f"❌ Prediction failed: {e}")
                continue
                
            preds = preds_norm * y_std + y_mean  # Denormalize
            
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
                "Model": "VQR",
                "Layers": N_LAYERS,
                "Optimizer": optimizer_name,
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
                "Model": "VQR",
                "Observed_CCS": y_test,
                "Predicted_CCS": preds
            })
            pred_file = os.path.join(
                output_dir,
                "pred_vs_obs_vqr",
                f"pred_vs_obs_vqr_{class_name}_{n_samples}.csv"
            )
            pred_df.to_csv(pred_file, index=False)
            
            # Predict on experimental samples
            exp_file = os.path.join(experimental_dir, f"{class_name}_experimental.csv")
            if os.path.exists(exp_file):
                try:
                    exp_df = pd.read_csv(exp_file)
                    
                    if all(col in exp_df.columns for col in FEATURE_COLS):
                        X_exp = exp_df[FEATURE_COLS].values
                        
                        # Predict and denormalize
                        exp_preds_norm = np.array([qnode(weights, x) for x in X_exp])
                        exp_preds = exp_preds_norm * y_std + y_mean
                        
                        # Add to dataframe
                        exp_df[f"Predicted_CCS_VQR_train{n_samples}"] = exp_preds
                        
                        # Save
                        exp_out = os.path.join(
                            output_dir,
                            "experimental_predictions_vqr",
                            f"{class_name}_experimental_with_vqr_preds_{n_samples}.csv"
                        )
                        exp_df.to_csv(exp_out, index=False)
                        
                        if verbose:
                            print(f"✓ Experimental predictions saved ({len(X_exp)} samples)")
                    else:
                        print(f"⚠️  Experimental file missing required columns")
                        
                except Exception as e:
                    print(f"⚠️  Experimental prediction failed: {e}")
    
    # Save overall summary
    results_df = pd.DataFrame(results)
    summary_file = os.path.join(output_dir, "vqr_results_all_classes.csv")
    results_df.to_csv(summary_file, index=False)
    
    if verbose:
        print(f"\n{'='*80}")
        print("✅ VARIATIONAL QUANTUM REGRESSION COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {output_dir}/")
        print(f"  - Summary: vqr_results_all_classes.csv")
        print(f"  - Per-sample predictions: pred_vs_obs_vqr/")
        print(f"  - Experimental predictions: experimental_predictions_vqr/")
        if len(results_df) > 0:
            print(f"\nOverall performance (average R² by class):")
            print(results_df.groupby('Class')['R2'].mean().to_string())
        print("="*80 + "\n")
    
    return results_df

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Variational Quantum Regression for CCS Prediction"
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
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    # Run VQR experiments
    results = run_vqr_experiments(
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
