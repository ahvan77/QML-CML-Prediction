import pandas as pd
import numpy as np
import os
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.linear_model import Ridge as RidgeMeta
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict

np.random.seed(42)

feature_cols = ["A_scaled", "B_scaled", "C_scaled", "D_scaled", "E_scaled"]
train_sizes = [200, 400, 600]
reps_list = [1, 2, 3]  # Test with 1, 2, and 3 reps
classes = [
    "Carbohydrate", "Lignin", "Lipid",
    "Protein", "Others"
]

DATA_DIR = "Data/features"
EXPERIMENTAL_DIR = "Data/experimental"

# Best alphas for each class (starting values for tuning)
best_alpha_by_class = {
    "Carbohydrate": 0.1,
    "Lignin": 1.0,
    "Lipid": 0.01,
    "Protein": 0.1,
    "Others": 0.1,
    "Tannin": 0.1,
    "Unsaturated Hydrocarbon": 0.1,
}

results = []

for class_name in classes:
    class_dir = os.path.join(DATA_DIR, class_name)
    train_file = os.path.join(class_dir, "train.csv")
    test_file = os.path.join(class_dir, "test.csv")
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"Skipping {class_name}: Missing train or test file.")
        continue

    df_train = pd.read_csv(train_file)
    X_train_full = df_train[feature_cols].values
    y_train_full = df_train["CCS"].values

    df_test = pd.read_csv(test_file)
    X_test = df_test[feature_cols].values
    y_test = df_test["CCS"].values

    # Get class-specific best alpha
    alpha_init = best_alpha_by_class.get(class_name, 0.1)

    for n_samples in train_sizes:
        if len(X_train_full) < n_samples:
            print(f"{class_name}: Not enough samples for train size {n_samples}. Skipping.")
            continue

        # Random sampling for each size
        X_sub, _, y_sub, _ = train_test_split(
            X_train_full, y_train_full, train_size=n_samples, random_state=42
        )

        # Normalize CCS targets
        y_mean = y_sub.mean()
        y_std = y_sub.std() if y_sub.std() != 0 else 1.0
        y_sub_norm = (y_sub - y_mean) / y_std

        print(f"\n{'='*80}")
        print(f"CLASS: {class_name} | Train Size: {n_samples}")
        print(f"{'='*80}")

        # Loop through different reps values
        for reps in reps_list:
            print(f"\n{'-'*80}")
            print(f"Testing with reps = {reps}")
            print(f"{'-'*80}")

            # Define feature map based on number of features
            if len(feature_cols) == 1:
                print(f"⚠️  Single feature detected. Using ZFeatureMap (1D compatible).")
                feature_map = ZFeatureMap(feature_dimension=len(feature_cols), reps=reps)
            else:
                feature_map = ZZFeatureMap(feature_dimension=len(feature_cols), reps=reps)
            
            quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

            try:
                # Precompute kernel matrices
                print(f"Computing quantum kernel matrices (reps={reps})...")
                K_train = quantum_kernel.evaluate(X_sub)
                K_test = quantum_kernel.evaluate(X_test, X_sub)
            except (MemoryError, Exception) as e:
                print(f"{type(e).__name__}: Failed for {class_name} at "
                      f"size={n_samples}, reps={reps}: {e}")
                continue

            # =================================================================
            # MODEL 1: Kernel Ridge Regression
            # =================================================================
            print(f"\n--- Kernel Ridge Regression (reps={reps}) ---")
            
            param_grid_kr = {"alpha": np.logspace(np.log10(alpha_init) - 1, np.log10(alpha_init) + 1, 5)}
            kr = KernelRidge(kernel="precomputed")
            grid_kr = GridSearchCV(kr, param_grid_kr, cv=5, scoring="neg_mean_squared_error")
            grid_kr.fit(K_train, y_sub_norm)

            best_alpha = grid_kr.best_params_["alpha"]
            regressor_kr = KernelRidge(alpha=best_alpha, kernel="precomputed")
            regressor_kr.fit(K_train, y_sub_norm)
            preds_norm_kr = regressor_kr.predict(K_test)
            preds_kr = preds_norm_kr * y_std + y_mean

            mae_kr = mean_absolute_error(y_test, preds_kr)
            rmse_kr = mean_squared_error(y_test, preds_kr) ** 0.5
            r2_kr = r2_score(y_test, preds_kr)
            
            print(f"Best alpha: {best_alpha:.3g}")
            print(f"MAE: {mae_kr:.3f} | RMSE: {rmse_kr:.3f} | R²: {r2_kr:.3f}")

            results.append({
                "Class": class_name,
                "TrainSize": n_samples,
                "Model": "KernelRidge",
                "Reps": reps,
                "BestAlpha": best_alpha,
                "MAE": mae_kr,
                "RMSE": rmse_kr,
                "R2": r2_kr
            })

            # Save per-sample predictions for KR
            os.makedirs("pred_vs_obs_quantum", exist_ok=True)
            pred_obs_kr = pd.DataFrame({
                "Class": class_name,
                "TrainSize": n_samples,
                "Model": "KernelRidge",
                "Reps": reps,
                "BestAlpha": best_alpha,
                "Observed_CCS": y_test,
                "Predicted_CCS": preds_kr
            })
            pred_obs_kr.to_csv(
                os.path.join("pred_vs_obs_quantum", 
                            f"pred_vs_obs_KR_{class_name}_n{n_samples}_R{reps}.csv"),
                index=False
            )

            # =================================================================
            # MODEL 2: Support Vector Regression (for ensemble)
            # =================================================================
            print(f"\n--- Support Vector Regression (reps={reps}) ---")
            
            param_grid_svr = {
                "C": np.logspace(-2, 2, 5),
                "epsilon": [0.01, 0.1, 0.5]
            }
            svr = SVR(kernel="precomputed")
            grid_svr = GridSearchCV(svr, param_grid_svr, cv=5, scoring="neg_mean_squared_error")
            grid_svr.fit(K_train, y_sub_norm)

            best_svr = grid_svr.best_estimator_
            preds_norm_svr = best_svr.predict(K_test)
            preds_svr = preds_norm_svr * y_std + y_mean

            mae_svr = mean_absolute_error(y_test, preds_svr)
            rmse_svr = mean_squared_error(y_test, preds_svr) ** 0.5
            r2_svr = r2_score(y_test, preds_svr)
            
            print(f"Best params: C={grid_svr.best_params_['C']:.3g}, "
                  f"epsilon={grid_svr.best_params_['epsilon']:.3g}")
            print(f"MAE: {mae_svr:.3f} | RMSE: {rmse_svr:.3f} | R²: {r2_svr:.3f}")

            # =================================================================
            # MODEL 3: Ensemble Stacking (KR + SVR)
            # =================================================================
            print(f"\n--- Ensemble Stacking (KR + SVR, reps={reps}) ---")
            
            try:
                # Get out-of-fold predictions for training set
                preds_kr_train_oof = cross_val_predict(
                    KernelRidge(alpha=best_alpha, kernel="precomputed"),
                    K_train, y_sub_norm, cv=5
                )
                preds_svr_train_oof = cross_val_predict(
                    SVR(kernel="precomputed", C=grid_svr.best_params_['C'], 
                        epsilon=grid_svr.best_params_['epsilon']),
                    K_train, y_sub_norm, cv=5
                )
                
                # Create meta-features
                X_meta_train = np.column_stack([preds_kr_train_oof, preds_svr_train_oof])
                X_meta_test = np.column_stack([preds_norm_kr, preds_norm_svr])
                
                # Train and predict with meta-learner
                meta_model = RidgeMeta(alpha=0.1)
                meta_model.fit(X_meta_train, y_sub_norm)
                preds_norm_stacked = meta_model.predict(X_meta_test)
                preds_stacked = preds_norm_stacked * y_std + y_mean
                
                mae_stacked = mean_absolute_error(y_test, preds_stacked)
                rmse_stacked = mean_squared_error(y_test, preds_stacked) ** 0.5
                r2_stacked = r2_score(y_test, preds_stacked)
                
                meta_weights = meta_model.coef_
                print(f"Meta-weights: KR={meta_weights[0]:.3f}, SVR={meta_weights[1]:.3f}")
                print(f"MAE: {mae_stacked:.3f} | RMSE: {rmse_stacked:.3f} | R²: {r2_stacked:.3f}")
                
                results.append({
                    "Class": class_name,
                    "TrainSize": n_samples,
                    "Model": "Ensemble_Stacking",
                    "Reps": reps,
                    "MetaWeight_KR": meta_weights[0],
                    "MetaWeight_SVR": meta_weights[1],
                    "MAE": mae_stacked,
                    "RMSE": rmse_stacked,
                    "R2": r2_stacked
                })
                
                # Save stacked predictions
                pred_obs_stacked = pd.DataFrame({
                    "Class": class_name,
                    "TrainSize": n_samples,
                    "Model": "Ensemble_Stacking",
                    "Reps": reps,
                    "Observed_CCS": y_test,
                    "Predicted_CCS": preds_stacked
                })
                pred_obs_stacked.to_csv(
                    os.path.join("pred_vs_obs_quantum", 
                                f"pred_vs_obs_Stacking_{class_name}_n{n_samples}_R{reps}.csv"),
                    index=False
                )
                
            except Exception as e:
                print(f"Stacking failed: {type(e).__name__}: {e}")

            # =================================================================
            # Comparison Summary for this reps value
            # =================================================================
            print(f"\n{'='*80}")
            print(f"COMPARISON: {class_name} (n={n_samples}, reps={reps})")
            print(f"{'='*80}")
            print(f"{'Model':<25} {'MAE':<10} {'RMSE':<10} {'R²':<10}")
            print(f"{'-'*80}")
            print(f"{'KernelRidge':<25} {mae_kr:<10.3f} {rmse_kr:<10.3f} {r2_kr:<10.3f}")
            try:
                print(f"{'Ensemble_Stacking':<25} {mae_stacked:<10.3f} {rmse_stacked:<10.3f} {r2_stacked:<10.3f}")
                improvement = ((r2_stacked - r2_kr) / r2_kr * 100) if r2_kr != 0 else 0
                print(f"{'Improvement':<25} {improvement:>10.2f}%")
            except:
                print(f"{'Ensemble_Stacking':<25} {'FAILED':<10} {'FAILED':<10} {'FAILED':<10}")
            print(f"{'='*80}\n")

            # =================================================================
            # Experimental Predictions (both models)
            # =================================================================
            exp_file = os.path.join(EXPERIMENTAL_DIR, f"{class_name}_experimental.csv")
            if os.path.exists(exp_file):
                exp_df = pd.read_csv(exp_file)
                if all(col in exp_df.columns for col in feature_cols):
                    X_exp = exp_df[feature_cols].values
                    try:
                        K_exp = quantum_kernel.evaluate(X_exp, X_sub)
                        
                        # KR predictions
                        exp_preds_norm_kr = regressor_kr.predict(K_exp)
                        exp_preds_kr = exp_preds_norm_kr * y_std + y_mean
                        exp_df[f"Pred_CCS_KR_n{n_samples}_R{reps}"] = exp_preds_kr
                        
                        # Stacking predictions
                        try:
                            exp_preds_norm_svr = best_svr.predict(K_exp)
                            X_exp_meta = np.column_stack([exp_preds_norm_kr, exp_preds_norm_svr])
                            exp_preds_norm_stacked = meta_model.predict(X_exp_meta)
                            exp_preds_stacked = exp_preds_norm_stacked * y_std + y_mean
                            exp_df[f"Pred_CCS_Stacking_n{n_samples}_R{reps}"] = exp_preds_stacked
                        except:
                            print("Stacking experimental prediction skipped")
                        
                        os.makedirs("experimental_predictions_quantum", exist_ok=True)
                        out_exp = os.path.join(
                            "experimental_predictions_quantum",
                            f"{class_name}_experimental_n{n_samples}_R{reps}.csv"
                        )
                        exp_df.to_csv(out_exp, index=False)
                        print(f"✓ Saved experimental predictions (reps={reps})")
                        
                    except (MemoryError, Exception) as e:
                        print(f"Experimental prediction failed: {type(e).__name__}: {e}")

# Save summary results
results_df = pd.DataFrame(results)
results_df.to_csv("quantum_kernel_KR_vs_Stacking_reps_comparison.csv", index=False)

# Create pivot tables for easy comparison
print("\n" + "="*80)
print("FINAL SUMMARY - KernelRidge R² by Reps")
print("="*80)
pivot_kr = results_df[results_df['Model'] == 'KernelRidge'].pivot_table(
    values='R2', 
    index=['Class', 'TrainSize'], 
    columns='Reps'
)
print(pivot_kr)

print("\n" + "="*80)
print("FINAL SUMMARY - Ensemble Stacking R² by Reps")
print("="*80)
pivot_stacking = results_df[results_df['Model'] == 'Ensemble_Stacking'].pivot_table(
    values='R2', 
    index=['Class', 'TrainSize'], 
    columns='Reps'
)
print(pivot_stacking)

print("\n" + "="*80)
print("FINAL SUMMARY - Best Model by Class and TrainSize")
print("="*80)
best_models = results_df.loc[results_df.groupby(['Class', 'TrainSize', 'Reps'])['R2'].idxmax()]
print(best_models[['Class', 'TrainSize', 'Reps', 'Model', 'R2']])

print("\n✅ All results saved!")
print(f"   - Summary: quantum_kernel_KR_vs_Stacking_reps_comparison.csv")
print(f"   - Per-sample predictions: pred_vs_obs_quantum/")
print(f"   - Experimental predictions: experimental_predictions_quantum/")
