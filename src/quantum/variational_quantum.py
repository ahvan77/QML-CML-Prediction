import os
import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

DATA_DIR = "Data/features"
EXPERIMENTAL_DIR = "Data/experimental"  # optional experimental data directory

feature_cols = ["A_scaled", "B_scaled", "C_scaled", "D_scaled", "E_scaled"]
train_sizes = [200, 400, 600]
classes = [
    "Carbohydrate", "Lignin", "Lipid",
    "Protein", "Tannin", "Others", "Unsaturated Hydrocarbon"
]

n_qubits = len(feature_cols)
n_wires = n_qubits

# Hyperparameter grids to explore
n_layers_list = [1, 2, 3]
batch_sizes = [16, 32]
steps_list = [30, 60]

dev = qml.device("default.qubit", wires=n_wires)

def circuit(weights, x):
    qml.AngleEmbedding(x, wires=range(n_wires))
    qml.BasicEntanglerLayers(weights, wires=range(n_wires))
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev, interface="autograd")
def qnode(weights, x):
    return circuit(weights, x)

def cost_batch(weights, X_batch, y_batch):
    preds = np.array([qnode(weights, x) for x in X_batch])
    return np.mean((preds - y_batch) ** 2)

# Best optimizers per class from your 400-sample study
# (optimizer_name, main_steps, main_lr, stage2_name, stage2_steps, stage2_lr)
best_opt_by_class = {
    "Carbohydrate": ("momentum", 80, 0.05, None,      0,   0.0),
    "Lignin":       ("rmsprop",  80, 0.01, None,      0,   0.0),
    "Lipid":        ("adam",     60, 0.01, "gd",     20,  0.05),  # Adam+GD
    "Protein":      ("adam",     80, 0.01, None,      0,   0.0),
    "Others":       ("rmsprop",  60, 0.01, "gd",     20,  0.05),  # RMSProp+GD
    # classes without explicit test: use Momentum as reasonable default
    "Tannin":                  ("momentum", 80, 0.05, None, 0, 0.0),
    "Unsaturated Hydrocarbon": ("momentum", 80, 0.05, None, 0, 0.0),
}

def make_optimizer(name, stepsize):
    if name == "gd":
        return qml.GradientDescentOptimizer(stepsize=stepsize)
    if name == "adam":
        return qml.AdamOptimizer(stepsize=stepsize)
    if name == "rmsprop":
        return qml.RMSPropOptimizer(stepsize=stepsize)
    if name == "momentum":
        return qml.MomentumOptimizer(stepsize=stepsize, momentum=0.9)
    raise ValueError(f"Unknown optimizer '{name}'")

results = []

for class_name in classes:
    class_dir = os.path.join(DATA_DIR, class_name)
    train_file = os.path.join(class_dir, "train.csv")
    test_file = os.path.join(class_dir, "test.csv")
    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        print(f"Skipping {class_name}: Missing train or test file.")
        continue

    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    X_train_full = df_train[feature_cols].values
    y_train_full = df_train["CCS"].values
    X_test = df_test[feature_cols].values
    y_test = df_test["CCS"].values

    # pick optimizer config for this class
    main_opt, main_steps_default, main_lr_default, stage2_opt, stage2_steps_default, stage2_lr_default = \
        best_opt_by_class[class_name]

    for n_samples in train_sizes:
        if len(X_train_full) < n_samples:
            print(f"{class_name}: Not enough samples for train size {n_samples}, skipping.")
            continue

        X_sub, _, y_sub, _ = train_test_split(
            X_train_full, y_train_full, train_size=n_samples, random_state=42
        )

        # Normalize targets for stability
        y_mean = y_sub.mean()
        y_std = y_sub.std() if y_sub.std() != 0 else 1.0
        y_sub_norm = (y_sub - y_mean) / y_std

        num_samples = len(X_sub)

        # Loop over VQR hyperparameters
        for n_layers in n_layers_list:
            for batch_size in batch_sizes:
                for steps in steps_list:
                    print(
                        f"\n=== {class_name}, n={n_samples}, "
                        f"layers={n_layers}, batch={batch_size}, steps={steps} ==="
                    )

                    # Initialize trainable parameters
                    weights = np.array(
                        np.random.randn(n_layers, n_qubits), requires_grad=True
                    )

                    # Stage 1 optimizer for this class
                    main_steps = min(steps, main_steps_default)
                    opt1 = make_optimizer(main_opt, main_lr_default)

                    for i in range(main_steps):
                        bs = min(batch_size, num_samples)
                        idx = np.random.choice(num_samples, size=bs, replace=False)
                        X_batch = X_sub[idx]
                        y_batch = y_sub_norm[idx]
                        weights, c = opt1.step_and_cost(
                            lambda w: cost_batch(w, X_batch, y_batch), weights
                        )
                        if i % 10 == 0:
                            print(f"  {main_opt} step {i}, batch cost={c:.4f}")

                    # Optional stage-2 fine-tune if configured and steps allow
                    if stage2_opt is not None and steps > main_steps:
                        s2_steps = min(stage2_steps_default, steps - main_steps)
                        opt2 = make_optimizer(stage2_opt, stage2_lr_default)
                        for j in range(s2_steps):
                            bs = min(batch_size, num_samples)
                            idx = np.random.choice(num_samples, size=bs, replace=False)
                            X_batch = X_sub[idx]
                            y_batch = y_sub_norm[idx]
                            weights, c = opt2.step_and_cost(
                                lambda w: cost_batch(w, X_batch, y_batch), weights
                            )
                            if j % 10 == 0:
                                print(f"  {stage2_opt} fine-tune step {j}, batch cost={c:.4f}")

                    # Predictions (denormalized)
                    preds_norm = np.array([qnode(weights, x) for x in X_test])
                    preds = preds_norm * y_std + y_mean

                    mae = mean_absolute_error(y_test, preds)
                    rmse = mean_squared_error(y_test, preds) ** 0.5
                    r2 = r2_score(y_test, preds)
                    print(
                        f"VQR {class_name} n={n_samples}, "
                        f"layers={n_layers}, batch={batch_size}, steps={steps}: "
                        f"MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.2f}"
                    )

                    # Record config and metrics
                    results.append(
                        {
                            "Class": class_name,
                            "TrainSize": n_samples,
                            "Model": "VQR",
                            "Layers": n_layers,
                            "BatchSize": batch_size,
                            "Steps": steps,
                            "MainOptimizer": main_opt,
                            "MainLR": main_lr_default,
                            "Stage2Optimizer": stage2_opt if stage2_opt else "None",
                            "Stage2LR": stage2_lr_default if stage2_opt else 0.0,
                            "MAE": mae,
                            "RMSE": rmse,
                            "R2": r2,
                        }
                    )

                    # Per-sample predicted vs observed
                    os.makedirs("pred_vs_obs_vqr", exist_ok=True)
                    pred_obs_df = pd.DataFrame(
                        {
                            "Class": class_name,
                            "TrainSize": n_samples,
                            "Model": "VQR",
                            "Layers": n_layers,
                            "BatchSize": batch_size,
                            "Steps": steps,
                            "MainOptimizer": main_opt,
                            "Stage2Optimizer": stage2_opt if stage2_opt else "None",
                            "Observed_CCS": y_test,
                            "Predicted_CCS": preds,
                        }
                    )
                    pred_obs_path = os.path.join(
                        "pred_vs_obs_vqr",
                        f"pred_vs_obs_vqr_{class_name}_n{n_samples}_"
                        f"L{n_layers}_B{batch_size}_S{steps}_{main_opt}.csv",
                    )
                    pred_obs_df.to_csv(pred_obs_path, index=False)

                    # Optional: experimental predictions
                    exp_file = os.path.join(EXPERIMENTAL_DIR, f"{class_name}_experimental.csv")
                    if os.path.exists(exp_file):
                        exp_df = pd.read_csv(exp_file)
                        if all(col in exp_df.columns for col in feature_cols):
                            X_exp = exp_df[feature_cols].values
                            exp_preds_norm = np.array([qnode(weights, x) for x in X_exp])
                            exp_preds = exp_preds_norm * y_std + y_mean

                            col_name = (
                                f"Pred_CCS_VQR_{main_opt}_n{n_samples}_"
                                f"L{n_layers}_B{batch_size}_S{steps}"
                            )
                            exp_df[col_name] = exp_preds
                            os.makedirs("experimental_predictions_vqr", exist_ok=True)
                            out_exp = os.path.join(
                                "experimental_predictions_vqr",
                                f"{class_name}_experimental_VQR_{main_opt}_"
                                f"n{n_samples}_L{n_layers}_B{batch_size}_S{steps}.csv",
                            )
                            exp_df.to_csv(out_exp, index=False)
                        else:
                            print(
                                f"Experimental file for {class_name} missing required feature columns."
                            )

# Save overall VQR summary
pd.DataFrame(results).to_csv(
    "vqr_results_all_classes_hyperparam_scan_classwise_optimizers.csv",
    index=False,
)
print("✅ Saved VQR hyperparameter scan metrics with class-specific optimizers, "
      "per-sample predictions, and experimental predictions (if available).")

