# Usage Guide

This guide provides detailed instructions for running the experiments and reproducing the results from our paper.

## Table of Contents

1. [Dataset Preparation](#dataset-preparation)
2. [Running Individual Models](#running-individual-models)
3. [Generating Figures](#generating-figures)
4. [Reproducing Paper Results](#reproducing-paper-results)
5. [Custom Experiments](#custom-experiments)

---

## Dataset Preparation

### Dataset Structure

The dataset should be organized as follows:

```
data/
├── features/
│   ├── Carbohydrate/
│   │   ├── train.csv  # 5 features + CCS target
│   │   └── test.csv
│   ├── Lignin/
│   ├── Lipid/
│   ├── Protein/
│   └── Others/
└── experimental/
    ├── Carbohydrate_experimental.csv
    └── ...
```

### Required Columns

**Training/Test Files:**
- `A_scaled`, `B_scaled`, `C_scaled`, `D_scaled`, `E_scaled` (features)
- `CCS` (target variable, Ų)

**Experimental Files:**
- Same 5 scaled features
- `CCS` column optional (for validation)

### Data Format Example

```csv
A_scaled,B_scaled,C_scaled,D_scaled,E_scaled,CCS
0.234,-0.456,1.234,-0.789,0.567,245.3
-0.123,0.678,-0.345,0.234,-0.890,312.7
...
```

---

## Running Individual Models

### 1. Quantum Kernel Regression

#### Basic Usage

```python
from src.quantum.quantum_kernel import run_quantum_kernel_experiments

# Run for all classes with default settings
results = run_quantum_kernel_experiments(
    data_dir="Data/features",
    train_sizes=[200, 400, 600],
    classes=["Carbohydrate", "Lignin", "Lipid", "Protein", "Others"]
)
```

#### Advanced Configuration

```python
# Custom configuration
results = run_quantum_kernel_experiments(
    data_dir="Data/features",
    train_sizes=[600],  # Only largest size
    classes=["Carbohydrate"],  # Single class
    reps=1,  # ZZFeatureMap repetitions (optimized)
    alpha=0.03,  # Regularization parameter
    output_dir="results/custom_qkr"
)
```

#### Command Line

```bash
# Run with default settings
python src/quantum/quantum_kernel.py

# Custom parameters
python src/quantum/quantum_kernel.py \
    --data-dir Data/features \
    --train-sizes 200 400 600 \
    --classes Carbohydrate Lipid \
    --reps 1 \
    --output-dir results/qkr_custom
```

#### Expected Output

```
✓ Carbohydrate (train size=600): MAE=12.34, RMSE=15.67, R2=0.945
✓ Saved predictions to: pred_vs_obs_quantum/pred_vs_obs_quantumkernel_Carbohydrate_600.csv
✓ Saved results to: quantum_kernel_results_all_classes.csv
```

---

### 2. Variational Quantum Regression

#### Basic Usage

```python
from src.quantum.variational_quantum import run_vqr_experiments

# Run VQR with optimized settings
results = run_vqr_experiments(
    data_dir="Data/features",
    train_sizes=[200, 400, 600],
    classes=["Carbohydrate", "Lignin", "Lipid", "Protein", "Others"],
    n_layers=1,  # Optimized circuit depth
    steps=30,    # Training steps
    batch_size=32
)
```

#### Custom Optimizer Per Class

```python
# Class-specific optimizers (as determined in paper)
optimizer_config = {
    "Carbohydrate": "momentum",
    "Lignin": "rmsprop",
    "Lipid": "adam",
    "Protein": "adam",
    "Others": "rmsprop_gd"
}

results = run_vqr_experiments(
    data_dir="Data/features",
    optimizer_config=optimizer_config
)
```

#### Command Line

```bash
# Run with default (optimized) settings
python src/quantum/variational_quantum.py

# Custom configuration
python src/quantum/variational_quantum.py \
    --n-layers 1 \
    --steps 30 \
    --batch-size 32 \
    --learning-rate 0.1
```

---

### 3. Classical Machine Learning

#### Basic Usage

```python
from src.classical.classical_ml import run_classical_experiments

# Run all classical models
results = run_classical_experiments(
    data_dir="Data/features",
    train_sizes=[200, 400, 600],
    models=["RandomForest", "SVR", "Lasso", "VotingRegressor"]
)
```

#### Single Model

```python
# Run only Random Forest
results = run_classical_experiments(
    data_dir="Data/features",
    models=["RandomForest"],
    train_sizes=[600]
)
```

#### Command Line

```bash
# All models
python src/classical/classical_ml.py

# Specific models
python src/classical/classical_ml.py \
    --models RandomForest Lasso \
    --train-sizes 200 400 600
```

---

## Generating Figures

### Main Manuscript Figures

```python
from src.visualization.unified_visualization import generate_all_figures_and_tables

# Generate all main figures and tables
generate_all_figures_and_tables()
```

**Output:**
- `Figure_4_test_predictions_n600.png/pdf`
- `Figure_5_experimental_predictions.png/pdf`
- `Table_1_model_performance_summary.csv`
- Supplementary figures and tables

### Custom Visualizations

```python
from src.visualization.unified_visualization import create_figure_4_test_predictions

# Generate specific figure
create_figure_4_test_predictions(
    train_size=600,
    classes_subset=["Carbohydrate", "Lipid"]
)
```

### Command Line

```bash
# Generate all figures
python src/visualization/unified_visualization.py

# Generate supplementary figures
python src/visualization/advanced_comparison.py

# Or use the comprehensive script
bash scripts/generate_all_figures.sh
```

---

## Reproducing Paper Results

### Complete Workflow

To reproduce all results from the paper:

```bash
# 1. Run quantum kernel experiments
bash scripts/run_quantum_kernel.sh

# 2. Run VQR experiments
bash scripts/run_vqr.sh

# 3. Run classical baselines
bash scripts/run_classical.sh

# 4. Generate all figures and tables
bash scripts/generate_all_figures.sh
```

**Total Expected Runtime:** ~2-4 hours (depending on hardware)

### Verify Results

```python
# Load and compare your results with paper
import pandas as pd

# Your results
your_qkr = pd.read_csv("results/quantum_kernel_results_all_classes.csv")

# Paper results (included in repository)
paper_qkr = pd.read_csv("paper_results/quantum_kernel_results.csv")

# Compare
comparison = your_qkr.merge(
    paper_qkr,
    on=["Class", "TrainSize"],
    suffixes=("_yours", "_paper")
)

print("R² Comparison:")
print(comparison[["Class", "TrainSize", "R2_yours", "R2_paper"]])
```

---

## Custom Experiments

### Testing New Quantum Feature Maps

```python
from qiskit.circuit.library import PauliFeatureMap
from src.quantum.quantum_kernel import QuantumKernelRegressor

# Create custom feature map
custom_feature_map = PauliFeatureMap(
    feature_dimension=5,
    reps=2,
    paulis=['Z', 'ZZ']
)

# Run experiments
model = QuantumKernelRegressor(
    feature_map=custom_feature_map,
    alpha=0.1
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

### Testing Different VQR Ansätze

```python
import pennylane as qml
from src.quantum.variational_quantum import train_vqr

# Custom ansatz
def custom_ansatz(weights, x, n_qubits=5):
    # Encoding
    qml.AngleEmbedding(x, wires=range(n_qubits))
    
    # Your custom variational layers
    for layer in range(weights.shape[0]):
        for qubit in range(n_qubits):
            qml.RY(weights[layer, qubit], wires=qubit)
        # Custom entanglement pattern
        for i in range(n_qubits-1):
            qml.CZ(wires=[i, i+1])
    
    return qml.expval(qml.PauliZ(0))

# Train with custom ansatz
results = train_vqr(
    X_train, y_train,
    ansatz=custom_ansatz,
    n_layers=3
)
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV
from src.quantum.quantum_kernel import QuantumKernelWrapper

# Grid search for optimal alpha
param_grid = {'alpha': [0.01, 0.03, 0.1, 0.3, 1.0]}

qkr_model = QuantumKernelWrapper(reps=1)
grid_search = GridSearchCV(
    qkr_model,
    param_grid,
    cv=5,
    scoring='r2'
)

grid_search.fit(X_train, y_train)
print(f"Best alpha: {grid_search.best_params_['alpha']}")
print(f"Best R²: {grid_search.best_score_:.3f}")
```

---

## Performance Tips

### Memory Management

For large datasets (n > 400):

```python
# Process classes sequentially to reduce memory usage
for class_name in classes:
    results = run_quantum_kernel_experiments(
        classes=[class_name],
        train_sizes=[600]
    )
    # Results saved automatically
    del results  # Free memory
```

### Parallel Processing

```python
from joblib import Parallel, delayed

# Run multiple classes in parallel
results = Parallel(n_jobs=4)(
    delayed(run_quantum_kernel_experiments)(
        classes=[cls],
        train_sizes=[600]
    )
    for cls in classes
)
```

### GPU Acceleration (PennyLane)

```python
import pennylane as qml

# Use GPU device for VQR
dev = qml.device("lightning.gpu", wires=5)

@qml.qnode(dev)
def circuit(weights, x):
    # ... your circuit ...
    return qml.expval(qml.PauliZ(0))
```

---

## Troubleshooting

### Common Issues

**1. Memory Error with Quantum Kernel**

```
MemoryError: Unable to allocate array
```

**Solution:** Reduce training size or process classes sequentially

**2. Qiskit Version Conflicts**

```
ImportError: cannot import name 'FidelityQuantumKernel'
```

**Solution:** Ensure correct Qiskit ML version:
```bash
pip install qiskit-machine-learning==0.7.2
```

**3. PennyLane Gradient Issues**

```
ValueError: gradient computation failed
```

**Solution:** Use finite-difference gradients:
```python
@qml.qnode(dev, diff_method="finite-diff")
def circuit(weights, x):
    # ...
```

---

## Next Steps

- Explore [Tutorial Notebooks](../notebooks/) for interactive examples
- See [Methods Documentation](METHODS.md) for algorithmic details
- Check [Contributing Guidelines](../CONTRIBUTING.md) to add new features

---

**Questions?** Open an [issue](https://github.com/yourusername/qml-ccs-prediction/issues) or contact the authors.
