# Quick Start Guide

Complete setup and usage guide for reproducing the manuscript results.

## 📋 Prerequisites

- Python 3.8 or higher
- 16+ GB RAM (for quantum kernel with n=600)
- ~50 GB free disk space
- Linux, macOS, or Windows

## 🚀 Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/qml-ccs-prediction.git
cd qml-ccs-prediction
```

### Step 2: Create Virtual Environment

**Option A: Using venv (recommended)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Option B: Using conda**

```bash
conda env create -f environment.yml
conda activate qml-ccs
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Verify installation:**

```bash
python -c "import qiskit; import pennylane; import sklearn; print('✅ All packages installed')"
```

## 📁 Data Preparation

### Step 1: Organize Your Data

Your data should be organized as follows:

```
Data/
├── features/
│   ├── Carbohydrate/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── Lignin/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── Lipid/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── Protein/
│   │   ├── train.csv
│   │   └── test.csv
│   └── Others/
│       ├── train.csv
│       └── test.csv
└── experimental/  # Optional
    ├── Carbohydrate_experimental.csv
    ├── Lignin_experimental.csv
    └── ...
```

### Step 2: Data Format

Each CSV file must contain:

**Required columns:**
- `A_scaled` - m/z (Z-score normalized)
- `B_scaled` - DBE (Z-score normalized)
- `C_scaled` - AImod (Z-score normalized)
- `D_scaled` - H/C ratio (Z-score normalized)
- `E_scaled` - O/C ratio (Z-score normalized)
- `CCS` - Collision cross-section (Ų)

**Example:**

```csv
A_scaled,B_scaled,C_scaled,D_scaled,E_scaled,CCS
0.234,-0.456,1.234,-0.789,0.567,245.3
-0.123,0.678,-0.345,0.234,-0.890,312.7
```

**Verify your data:**

```bash
python -c "
import pandas as pd
df = pd.read_csv('Data/features/Carbohydrate/train.csv')
required_cols = ['A_scaled', 'B_scaled', 'C_scaled', 'D_scaled', 'E_scaled', 'CCS']
assert all(col in df.columns for col in required_cols), 'Missing required columns'
print(f'✅ Data format correct ({len(df)} samples)')
"
```

## 🏃 Running Experiments

### Complete Workflow (All Experiments)

```bash
# Make script executable
chmod +x run_all_experiments.sh

# Run everything
bash run_all_experiments.sh
```

**Expected runtime:** 2-4 hours (depending on hardware)

### Run Individual Components

**1. Quantum Kernel Regression Only:**

```bash
python src/quantum/quantum_kernel_regression.py
```

Expected runtime: ~30-60 minutes per class

**2. Variational Quantum Regression Only:**

```bash
python src/quantum/variational_quantum_regression.py
```

Expected runtime: ~20-45 minutes per class

**3. Classical Baselines Only:**

```bash
python src/classical/classical_baselines.py
```

Expected runtime: ~5-10 minutes total

**4. Generate Figures Only:**

```bash
python src/visualization/unified_visualization.py
python src/visualization/advanced_comparison.py
```

Expected runtime: ~2-5 minutes

### Custom Configurations

**Run specific classes only:**

```bash
python src/quantum/quantum_kernel_regression.py \
    --classes Carbohydrate Lipid
```

**Run specific training sizes:**

```bash
python src/quantum/quantum_kernel_regression.py \
    --train-sizes 600
```

**Combine options:**

```bash
python src/classical/classical_baselines.py \
    --classes Carbohydrate \
    --train-sizes 200 400 600 \
    --models RandomForest Lasso
```

## 📊 Output Structure

After running experiments, you'll have:

```
qml-ccs-prediction/
├── results/
│   ├── quantum_kernel_results_all_classes.csv
│   ├── vqr_results_all_classes.csv
│   ├── classical_ml_size_scaling_results.csv
│   ├── pred_vs_obs_quantum/
│   ├── pred_vs_obs_vqr/
│   ├── pred_vs_obs/
│   ├── experimental_predictions_quantum/
│   ├── experimental_predictions_vqr/
│   └── experimental_predictions/
│
├── figures/
│   ├── main/
│   │   ├── Figure_4_test_predictions_n600.pdf
│   │   └── Figure_5_experimental_predictions.pdf
│   └── supplementary/
│       ├── Figure_S3_combined_quantum_vs_classical.pdf
│       ├── Figure_S4_residual_plots_n600.pdf
│       ├── Figure_S5_training_size_effect.pdf
│       └── Figure_S6_experimental_distributions.pdf
│
└── tables/
    ├── Table_1_model_performance_summary.csv
    ├── Table_S4_comprehensive_prediction_metrics.csv
    ├── Table_S5_experimental_prediction_statistics.csv
    └── Table_S6_statistical_comparisons.csv
```

## 🔍 Verifying Results

### Check Summary Statistics

```python
import pandas as pd

# Load results
qkr = pd.read_csv("results/quantum_kernel_results_all_classes.csv")
vqr = pd.read_csv("results/vqr_results_all_classes.csv")
classical = pd.read_csv("results/classical_ml_size_scaling_results.csv")

# Average R² by model
print("Average R² by model:")
print(qkr.groupby('Model')['R2'].mean())
print(vqr.groupby('Model')['R2'].mean())
print(classical.groupby('Model')['R2'].mean())
```

### Compare with Manuscript

Expected results (approximate, n=600):

| Model | Carbohydrate R² | Lipid R² | Protein R² |
|-------|-----------------|----------|------------|
| **QuantumKernel** | 0.95 | 0.94 | 0.48 |
| **VQR** | 0.45 | 0.19 | 0.06 |
| **RandomForest** | 0.97 | 0.96 | 0.90 |
| **Lasso** | 0.96 | 0.95 | 0.88 |

## 🐛 Troubleshooting

### Issue 1: Memory Error (Quantum Kernel)

```
MemoryError: Unable to allocate array
```

**Solution:** Reduce training size or process classes sequentially

```bash
# Run one class at a time
python src/quantum/quantum_kernel_regression.py --classes Carbohydrate
python src/quantum/quantum_kernel_regression.py --classes Lignin
# etc.
```

### Issue 2: Import Error (Qiskit)

```
ImportError: cannot import name 'FidelityQuantumKernel'
```

**Solution:** Ensure correct Qiskit ML version

```bash
pip install qiskit-machine-learning==0.7.2 --force-reinstall
```

### Issue 3: PennyLane Device Error

```
DeviceError: device does not exist
```

**Solution:** Reinstall PennyLane

```bash
pip install pennylane==0.38.0 --force-reinstall
```

### Issue 4: Missing Data Files

```
Skipping Carbohydrate: Missing train or test file
```

**Solution:** Check data directory structure matches the required format

```bash
ls -R Data/features/Carbohydrate/
# Should show: train.csv  test.csv
```

### Issue 5: Slow Performance

**For Quantum Experiments:**
- Use smaller training sizes first (200 samples)
- Process one class at a time
- Consider running overnight

**For VQR:**
- Reduce number of steps: `--steps 10`
- Increase batch size: `--batch-size 64`

## 💡 Tips & Best Practices

### 1. Start Small

Test with one class and small training size:

```bash
python src/quantum/quantum_kernel_regression.py \
    --classes Carbohydrate \
    --train-sizes 200
```

### 2. Monitor Progress

Use verbose output (default) to track progress:

```bash
python src/quantum/quantum_kernel_regression.py
# Shows: Computing kernel matrices...
# Shows: ✓ Kernel matrices computed
```

### 3. Save Intermediate Results

Results are automatically saved after each class/size combination. If a run fails, you won't lose completed work.

### 4. Use Screen/Tmux for Long Runs

```bash
# Start screen session
screen -S qml-experiments

# Run experiments
bash run_all_experiments.sh

# Detach: Ctrl+A, then D
# Reattach: screen -r qml-experiments
```

### 5. Check Disk Space

Quantum kernel matrices can be large:

```bash
# Check available space
df -h .

# Monitor during execution
watch -n 60 'du -sh results/'
```

## 📚 Next Steps

1. ✅ Run complete workflow: `bash run_all_experiments.sh`
2. ✅ Review results in `results/` directory
3. ✅ Check figures in `figures/` directory
4. ✅ Compare with manuscript results
5. ✅ Explore notebooks in `notebooks/` for interactive analysis

## 🤝 Getting Help

- **Issues:** Open an issue on GitHub
- **Questions:** Contact [your-email@university.edu]
- **Documentation:** See `docs/` directory

## 📖 Additional Resources

- Full documentation: `docs/USAGE.md`
- API reference: `docs/API.md`
- Tutorial notebooks: `notebooks/`
- Paper: [DOI link]

---

**Happy experimenting! 🎉**
