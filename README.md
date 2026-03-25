# QML-CML-Prediction
# Quantum vs. Classical Machine Learning for CCS Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-0.7.2-6133BD)](https://qiskit.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-0.38.0-blue)](https://pennylane.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Scaling Behavior of Quantum Kernel Regression for Chemical Data: Insights from Homogeneous Subsets and CCS Prediction**

This repository contains the complete code, data, and analysis for our systematic comparison of **quantum machine learning (QML)** and **classical machine learning (CML)** methods for predicting collision cross-section (CCS) values of dissolved organic matter (DOM) molecules.


## 🗂️ Repository Structure

```
qml-ccs-prediction/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── environment.yml                    # Conda environment file
│
├── data/                              # Dataset files
│   ├── features/                      # Train/test splits by chemical class
│   │   ├── Carbohydrate/
│   │   │   ├── train.csv
│   │   │   └── test.csv
│   │   ├── Lignin/
│   │   ├── Lipid/
│   │   ├── Protein/
│   │   └── Others/
│   ├── experimental/                  # Experimental validation samples
│   │   ├── Carbohydrate_experimental.csv
│   │   └── ...
│   └── molecules.csv                  # Full dataset
│
├── src/                               # Source code
│   ├── quantum/
│   │   ├── quantum_kernel.py          # QKR implementation
│   │   └── variational_quantum.py     # VQR implementation
│   ├── classical/
│   │   └── classical_ml.py            # Classical ML baselines
│   ├── visualization/
│   │   ├── unified_visualization.py   # Main figures generator
│   │   └── advanced_comparison.py     # Supplementary figures
│   └── utils/
│       ├── data_loader.py             # Data loading utilities
│       └── metrics.py                 # Evaluation metrics
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_data_exploration.ipynb      # Dataset analysis
│   ├── 02_quantum_kernel_analysis.ipynb
│   ├── 03_vqr_optimization.ipynb
│   └── 04_results_visualization.ipynb
│
├── results/                           # Output files
│   ├── quantum_kernel_results_all_classes.csv
│   ├── vqr_results_all_classes.csv
│   ├── classical_ml_size_scaling_results.csv
│   ├── pred_vs_obs_quantum/           # Prediction files
│   ├── pred_vs_obs_vqr/
│   ├── pred_vs_obs/                   # Classical predictions
│   ├── experimental_predictions_quantum/
│   ├── experimental_predictions_vqr/
│   └── experimental_predictions/
│
├── figures/                           # Publication-ready figures
│   ├── main/
│   │   ├── Figure_1_QKR_workflow.pdf
│   │   ├── Figure_3_VQR_workflow.pdf
│   │   ├── Figure_4_R2_scaling.pdf
│   │   ├── Figure_5_RMSE_scaling.pdf
│   │   └── Figure_6_calibration_plots.pdf
│   └── supplementary/
│       ├── Figure_S1_ZZFeatureMap.pdf
│       ├── Figure_S2_MAE_scaling.pdf
│       ├── Figure_S3_data_distributions.pdf
│       └── ...
│
├── tables/                            # CSV tables for manuscript
│   ├── Table_S1_circuit_depth_comparison.csv
│   ├── Table_S2_optimal_alpha_values.csv
│   ├── Table_S3_readout_comparison.csv
│   ├── Table_S4_vqr_depth_optimization.csv
│   └── Table_S5_optimizer_comparison.csv
│
├── scripts/                           # Standalone execution scripts
│   ├── run_quantum_kernel.sh          # Run QKR experiments
│   ├── run_vqr.sh                     # Run VQR experiments
│   ├── run_classical.sh               # Run classical ML
│   └── generate_all_figures.sh        # Generate all figures
│
└── docs/                              # Additional documentation
    ├── METHODS.md                     # Detailed methodology
    ├── INSTALLATION.md                # Installation guide
    └── USAGE.md                       # Usage examples
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- 16+ GB RAM (for quantum kernel with n=600)
- Linux/macOS/Windows

### Installation

#### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/qml-ccs-prediction.git
cd qml-ccs-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Using conda

```bash
# Clone the repository
git clone https://github.com/yourusername/qml-ccs-prediction.git
cd qml-ccs-prediction

# Create conda environment
conda env create -f environment.yml
conda activate qml-ccs
```

### Requirements

```txt
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0
scikit-learn==1.5.1

# Quantum ML frameworks
qiskit>=0.45.0
qiskit-machine-learning==0.7.2
qiskit-aer>=0.13.0
pennylane==0.38.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0

# Utilities
joblib>=1.1.0
scipy>=1.7.0
tqdm>=4.62.0
```

---

## 📊 Running the Experiments

### 1. Quantum Kernel Regression (QKR)

```bash
# Run QKR for all classes and training sizes
python src/quantum/quantum_kernel.py

# Or use the script
bash scripts/run_quantum_kernel.sh
```

**Expected runtime:** ~30-60 minutes per class (n=600)  
**Output:** `results/quantum_kernel_results_all_classes.csv`

### 2. Variational Quantum Regression (VQR)

```bash
# Run VQR with optimized configurations
python src/quantum/variational_quantum.py

# Or use the script
bash scripts/run_vqr.sh
```

**Expected runtime:** ~20-45 minutes per class (n=600)  
**Output:** `results/vqr_results_all_classes.csv`

### 3. Classical Machine Learning Baselines

```bash
# Run all classical models (RF, SVR, Lasso, VR)
python src/classical/classical_ml.py

# Or use the script
bash scripts/run_classical.sh
```

**Expected runtime:** ~5-10 minutes for all classes  
**Output:** `results/classical_ml_size_scaling_results.csv`

### 4. Generate Figures

```bash
# Generate all manuscript figures
python src/visualization/unified_visualization.py
python src/visualization/advanced_comparison.py

# Or use the comprehensive script
bash scripts/generate_all_figures.sh
```

**Output:** Figures saved in `figures/main/` and `figures/supplementary/`

---

## 💡 Key Implementation Details

### Quantum Kernel Regression

```python
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.kernel_ridge import KernelRidge

# 5-qubit ZZFeatureMap with 1 repetition (optimized)
feature_map = ZZFeatureMap(feature_dimension=5, reps=1)

# Fidelity-based quantum kernel
quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

# Kernel matrices
K_train = quantum_kernel.evaluate(X_train)
K_test = quantum_kernel.evaluate(X_test, X_train)

# Kernel Ridge Regression with optimized α
model = KernelRidge(alpha=optimal_alpha, kernel='precomputed')
model.fit(K_train, y_train)
predictions = model.predict(K_test)
```

**Key Parameters:**
- Feature map: `ZZFeatureMap(5 qubits, reps=1)`
- Regularization α: Class-specific (0.01-1.0)
- Training sizes: 200, 400, 600

### Variational Quantum Regression

```python
import pennylane as qml

# Circuit definition
dev = qml.device("default.qubit", wires=5)

@qml.qnode(dev, interface="autograd")
def circuit(weights, x):
    # Data encoding
    qml.AngleEmbedding(x, wires=range(5))
    
    # Variational layer (optimized: 1 layer)
    qml.BasicEntanglerLayers(weights, wires=range(5))
    
    # Measurement
    return qml.expval(qml.PauliZ(0))

# Training with class-specific optimizer
optimizer = qml.MomentumOptimizer(stepsize=0.1)  # Example
```

**Key Parameters:**
- Circuit depth: 1 layer (optimized)
- Optimizer: Class-specific (Adam, Momentum, RMSProp)
- Training steps: 30 with batch size 32
- Target normalization: Z-score

---

## 📈 Results Overview

### Overall Performance (n=600)

| Model | Carbohydrates R² | Lipids R² | Proteins R² | Lignins R² | Others R² |
|-------|------------------|-----------|-------------|------------|-----------|
| **Random Forest** | 0.97 | 0.96 | 0.90 | 0.92 | 0.89 |
| **Lasso** | 0.96 | 0.95 | 0.88 | 0.91 | 0.87 |
| **Quantum Kernel** | 0.95 | 0.94 | 0.48 | 0.56 | 0.34 |
| **VQR** | 0.45 | 0.19 | 0.06 | 0.17 | 0.02 |

### Key Insights

1. **Classical Dominance:** RF and Lasso consistently outperform quantum methods
2. **QKR Data Efficiency:** Shows strong improvement with increasing training size
3. **Class Dependency:** QKR performs best on homogeneous classes (Carbohydrates, Lipids)
4. **VQR Limitations:** Current ansatz insufficient for complex CCS prediction

---

## 📚 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{yourname2025qml,
  title={Scaling Behavior of Quantum Kernel Regression for Chemical Data: 
         Insights from Homogeneous Subsets and CCS Prediction},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2025},
  doi={XX.XXXX/XXXXXX}
}
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Areas for Contribution

- [ ] Alternative quantum feature maps (IQP, Pauli)
- [ ] Advanced VQR ansätze (hardware-efficient, problem-inspired)
- [ ] Hybrid quantum-classical architectures
- [ ] Additional molecular descriptor sets
- [ ] Real quantum hardware experiments (IBMQ, IonQ)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Qiskit team for quantum computing framework
- PennyLane team for quantum machine learning tools
- [Funding agencies]
- [Institutional support]

---

## 📧 Contact

- **Lead Author:** [Your Name] - [email@university.edu]
- **Repository:** [https://github.com/yourusername/qml-ccs-prediction](https://github.com/yourusername/qml-ccs-prediction)
- **Issues:** [GitHub Issues](https://github.com/yourusername/qml-ccs-prediction/issues)

---

## 📖 Additional Resources

- [Paper Preprint](link-to-preprint)
- [Supplementary Information](docs/SUPPLEMENTARY.pdf)
- [Tutorial Notebooks](notebooks/)
- [Video Presentation](link-if-available)

---

**Last Updated:** March 2026  
**Version:** 1.0.0
