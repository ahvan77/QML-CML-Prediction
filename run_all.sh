#!/bin/bash

# ============================================================================
# Complete Workflow Script for QML-CCS Prediction
# ============================================================================
# This script runs the complete experimental pipeline:
# 1. Quantum Kernel Regression
# 2. Variational Quantum Regression  
# 3. Classical Machine Learning
# 4. Figure and Table Generation
# ============================================================================

set -e  # Exit on error

echo "============================================================================"
echo "QML vs CML for CCS Prediction - Complete Workflow"
echo "============================================================================"
echo ""

# Configuration
DATA_DIR="Data/features"
EXPERIMENTAL_DIR="Data/experimental"
RESULTS_DIR="results"
FIGURES_DIR="figures"

# Create output directories
mkdir -p $RESULTS_DIR
mkdir -p $FIGURES_DIR/main
mkdir -p $FIGURES_DIR/supplementary
mkdir -p tables

# ============================================================================
# Step 1: Quantum Kernel Regression
# ============================================================================

echo "Step 1/4: Running Quantum Kernel Regression..."
echo "Expected runtime: ~30-60 minutes per class"
echo "----------------------------------------------------------------------"

python src/quantum/quantum_kernel.py \
    --data-dir $DATA_DIR \
    --experimental-dir $EXPERIMENTAL_DIR \
    --train-sizes 200 400 600 \
    --classes Carbohydrate Lignin Lipid Protein Others \
    --output-dir $RESULTS_DIR

if [ $? -eq 0 ]; then
    echo "✅ Quantum Kernel Regression completed successfully"
else
    echo "❌ Quantum Kernel Regression failed"
    exit 1
fi

echo ""

# ============================================================================
# Step 2: Variational Quantum Regression
# ============================================================================

echo "Step 2/4: Running Variational Quantum Regression..."
echo "Expected runtime: ~20-45 minutes per class"
echo "----------------------------------------------------------------------"

python src/quantum/variational_quantum.py \
    --data-dir $DATA_DIR \
    --experimental-dir $EXPERIMENTAL_DIR \
    --train-sizes 200 400 600 \
    --classes Carbohydrate Lignin Lipid Protein Others \
    --n-layers 1 \
    --steps 30 \
    --batch-size 32 \
    --output-dir $RESULTS_DIR

if [ $? -eq 0 ]; then
    echo "✅ Variational Quantum Regression completed successfully"
else
    echo "❌ Variational Quantum Regression failed"
    exit 1
fi

echo ""

# ============================================================================
# Step 3: Classical Machine Learning
# ============================================================================

echo "Step 3/4: Running Classical Machine Learning Baselines..."
echo "Expected runtime: ~5-10 minutes"
echo "----------------------------------------------------------------------"

python src/classical/classical_ml.py \
    --data-dir $DATA_DIR \
    --experimental-dir $EXPERIMENTAL_DIR \
    --train-sizes 200 400 600 \
    --models RandomForest SVR Lasso VotingRegressor \
    --output-dir $RESULTS_DIR

if [ $? -eq 0 ]; then
    echo "✅ Classical ML Baselines completed successfully"
else
    echo "❌ Classical ML Baselines failed"
    exit 1
fi

echo ""

# ============================================================================
# Step 4: Generate Figures and Tables
# ============================================================================

echo "Step 4/4: Generating Figures and Tables..."
echo "Expected runtime: ~2-5 minutes"
echo "----------------------------------------------------------------------"

# Main figures
python src/visualization/unified_visualization.py

if [ $? -eq 0 ]; then
    echo "✅ Main figures generated successfully"
else
    echo "❌ Main figure generation failed"
    exit 1
fi

# Supplementary figures
python src/visualization/advanced_comparison.py

if [ $? -eq 0 ]; then
    echo "✅ Supplementary figures generated successfully"
else
    echo "❌ Supplementary figure generation failed"
    exit 1
fi

echo ""

# ============================================================================
# Summary
# ============================================================================

echo "============================================================================"
echo "✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY"
echo "============================================================================"
echo ""
echo "Results saved in:"
echo "  - $RESULTS_DIR/"
echo "  - $FIGURES_DIR/"
echo "  - tables/"
echo ""
echo "Key Output Files:"
echo "  - quantum_kernel_results_all_classes.csv"
echo "  - vqr_results_all_classes.csv"
echo "  - classical_ml_size_scaling_results.csv"
echo "  - Figure_4_test_predictions_n600.pdf"
echo "  - Figure_5_experimental_predictions.pdf"
echo "  - Table_1_model_performance_summary.csv"
echo ""
echo "Next steps:"
echo "  1. Review results in $RESULTS_DIR/"
echo "  2. Check figures in $FIGURES_DIR/"
echo "  3. Compare with paper results in paper_results/"
echo ""
echo "============================================================================"

# ============================================================================
# INDIVIDUAL SCRIPTS
# ============================================================================

# Create individual execution scripts

# run_quantum_kernel.sh
cat > scripts/run_quantum_kernel.sh << 'EOF'
#!/bin/bash
set -e

echo "Running Quantum Kernel Regression..."
python src/quantum/quantum_kernel.py \
    --data-dir Data/features \
    --experimental-dir Data/experimental \
    --train-sizes 200 400 600 \
    --classes Carbohydrate Lignin Lipid Protein Others \
    --output-dir results

echo "✅ Quantum Kernel Regression completed"
EOF

chmod +x scripts/run_quantum_kernel.sh

# run_vqr.sh
cat > scripts/run_vqr.sh << 'EOF'
#!/bin/bash
set -e

echo "Running Variational Quantum Regression..."
python src/quantum/variational_quantum.py \
    --data-dir Data/features \
    --experimental-dir Data/experimental \
    --train-sizes 200 400 600 \
    --classes Carbohydrate Lignin Lipid Protein Others \
    --n-layers 1 \
    --steps 30 \
    --batch-size 32 \
    --output-dir results

echo "✅ VQR completed"
EOF

chmod +x scripts/run_vqr.sh

# run_classical.sh
cat > scripts/run_classical.sh << 'EOF'
#!/bin/bash
set -e

echo "Running Classical Machine Learning..."
python src/classical/classical_ml.py \
    --data-dir Data/features \
    --experimental-dir Data/experimental \
    --train-sizes 200 400 600 \
    --models RandomForest SVR Lasso VotingRegressor \
    --output-dir results

echo "✅ Classical ML completed"
EOF

chmod +x scripts/run_classical.sh

# generate_all_figures.sh
cat > scripts/generate_all_figures.sh << 'EOF'
#!/bin/bash
set -e

echo "Generating all figures and tables..."

# Main figures
python src/visualization/unified_visualization.py

# Supplementary figures
python src/visualization/advanced_comparison.py

echo "✅ All figures generated"
echo "Check figures/ and tables/ directories"
EOF

chmod +x scripts/generate_all_figures.sh

echo "Individual scripts created in scripts/ directory"
