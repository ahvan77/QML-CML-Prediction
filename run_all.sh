#!/bin/bash

################################################################################
# Master Execution Script - QML vs CML for CCS Prediction
################################################################################
#
# This script runs the complete experimental workflow:
#   1. Quantum Kernel Regression
#   2. Variational Quantum Regression
#   3. Classical Machine Learning Baselines
#   4. Generate Figures and Tables
#
# Usage:
#   bash run_all_experiments.sh
#
# Or run individual components:
#   bash run_all_experiments.sh --quantum-only
#   bash run_all_experiments.sh --classical-only
#   bash run_all_experiments.sh --figures-only
#
################################################################################

set -e  # Exit on error

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR="Data/features"
EXPERIMENTAL_DIR="Data/experimental"
RESULTS_DIR="results"
FIGURES_DIR="figures"

# Parse command line arguments
QUANTUM_ONLY=false
CLASSICAL_ONLY=false
FIGURES_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quantum-only)
            QUANTUM_ONLY=true
            shift
            ;;
        --classical-only)
            CLASSICAL_ONLY=true
            shift
            ;;
        --figures-only)
            FIGURES_ONLY=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--quantum-only | --classical-only | --figures-only]"
            exit 1
            ;;
    esac
done

# ============================================================================
# SETUP
# ============================================================================

echo ""
echo "================================================================================"
echo "QML vs CML for CCS Prediction - Complete Experimental Workflow"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Data directory:         $DATA_DIR"
echo "  Experimental directory: $EXPERIMENTAL_DIR"
echo "  Results directory:      $RESULTS_DIR"
echo "  Figures directory:      $FIGURES_DIR"
echo ""
echo "================================================================================"
echo ""

# Create output directories
mkdir -p $RESULTS_DIR
mkdir -p $FIGURES_DIR/main
mkdir -p $FIGURES_DIR/supplementary
mkdir -p tables
mkdir -p models

# Record start time
START_TIME=$(date +%s)

# ============================================================================
# STEP 1: Quantum Kernel Regression
# ============================================================================

if [ "$CLASSICAL_ONLY" = false ] && [ "$FIGURES_ONLY" = false ]; then
    echo ""
    echo "================================================================================"
    echo "STEP 1/3: Quantum Kernel Regression"
    echo "================================================================================"
    echo ""
    echo "Configuration:"
    echo "  - Feature map: ZZFeatureMap (5 qubits, reps=1)"
    echo "  - Kernel: FidelityQuantumKernel"
    echo "  - Regressor: KernelRidge (α tuned via 5-fold CV)"
    echo "  - Training sizes: 200, 400, 600"
    echo "  - Expected runtime: ~30-60 minutes per class"
    echo ""
    echo "--------------------------------------------------------------------------------"
    
    python src/quantum/quantum_kernel_regression.py \
        --data-dir $DATA_DIR \
        --experimental-dir $EXPERIMENTAL_DIR \
        --output-dir $RESULTS_DIR
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Quantum Kernel Regression completed successfully"
    else
        echo ""
        echo "❌ Quantum Kernel Regression failed"
        exit 1
    fi
fi

# ============================================================================
# STEP 2: Variational Quantum Regression
# ============================================================================

if [ "$CLASSICAL_ONLY" = false ] && [ "$FIGURES_ONLY" = false ]; then
    echo ""
    echo "================================================================================"
    echo "STEP 2/3: Variational Quantum Regression"
    echo "================================================================================"
    echo ""
    echo "Configuration:"
    echo "  - Circuit: 5 qubits, AngleEmbedding + 2 BasicEntanglerLayers"
    echo "  - Parameters: 10 trainable (2 layers × 5 qubits)"
    echo "  - Training: 30 steps, batch size 32"
    echo "  - Optimizer: Class-specific (from Table S5)"
    echo "  - Expected runtime: ~20-45 minutes per class"
    echo ""
    echo "--------------------------------------------------------------------------------"
    
    python src/quantum/variational_quantum_regression.py \
        --data-dir $DATA_DIR \
        --experimental-dir $EXPERIMENTAL_DIR \
        --output-dir $RESULTS_DIR
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Variational Quantum Regression completed successfully"
    else
        echo ""
        echo "❌ Variational Quantum Regression failed"
        exit 1
    fi
fi

# ============================================================================
# STEP 3: Classical Machine Learning Baselines
# ============================================================================

if [ "$QUANTUM_ONLY" = false ] && [ "$FIGURES_ONLY" = false ]; then
    echo ""
    echo "================================================================================"
    echo "STEP 3/3: Classical Machine Learning Baselines"
    echo "================================================================================"
    echo ""
    echo "Models:"
    echo "  - Random Forest"
    echo "  - Support Vector Regression (RBF kernel)"
    echo "  - Lasso Regression"
    echo "  - Voting Regressor (ensemble)"
    echo ""
    echo "Configuration:"
    echo "  - Training sizes: 200, 400, 600"
    echo "  - Expected runtime: ~5-10 minutes"
    echo ""
    echo "--------------------------------------------------------------------------------"
    
    python src/classical/classical_baselines.py \
        --data-dir $DATA_DIR \
        --experimental-dir $EXPERIMENTAL_DIR \
        --output-dir $RESULTS_DIR
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Classical ML Baselines completed successfully"
    else
        echo ""
        echo "❌ Classical ML Baselines failed"
        exit 1
    fi
fi

# ============================================================================
# STEP 4: Generate Figures and Tables
# ============================================================================

if [ "$QUANTUM_ONLY" = false ] && [ "$CLASSICAL_ONLY" = false ]; then
    echo ""
    echo "================================================================================"
    echo "STEP 4/4: Generating Figures and Tables"
    echo "================================================================================"
    echo ""
    echo "Generating:"
    echo "  - Main manuscript figures (Figures 4-5)"
    echo "  - Supplementary figures (Figures S3-S6)"
    echo "  - Tables (Table 1, Tables S4-S6)"
    echo ""
    echo "Expected runtime: ~2-5 minutes"
    echo ""
    echo "--------------------------------------------------------------------------------"
    
    # Main figures
    echo "Generating main figures..."
    python src/visualization/unified_visualization.py
    
    if [ $? -ne 0 ]; then
        echo "⚠️  Warning: Main figure generation had issues"
    fi
    
    # Supplementary figures
    echo "Generating supplementary figures..."
    python src/visualization/advanced_comparison.py
    
    if [ $? -ne 0 ]; then
        echo "⚠️  Warning: Supplementary figure generation had issues"
    fi
    
    echo ""
    echo "✅ Figures and tables generated"
fi

# ============================================================================
# SUMMARY
# ============================================================================

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "================================================================================"
echo "✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY"
echo "================================================================================"
echo ""
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Results saved in:"
echo "  📁 $RESULTS_DIR/"
echo "      ├── quantum_kernel_results_all_classes.csv"
echo "      ├── vqr_results_all_classes.csv"
echo "      ├── classical_ml_size_scaling_results.csv"
echo "      ├── pred_vs_obs_quantum/"
echo "      ├── pred_vs_obs_vqr/"
echo "      ├── pred_vs_obs/"
echo "      ├── experimental_predictions_quantum/"
echo "      ├── experimental_predictions_vqr/"
echo "      └── experimental_predictions/"
echo ""
echo "  📊 $FIGURES_DIR/"
echo "      ├── main/"
echo "      │   ├── Figure_4_test_predictions_n600.pdf"
echo "      │   └── Figure_5_experimental_predictions.pdf"
echo "      └── supplementary/"
echo "          ├── Figure_S3_combined_quantum_vs_classical.pdf"
echo "          ├── Figure_S4_residual_plots_n600.pdf"
echo "          ├── Figure_S5_training_size_effect.pdf"
echo "          └── Figure_S6_experimental_distributions.pdf"
echo ""
echo "  📋 tables/"
echo "      ├── Table_1_model_performance_summary.csv"
echo "      ├── Table_S4_comprehensive_prediction_metrics.csv"
echo "      ├── Table_S5_experimental_prediction_statistics.csv"
echo "      └── Table_S6_statistical_comparisons.csv"
echo ""
echo "Next steps:"
echo "  1. Review results in $RESULTS_DIR/"
echo "  2. Check figures in $FIGURES_DIR/"
echo "  3. Verify tables in tables/"
echo "  4. Compare with manuscript figures and tables"
echo ""
echo "================================================================================"
echo ""
