#!/bin/bash

# Usage Example:
# bash pretrain_navi.sh

# NAVI Model Training Script
# Trains NAVI model with default parameters, hyperparameter variants, and ablation variants

set -e  # Exit on any error

# ===============================
# Configuration
# ===============================
DATASET_LIST=("Movie" "Product")
GPU_IDX=0

# Default parameters
DEFAULT_ALIGNMENT_WEIGHT=0.5
DEFAULT_HV_WEIGHT=0.8
DEFAULT_VALUE_RATIO=0.5
DEFAULT_LOW_ENTROPY_TAU=0.13

# Hyperparameter variants (excluding default)
ALIGNMENT_WEIGHT_VARIANTS=(0.25 1.0 2.0 4.0)
HV_WEIGHT_VALUE_RATIO_VARIANTS=(
    "0.8 0.25"
    "0.8 0.75"
    "0.4 0.25"
    "0.4 0.5"
    "0.4 0.75"
)
TAU_VARIANTS=(0.07 0.1)

# Ablation variants
ABLATION_VARIANTS=("woSSI" "woMSM" "woESA")

# ===============================
# Create necessary directories
# ===============================
mkdir -p logs
mkdir -p models

# ===============================
# Helper Functions
# ===============================
convert_to_safe_name() {
    echo "$1" | sed 's/\./p/g'
}

train_navi_model() {
    local dataset=$1
    local alignment_weight=$2
    local hv_weight=$3
    local value_ratio=$4
    local low_entropy_tau=$5
    local ablation_type=$6
    local model_name=$7
    local log_suffix=$8
    
    local data_path="./data/${dataset}_top100_cleaned"
    local output_dir="./models/${model_name}"
    local log_file="navi_${dataset,,}_${log_suffix}.log"
    
    echo "=========================================="
    echo "Training NAVI model: ${model_name}"
    echo "Dataset: ${dataset}"
    echo "Alignment Weight: ${alignment_weight}"
    echo "HV Weight: ${hv_weight}"
    echo "Value Ratio: ${value_ratio}"
    echo "Low Entropy Tau: ${low_entropy_tau}"
    echo "Ablation Type: ${ablation_type}"
    echo "Output: ${output_dir}"
    echo "Log: ${log_file}"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=${GPU_IDX} PYTHONPATH=$(pwd) python3 -u training/train_navi.py \
        --data_path "${data_path}" \
        --output_dir "${output_dir}" \
        --masking_strategy "HVB" \
        --ablation_type "${ablation_type}" \
        --hv_weight "${hv_weight}" \
        --alignment_weight "${alignment_weight}" \
        --value_ratio "${value_ratio}" \
        --low_entropy_tau "${low_entropy_tau}" \
        --batch_size 32 \
        --gradient_accumulation_steps 2 \
        --tables_per_batch 4 \
        --num_epochs 2 \
        --learning_rate 3e-5 \
        --weight_decay 0.01 \
        --warmup_steps 50 \
        --max_grad_norm 1.0 \
        --log_steps 100 > "logs/${log_file}" 2>&1
    
    echo "✓ NAVI training completed: ${model_name}"
    echo "---"
}

# ===============================
# Main Execution
# ===============================
echo "=========================================="
echo "Starting NAVI Model Training"
echo "Date: $(date)"
echo "GPU Index: ${GPU_IDX}"
echo "Datasets: ${DATASET_LIST[*]}"
echo "=========================================="

# Get start time
START_TIME=$(date +%s)

# ===============================
# Phase 1: Default Models
# ===============================
echo "Phase 1: Training Default NAVI Models..."
for dataset in "${DATASET_LIST[@]}"; do
    train_navi_model \
        "$dataset" \
        "$DEFAULT_ALIGNMENT_WEIGHT" \
        "$DEFAULT_HV_WEIGHT" \
        "$DEFAULT_VALUE_RATIO" \
        "$DEFAULT_LOW_ENTROPY_TAU" \
        "full" \
        "navi_${dataset,,}" \
        "default"
done

# ===============================
# Phase 2: Hyperparameter Variants
# ===============================
echo "Phase 2: Training Hyperparameter Variants..."

# Alignment Weight Variants
echo "Training Alignment Weight Variants..."
for dataset in "${DATASET_LIST[@]}"; do
    for alignment_weight in "${ALIGNMENT_WEIGHT_VARIANTS[@]}"; do
        alignment_safe=$(convert_to_safe_name "$alignment_weight")
        train_navi_model \
            "$dataset" \
            "$alignment_weight" \
            "$DEFAULT_HV_WEIGHT" \
            "$DEFAULT_VALUE_RATIO" \
            "$DEFAULT_LOW_ENTROPY_TAU" \
            "full" \
            "navi_${dataset,,}_align${alignment_safe}" \
            "align${alignment_safe}"
    done
done

# HV Weight & Value Ratio Variants
echo "Training HV Weight & Value Ratio Variants..."
for dataset in "${DATASET_LIST[@]}"; do
    for hv_value_pair in "${HV_WEIGHT_VALUE_RATIO_VARIANTS[@]}"; do
        hv_weight=$(echo $hv_value_pair | cut -d' ' -f1)
        value_ratio=$(echo $hv_value_pair | cut -d' ' -f2)
        hv_safe=$(convert_to_safe_name "$hv_weight")
        vr_safe=$(convert_to_safe_name "$value_ratio")
        train_navi_model \
            "$dataset" \
            "$DEFAULT_ALIGNMENT_WEIGHT" \
            "$hv_weight" \
            "$value_ratio" \
            "$DEFAULT_LOW_ENTROPY_TAU" \
            "full" \
            "navi_${dataset,,}_hv${hv_safe}_vr${vr_safe}" \
            "hv${hv_safe}_vr${vr_safe}"
    done
done

# Tau Variants
echo "Training Tau Variants..."
for dataset in "${DATASET_LIST[@]}"; do
    for tau in "${TAU_VARIANTS[@]}"; do
        tau_safe=$(convert_to_safe_name "$tau")
        train_navi_model \
            "$dataset" \
            "$DEFAULT_ALIGNMENT_WEIGHT" \
            "$DEFAULT_HV_WEIGHT" \
            "$DEFAULT_VALUE_RATIO" \
            "$tau" \
            "full" \
            "navi_${dataset,,}_tau${tau_safe}" \
            "tau${tau_safe}"
    done
done

# ===============================
# Phase 3: Ablation Variants
# ===============================
echo "Phase 3: Training Ablation Variants..."
for dataset in "${DATASET_LIST[@]}"; do
    for ablation in "${ABLATION_VARIANTS[@]}"; do
        train_navi_model \
            "$dataset" \
            "$DEFAULT_ALIGNMENT_WEIGHT" \
            "$DEFAULT_HV_WEIGHT" \
            "$DEFAULT_VALUE_RATIO" \
            "$DEFAULT_LOW_ENTROPY_TAU" \
            "$ablation" \
            "navi_${dataset,,}_${ablation}" \
            "$ablation"
    done
done

# ===============================
# Summary
# ===============================
# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo "=========================================="
echo "All NAVI Training Jobs Completed!"
echo "End time: $(date)"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=========================================="

echo "Generated models:"
echo ""
echo "Default Models:"
for dataset in "${DATASET_LIST[@]}"; do
    echo "- navi_${dataset,,}: ./models/navi_${dataset,,}"
done

echo ""
echo "Hyperparameter Variants:"
echo "Alignment Weight Variants:"
for dataset in "${DATASET_LIST[@]}"; do
    for alignment_weight in "${ALIGNMENT_WEIGHT_VARIANTS[@]}"; do
        alignment_safe=$(convert_to_safe_name "$alignment_weight")
        echo "- navi_${dataset,,}_align${alignment_safe}: ./models/navi_${dataset,,}_align${alignment_safe}"
    done
done

echo "HV Weight & Value Ratio Variants:"
for dataset in "${DATASET_LIST[@]}"; do
    for hv_value_pair in "${HV_WEIGHT_VALUE_RATIO_VARIANTS[@]}"; do
        hv_weight=$(echo $hv_value_pair | cut -d' ' -f1)
        value_ratio=$(echo $hv_value_pair | cut -d' ' -f2)
        hv_safe=$(convert_to_safe_name "$hv_weight")
        vr_safe=$(convert_to_safe_name "$value_ratio")
        echo "- navi_${dataset,,}_hv${hv_safe}_vr${vr_safe}: ./models/navi_${dataset,,}_hv${hv_safe}_vr${vr_safe}"
    done
done

echo "Tau Variants:"
for dataset in "${DATASET_LIST[@]}"; do
    for tau in "${TAU_VARIANTS[@]}"; do
        tau_safe=$(convert_to_safe_name "$tau")
        echo "- navi_${dataset,,}_tau${tau_safe}: ./models/navi_${dataset,,}_tau${tau_safe}"
    done
done

echo ""
echo "Ablation Variants:"
for dataset in "${DATASET_LIST[@]}"; do
    for ablation in "${ABLATION_VARIANTS[@]}"; do
        echo "- navi_${dataset,,}_${ablation}: ./models/navi_${dataset,,}_${ablation}"
    done
done

echo ""
echo "Total Models Trained:"
TOTAL_MODELS=$((2 * (1 + 4 + 5 + 2 + 3)))  # 2 datasets * (default + align + hv_vr + tau + ablation)
echo "- Default: 2 models"
echo "- Alignment Weight Variants: 8 models (4 variants × 2 datasets)"
echo "- HV Weight & Value Ratio Variants: 10 models (5 variants × 2 datasets)"
echo "- Tau Variants: 4 models (2 variants × 2 datasets)"
echo "- Ablation Variants: 6 models (3 variants × 2 datasets)"
echo "- Total: ${TOTAL_MODELS} models"

echo ""
echo "Log files are in logs/ directory with pattern: navi_<dataset>_<variant>.log"
echo ""
echo "To monitor all logs:"
echo "tail -f logs/navi_*.log"
echo ""
echo "To check GPU usage:"
echo "nvidia-smi"
