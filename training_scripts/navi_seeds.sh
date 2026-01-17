#!/bin/bash

# Usage Example:
# bash pretrain_navi_seed_variance.sh

# NAVI Model Training Script - Seed Variance Only
# Trains NAVI models with different seeds for variance analysis

set -e  # Exit on any error

# ===============================
# Configuration
# ===============================
DATASET_LIST=("Movie" "Product")

# Default parameters
DEFAULT_ALIGNMENT_WEIGHT=0.5
DEFAULT_HV_WEIGHT=0.8
DEFAULT_VALUE_RATIO=0.5
DEFAULT_LOW_ENTROPY_TAU=0.13
DEFAULT_HIGH_ENTROPY_TAU=0.07

# Seeds for variance analysis
SEEDS=(1 2)

# ===============================
# Create necessary directories
# ===============================
mkdir -p logs
mkdir -p models

# ===============================
# Training Function
# ===============================
train_navi_model() {
    local dataset=$1
    local alignment_weight=$2
    local hv_weight=$3
    local value_ratio=$4
    local low_entropy_tau=$5
    local high_entropy_tau=$6
    local ablation_type=$7
    local model_name=$8
    local log_suffix=$9
    local seed=${10:-42}
    local entropy_threshold_method=${11:-"quartile"}
    local low_threshold_percentile=${12:-25}
    local high_threshold_percentile=${13:-75}
    local gradient_accumulation_steps=${14:-2}
    local num_epochs=${15:-2}
    local header_encoder_mode=${16:-"full"}
    local gpu_idx=${17:-0}
    
    local data_path="./data/cleaned/${dataset}/train"
    local validation_dir="./data/cleaned/${dataset}/validation"
    local output_dir="./models/${model_name}"
    local log_file="navi_${dataset,,}_${log_suffix}_seed${seed}.log"
    
    echo "=========================================="
    echo "Training NAVI model: ${model_name}"
    echo "Dataset: ${dataset}"
    echo "GPU: ${gpu_idx}"
    echo "Seed: ${seed}"
    echo "Output: ${output_dir}"
    echo "Log: ${log_file}"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=${gpu_idx} PYTHONPATH=$(pwd) python3 -u training/train_navi.py \
        --data_path "${data_path}" \
        --validation_dir "${validation_dir}" \
        --output_dir "${output_dir}" \
        --masking_strategy "HVB" \
        --ablation_type "${ablation_type}" \
        --hv_weight "${hv_weight}" \
        --alignment_weight "${alignment_weight}" \
        --value_ratio "${value_ratio}" \
        --low_entropy_tau "${low_entropy_tau}" \
        --high_entropy_tau "${high_entropy_tau}" \
        --seed "${seed}" \
        --data_type "cleaned" \
        --entropy_threshold_method "${entropy_threshold_method}" \
        --low_threshold_percentile "${low_threshold_percentile}" \
        --high_threshold_percentile "${high_threshold_percentile}" \
        --batch_size 32 \
        --gradient_accumulation_steps "${gradient_accumulation_steps}" \
        --tables_per_batch 4 \
        --num_epochs "${num_epochs}" \
        --learning_rate 3e-5 \
        --weight_decay 0.01 \
        --warmup_steps 50 \
        --max_grad_norm 1.0 \
        --log_steps 100 \
        --header_encoder_mode "${header_encoder_mode}" > "logs/${log_file}" 2>&1
    
    echo "✓ NAVI training completed: ${model_name}"
    echo "---"
}

# ===============================
# Main Execution
# ===============================
echo "=========================================="
echo "Starting NAVI Seed Variance Training"
echo "Date: $(date)"
echo "GPU Assignment: Movie -> GPU 0, Product -> GPU 1"
echo "Datasets: ${DATASET_LIST[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "=========================================="

# Get start time
START_TIME=$(date +%s)

# ===============================
# Seed Variance Training
# ===============================
echo "Training NAVI Models with seed variance..."
for i in "${!DATASET_LIST[@]}"; do
    dataset="${DATASET_LIST[$i]}"
    gpu_idx=$i  # Movie (index 0) -> GPU 0, Product (index 1) -> GPU 1
    
    for seed in "${SEEDS[@]}"; do
        train_navi_model \
            "$dataset" \
            "$DEFAULT_ALIGNMENT_WEIGHT" \
            "$DEFAULT_HV_WEIGHT" \
            "$DEFAULT_VALUE_RATIO" \
            "$DEFAULT_LOW_ENTROPY_TAU" \
            "$DEFAULT_HIGH_ENTROPY_TAU" \
            "full" \
            "navi_${dataset,,}_seed${seed}" \
            "default" \
            "$seed" \
            "quartile" \
            "25" \
            "75" \
            "2" \
            "2" \
            "full" \
            "$gpu_idx" &
    done
done

# Wait for all background jobs to complete
wait

# ===============================
# Summary
# ===============================
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo "=========================================="
echo "Seed Variance Training Completed!"
echo "End time: $(date)"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=========================================="

echo "Generated models:"
for dataset in "${DATASET_LIST[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "- navi_${dataset,,}_seed${seed}: ./models/navi_${dataset,,}_seed${seed}"
    done
done

echo ""
echo "Log files are in logs/ directory"
echo "To monitor logs: tail -f logs/navi_*_default_seed*.log"