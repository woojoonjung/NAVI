#!/bin/bash

# Usage Example:
# bash training_scripts/pretrain_baselines.sh
# bash training_scripts/pretrain_baselines.sh Movie
# bash training_scripts/pretrain_baselines.sh Product 0

# Combined script to train all baseline models: BERT, HAETAE, and TAPAS
# This script runs all baseline training sequentially for comprehensive evaluation

set -euo pipefail  # Exit on any error / unset var / pipefail

# Always run from repo root (so relative paths work)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

# ===============================
# Configuration
# ===============================
# Domain selection:
# - default: both Movie + Product
# - pass "Movie" or "Product" as first argument to run one domain only
DOMAIN_ARG="${1:-Both}"
GPU_IDX="${2:-1}"

case "${DOMAIN_ARG,,}" in
  movie)
    DATASET_LIST=("Movie")
    ;;
  product)
    DATASET_LIST=("Product")
    ;;
  both|"")
    DATASET_LIST=("Movie" "Product")
    ;;
  *)
    echo "Unknown domain: ${DOMAIN_ARG}"
    echo "Usage: bash training_scripts/pretrain_baselines.sh [Movie|Product|Both] [GPU_IDX]"
    exit 2
    ;;
esac

# Match your current NAVI training defaults (override via env if needed)
DEFAULT_BATCH_SIZE="${DEFAULT_BATCH_SIZE:-32}"
DEFAULT_GRADIENT_ACCUMULATION_STEPS="${DEFAULT_GRADIENT_ACCUMULATION_STEPS:-1}"
DEFAULT_TABLES_PER_BATCH="${DEFAULT_TABLES_PER_BATCH:-4}"

# NAVI-specific defaults
DEFAULT_ALIGNMENT_WEIGHT="${DEFAULT_ALIGNMENT_WEIGHT:-0.05}"
DEFAULT_HV_WEIGHT="${DEFAULT_HV_WEIGHT:-0.8}"
DEFAULT_VALUE_RATIO="${DEFAULT_VALUE_RATIO:-0.5}"
DEFAULT_LOW_ENTROPY_TAU="${DEFAULT_LOW_ENTROPY_TAU:-0.1}"
DEFAULT_HIGH_ENTROPY_TAU="${DEFAULT_HIGH_ENTROPY_TAU:-0.02}"
DEFAULT_ENTROPY_THRESHOLD_METHOD="${DEFAULT_ENTROPY_THRESHOLD_METHOD:-percentile}"
DEFAULT_LOW_THRESHOLD_PERCENTILE="${DEFAULT_LOW_THRESHOLD_PERCENTILE:-10}"
DEFAULT_HIGH_THRESHOLD_PERCENTILE="${DEFAULT_HIGH_THRESHOLD_PERCENTILE:-90}"

# Create necessary directories
mkdir -p logs
mkdir -p models

# Function to train BERT model
train_bert() {
    local dataset=$1
    local data_path="./data/cleaned/${dataset}/train"
    local output_dir="./models/bert_${dataset,,}"
    local log_file="bert_${dataset,,}.log"
    
    echo "=========================================="
    echo "Starting BERT training for ${dataset} dataset..."
    echo "Data path: ${data_path}"
    echo "Output directory: ${output_dir}"
    echo "Log file: ${log_file}"
    echo "Batch size: ${DEFAULT_BATCH_SIZE} | Grad accum: ${DEFAULT_GRADIENT_ACCUMULATION_STEPS} | Tables/batch: ${DEFAULT_TABLES_PER_BATCH}"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=${GPU_IDX} PYTHONPATH=$(pwd) python3 -u training/train_bert.py \
        --data_path "${data_path}" \
        --output_dir "${output_dir}" \
        --batch_size "${DEFAULT_BATCH_SIZE}" \
        --gradient_accumulation_steps "${DEFAULT_GRADIENT_ACCUMULATION_STEPS}" \
        --tables_per_batch "${DEFAULT_TABLES_PER_BATCH}" > "logs/${log_file}" 2>&1
    
    echo "✓ BERT training for ${dataset} completed."
    echo "---"
}

# Function to train HAETAE model
train_haetae() {
    local dataset=$1
    local data_path="./data/cleaned/${dataset}/train"
    local output_dir="./models/haetae_${dataset,,}"
    local log_file="haetae_${dataset,,}.log"
    
    echo "=========================================="
    echo "Starting HAETAE training for ${dataset} dataset..."
    echo "Data path: ${data_path}"
    echo "Output directory: ${output_dir}"
    echo "Log file: ${log_file}"
    echo "Batch size: ${DEFAULT_BATCH_SIZE} | Grad accum: ${DEFAULT_GRADIENT_ACCUMULATION_STEPS} | Tables/batch: ${DEFAULT_TABLES_PER_BATCH}"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=${GPU_IDX} PYTHONPATH=$(pwd) python3 -u baselines/haetae/train.py \
        --data_path "${data_path}" \
        --output_dir "${output_dir}" \
        --batch_size "${DEFAULT_BATCH_SIZE}" \
        --gradient_accumulation_steps "${DEFAULT_GRADIENT_ACCUMULATION_STEPS}" \
        --tables_per_batch "${DEFAULT_TABLES_PER_BATCH}" > "logs/${log_file}" 2>&1
    
    echo "✓ HAETAE training for ${dataset} completed."
    echo "---"
}

# Function to train TAPAS model
train_tapas() {
    local dataset=$1
    local data_path="./data/cleaned/${dataset}/train"
    local output_dir="./models/tapas_${dataset,,}"
    local log_file="tapas_${dataset,,}.log"
    
    echo "=========================================="
    echo "Starting TAPAS training for ${dataset} dataset..."
    echo "Data path: ${data_path}"
    echo "Output directory: ${output_dir}"
    echo "Log file: ${log_file}"
    echo "Batch size: ${DEFAULT_BATCH_SIZE} | Grad accum: ${DEFAULT_GRADIENT_ACCUMULATION_STEPS} | Tables/batch: ${DEFAULT_TABLES_PER_BATCH}"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=${GPU_IDX} PYTHONPATH=$(pwd) python3 -u training/train_tapas.py \
        --data_path "${data_path}" \
        --output_dir "${output_dir}" \
        --batch_size "${DEFAULT_BATCH_SIZE}" \
        --gradient_accumulation_steps "${DEFAULT_GRADIENT_ACCUMULATION_STEPS}" \
        --tables_per_batch "${DEFAULT_TABLES_PER_BATCH}" \
        --pretrained_model "google/tapas-base-masklm" > "logs/${log_file}" 2>&1
    
    echo "✓ TAPAS training for ${dataset} completed."
    echo "---"
}

# Function to train NAVI model
train_navi() {
    local dataset=$1
    local data_path="./data/cleaned/${dataset}/train"
    local validation_dir="./data/cleaned/${dataset}/validation"
    local output_dir="./models/navi_${dataset,,}"
    local log_file="navi_${dataset,,}.log"
    
    echo "=========================================="
    echo "Starting NAVI training for ${dataset} dataset..."
    echo "Data path: ${data_path}"
    echo "Validation path: ${validation_dir}"
    echo "Output directory: ${output_dir}"
    echo "Log file: ${log_file}"
    echo "Batch size: ${DEFAULT_BATCH_SIZE} | Grad accum: ${DEFAULT_GRADIENT_ACCUMULATION_STEPS} | Tables/batch: ${DEFAULT_TABLES_PER_BATCH}"
    echo "Alignment weight: ${DEFAULT_ALIGNMENT_WEIGHT} | HV weight: ${DEFAULT_HV_WEIGHT} | Value ratio: ${DEFAULT_VALUE_RATIO}"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=${GPU_IDX} PYTHONPATH=$(pwd) python3 -u training/train_navi.py \
        --data_path "${data_path}" \
        --validation_dir "${validation_dir}" \
        --output_dir "${output_dir}" \
        --masking_strategy "HVB" \
        --ablation_type "full" \
        --hv_weight "${DEFAULT_HV_WEIGHT}" \
        --alignment_weight "${DEFAULT_ALIGNMENT_WEIGHT}" \
        --value_ratio "${DEFAULT_VALUE_RATIO}" \
        --low_entropy_tau "${DEFAULT_LOW_ENTROPY_TAU}" \
        --high_entropy_tau "${DEFAULT_HIGH_ENTROPY_TAU}" \
        --seed 42 \
        --data_type "cleaned" \
        --entropy_threshold_method "${DEFAULT_ENTROPY_THRESHOLD_METHOD}" \
        --low_threshold_percentile "${DEFAULT_LOW_THRESHOLD_PERCENTILE}" \
        --high_threshold_percentile "${DEFAULT_HIGH_THRESHOLD_PERCENTILE}" \
        --batch_size "${DEFAULT_BATCH_SIZE}" \
        --gradient_accumulation_steps "${DEFAULT_GRADIENT_ACCUMULATION_STEPS}" \
        --tables_per_batch "${DEFAULT_TABLES_PER_BATCH}" \
        --num_epochs 2 \
        --learning_rate 3e-5 \
        --weight_decay 0.01 \
        --warmup_steps 50 \
        --max_grad_norm 1.0 \
        --log_steps 100 \
        --header_encoder_mode "full" > "logs/${log_file}" 2>&1
    
    echo "✓ NAVI training for ${dataset} completed."
    echo "---"
}

# Main execution
echo "=========================================="
echo "Starting Baseline Models Pretraining"
echo "Date: $(date)"
echo "GPU Index: ${GPU_IDX}"
echo "Datasets: ${DATASET_LIST[*]}"
echo "Models: BERT, HAETAE, TAPAS, NAVI"
echo "Batch size: ${DEFAULT_BATCH_SIZE} | Grad accum: ${DEFAULT_GRADIENT_ACCUMULATION_STEPS} | Tables/batch: ${DEFAULT_TABLES_PER_BATCH}"
echo "=========================================="

# Get start time
START_TIME=$(date +%s)

# Train NAVI for all datasets
echo "Training NAVI models..."
for dataset in "${DATASET_LIST[@]}"; do
    train_navi "$dataset"
done

# Train BERT for all datasets
echo "Training BERT models..."
for dataset in "${DATASET_LIST[@]}"; do
    train_bert "$dataset"
done

# Train HAETAE for all datasets
echo "Training HAETAE models..."
for dataset in "${DATASET_LIST[@]}"; do
    train_haetae "$dataset"
done

# Train TAPAS for all datasets
echo "Training TAPAS models..."
for dataset in "${DATASET_LIST[@]}"; do
    train_tapas "$dataset"
done


# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo "=========================================="
echo "All Baseline Training Jobs Completed!"
echo "End time: $(date)"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=========================================="

echo "Generated models:"
echo "- BERT: ./models/bert_movie, ./models/bert_product"
echo "- HAETAE: ./models/haetae_movie, ./models/haetae_product"
echo "- TAPAS: ./models/tapas_movie, ./models/tapas_product"
echo "- NAVI: ./models/navi_movie, ./models/navi_product"

echo ""
echo "Log files:"
echo "- BERT: logs/bert_movie.log, logs/bert_product.log"
echo "- HAETAE: logs/haetae_movie.log, logs/haetae_product.log"
echo "- TAPAS: logs/tapas_movie.log, logs/tapas_product.log"
echo "- NAVI: logs/navi_movie.log, logs/navi_product.log"

echo ""
echo "To monitor all logs simultaneously:"
echo "tail -f logs/bert_*.log logs/haetae_*.log logs/tapas_*.log logs/navi_*.log"

echo ""
echo "To check GPU usage:"
echo "nvidia-smi"
