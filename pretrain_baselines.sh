#!/bin/bash

# Usage Example:
# bash pretrain_baselines.sh

# Combined script to train all baseline models: BERT, HAETAE, and TAPAS
# This script runs all baseline training sequentially for comprehensive evaluation

set -e  # Exit on any error

# Configuration
DATASET_LIST=("Movie" "Product")
GPU_IDX=0

# Create necessary directories
mkdir -p logs
mkdir -p models

# Function to train BERT model
train_bert() {
    local dataset=$1
    local data_path="./data/${dataset}_top100_cleaned"
    local output_dir="./models/bert_${dataset,,}"
    local log_file="bert_${dataset,,}.log"
    
    echo "=========================================="
    echo "Starting BERT training for ${dataset} dataset..."
    echo "Data path: ${data_path}"
    echo "Output directory: ${output_dir}"
    echo "Log file: ${log_file}"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=${GPU_IDX} PYTHONPATH=$(pwd) python3 -u training/train_bert.py \
        --data_path "${data_path}" \
        --output_dir "${output_dir}" > "logs/${log_file}" 2>&1
    
    echo "✓ BERT training for ${dataset} completed."
    echo "---"
}

# Function to train HAETAE model
train_haetae() {
    local dataset=$1
    local data_path="./data/${dataset}_top100_cleaned"
    local output_dir="./models/haetae_${dataset,,}"
    local log_file="haetae_${dataset,,}.log"
    
    echo "=========================================="
    echo "Starting HAETAE training for ${dataset} dataset..."
    echo "Data path: ${data_path}"
    echo "Output directory: ${output_dir}"
    echo "Log file: ${log_file}"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=${GPU_IDX} PYTHONPATH=$(pwd) python3 -u baselines/haetae/train.py \
        --data_path "${data_path}" \
        --output_dir "${output_dir}" > "logs/${log_file}" 2>&1
    
    echo "✓ HAETAE training for ${dataset} completed."
    echo "---"
}

# Function to train TAPAS model
train_tapas() {
    local dataset=$1
    local data_path="./data/${dataset}_top100_cleaned"
    local output_dir="./models/tapas_${dataset,,}"
    local log_file="tapas_${dataset,,}.log"
    
    echo "=========================================="
    echo "Starting TAPAS training for ${dataset} dataset..."
    echo "Data path: ${data_path}"
    echo "Output directory: ${output_dir}"
    echo "Log file: ${log_file}"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=${GPU_IDX} PYTHONPATH=$(pwd) python3 -u training/train_tapas.py \
        --data_path "${data_path}" \
        --output_dir "${output_dir}" \
        --pretrained_model "google/tapas-base-masklm" > "logs/${log_file}" 2>&1
    
    echo "✓ TAPAS training for ${dataset} completed."
    echo "---"
}

# Main execution
echo "=========================================="
echo "Starting Baseline Models Pretraining"
echo "Date: $(date)"
echo "GPU Index: ${GPU_IDX}"
echo "Datasets: ${DATASET_LIST[*]}"
echo "Models: BERT, HAETAE, TAPAS"
echo "=========================================="

# Get start time
START_TIME=$(date +%s)

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

echo ""
echo "Log files:"
echo "- BERT: logs/bert_movie.log, logs/bert_product.log"
echo "- HAETAE: logs/haetae_movie.log, logs/haetae_product.log"
echo "- TAPAS: logs/tapas_movie.log, logs/tapas_product.log"

echo ""
echo "To monitor all logs simultaneously:"
echo "tail -f logs/bert_*.log logs/haetae_*.log logs/tapas_*.log"

echo ""
echo "To check GPU usage:"
echo "nvidia-smi"
