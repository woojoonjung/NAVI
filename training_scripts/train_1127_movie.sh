#!/bin/bash

# Usage Example:
# bash train_custom_configs_sequential.sh

# NAVI Model Training Script (Custom Configurations - Sequential)
# Trains 4 specific NAVI model configurations
# Runs 1 job at a time (sequential execution)

set -e  # Exit on any error

# ===============================
# Configuration
# ===============================
DATASET_LIST=("Movie")
GPU_IDX=0

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
    echo "Alignment Weight: ${alignment_weight}"
    echo "HV Weight: ${hv_weight}"
    echo "Value Ratio: ${value_ratio}"
    echo "Low Entropy Tau: ${low_entropy_tau}"
    echo "High Entropy Tau: ${high_entropy_tau}"
    echo "Ablation Type: ${ablation_type}"
    echo "Seed: ${seed}"
    echo "Entropy Threshold Method: ${entropy_threshold_method}"
    if [ "${entropy_threshold_method}" = "percentile" ]; then
        echo "Low Threshold Percentile: ${low_threshold_percentile}"
        echo "High Threshold Percentile: ${high_threshold_percentile}"
    fi
    echo "Gradient Accumulation Steps: ${gradient_accumulation_steps}"
    echo "Number of Epochs: ${num_epochs}"
    echo "Header Encoder Mode: ${header_encoder_mode}"
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
echo "Starting NAVI Model Training (Custom Configurations - Sequential)"
echo "Date: $(date)"
echo "GPU: ${GPU_IDX}"
echo "Datasets: ${DATASET_LIST[*]}"
echo "Running 1 job at a time (sequential execution)"
echo "=========================================="

# Get start time
START_TIME=$(date +%s)

# Collect all job commands into an array
declare -a ALL_JOBS=()

# ===============================
# Custom Configurations
# ===============================
# Format: high_entropy_tau low_entropy_tau alignment_weight
CUSTOM_CONFIGS=(
    "0.01 0.14 0.1"
    "0.02 0.14 0.05"
    "0.01 0.14 0.05"
)

# Fixed parameters for all configurations
HV_WEIGHT=0.8
VALUE_RATIO=0.5
ENTROPY_THRESHOLD_METHOD="percentile"
LOW_THRESHOLD_PERCENTILE=10
HIGH_THRESHOLD_PERCENTILE=90
ABLATION_TYPE="full"
SEED=42
GRADIENT_ACCUMULATION_STEPS=2
NUM_EPOCHS=2
HEADER_ENCODER_MODE="full"

echo "Collecting Custom Configurations..."
for dataset in "${DATASET_LIST[@]}"; do
    for config in "${CUSTOM_CONFIGS[@]}"; do
        high_tau=$(echo $config | cut -d' ' -f1)
        low_tau=$(echo $config | cut -d' ' -f2)
        align_weight=$(echo $config | cut -d' ' -f3)
        
        high_tau_safe=$(convert_to_safe_name "$high_tau")
        low_tau_safe=$(convert_to_safe_name "$low_tau")
        align_safe=$(convert_to_safe_name "$align_weight")
        
        # Create model name: navi_<dataset>_tau<high>_<low>_align<align>_ethresh10_90
        model_name="navi_${dataset,,}_tau${high_tau_safe}_${low_tau_safe}_align${align_safe}_ethresh10_90"
        log_suffix="tau${high_tau_safe}_${low_tau_safe}_align${align_safe}_ethresh10_90"
        
        ALL_JOBS+=("train_navi_model \
            \"$dataset\" \
            \"$align_weight\" \
            \"$HV_WEIGHT\" \
            \"$VALUE_RATIO\" \
            \"$low_tau\" \
            \"$high_tau\" \
            \"$ABLATION_TYPE\" \
            \"$model_name\" \
            \"$log_suffix\" \
            \"$SEED\" \
            \"$ENTROPY_THRESHOLD_METHOD\" \
            \"$LOW_THRESHOLD_PERCENTILE\" \
            \"$HIGH_THRESHOLD_PERCENTILE\" \
            \"$GRADIENT_ACCUMULATION_STEPS\" \
            \"$NUM_EPOCHS\" \
            \"$HEADER_ENCODER_MODE\" \
            \"$GPU_IDX\"")
    done
done

# ===============================
# Execute jobs sequentially (1 at a time)
# ===============================
echo ""
echo "Total jobs collected: ${#ALL_JOBS[@]}"
echo "Executing sequentially (1 job at a time)..."
echo ""

total_jobs=${#ALL_JOBS[@]}

for ((i=0; i<total_jobs; i++)); do
    job="${ALL_JOBS[$i]}"
    echo "Job $((i+1)) of $total_jobs"
    eval "$job"
    echo "  Job $((i+1)) completed."
    echo "---"
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

for dataset in "${DATASET_LIST[@]}"; do
    for config in "${CUSTOM_CONFIGS[@]}"; do
        high_tau=$(echo $config | cut -d' ' -f1)
        low_tau=$(echo $config | cut -d' ' -f2)
        align_weight=$(echo $config | cut -d' ' -f3)
        
        high_tau_safe=$(convert_to_safe_name "$high_tau")
        low_tau_safe=$(convert_to_safe_name "$low_tau")
        align_safe=$(convert_to_safe_name "$align_weight")
        
        model_name="navi_${dataset,,}_tau${high_tau_safe}_${low_tau_safe}_align${align_safe}_ethresh10_90"
        echo "- ${model_name}: ./models/${model_name}"
    done
done

echo ""
echo "Completed training all models."
echo ""
echo "Log files are in logs/ directory with pattern: navi_<dataset>_<variant>_seed<seed>.log"
echo ""
echo "To monitor all logs:"
echo "tail -f logs/navi_*.log"
echo ""
echo "To check GPU usage:"
echo "nvidia-smi"