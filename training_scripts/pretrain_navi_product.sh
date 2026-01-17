#!/bin/bash

# Usage Example:
# bash pretrain_navi_product.sh

# NAVI Model Training Script (Single GPU)
# Trains NAVI model with hyperparameter variants and ablation variants
# Runs 2 jobs in parallel, waits for both to finish, then runs next 2

set -e  # Exit on any error

# ===============================
# Configuration
# ===============================
DATASET_LIST=("Product")
GPU_IDX=0

# Default parameters
DEFAULT_ALIGNMENT_WEIGHT=0.05
DEFAULT_HV_WEIGHT=0.8
DEFAULT_VALUE_RATIO=0.5
DEFAULT_LOW_ENTROPY_TAU=0.14
DEFAULT_HIGH_ENTROPY_TAU=0.02
DEFAULT_ENTROPY_THRESHOLD_METHOD="percentile"
DEFAULT_LOW_THRESHOLD_PERCENTILE=10
DEFAULT_HIGH_THRESHOLD_PERCENTILE=90
DEFAULT_GRADIENT_ACCUMULATION_STEPS=1

# Hyperparameter variants (excluding default)
ALIGNMENT_WEIGHT_VARIANTS=(0.01 0.25 1.25)
HV_WEIGHT_VALUE_RATIO_VARIANTS=(
    "0.8 0.25"
    "0.8 0.75"
    "0.4 0.25"
    "0.4 0.5"
    "0.4 0.75"
)

# Entropy threshold variants for sensitivity analysis
# Format: method low_percentile high_percentile
ENTROPY_THRESHOLD_VARIANTS=(
    "percentile 25 75"
    "percentile 40 60"
    "percentile 50 50"
)

# Temperature variants for sensitivity analysis
# Format: high_entropy_tau low_entropy_tau
TEMPERATURE_VARIANTS=(
    "0.02 0.02"
    "0.14 0.14"
)

# Negative set size (Gradient Accumulation Steps) variants for sensitivity analysis
# Format: gradient_accumulation_steps
GRADIENT_ACCUMULATION_VARIANTS=(2 4)

# # Ablation variants
# ABLATION_VARIANTS=("woSSI" "woMSM" "woESA")

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

# Run 2 jobs in parallel and wait for both to complete
run_batch_of_2() {
    local job1_cmd="$1"
    local job2_cmd="$2"
    
    echo "Starting batch of 2 jobs..."
    
    # Launch both jobs in background
    eval "$job1_cmd" &
    local pid1=$!
    echo "  Launched job 1 (PID: $pid1)"
    
    eval "$job2_cmd" &
    local pid2=$!
    echo "  Launched job 2 (PID: $pid2)"
    
    # Wait for both to complete
    wait "$pid1" 2>/dev/null || true
    wait "$pid2" 2>/dev/null || true
    
    echo "  Batch completed. Both jobs finished."
    echo "---"
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
    local entropy_threshold_method=${11:-"$DEFAULT_ENTROPY_THRESHOLD_METHOD"}
    local low_threshold_percentile=${12:-$DEFAULT_LOW_THRESHOLD_PERCENTILE}
    local high_threshold_percentile=${13:-$DEFAULT_HIGH_THRESHOLD_PERCENTILE}
    local gradient_accumulation_steps=${14:-$DEFAULT_GRADIENT_ACCUMULATION_STEPS}
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
echo "Starting NAVI Model Training (Single GPU)"
echo "Date: $(date)"
echo "GPU: ${GPU_IDX}"
echo "Datasets: ${DATASET_LIST[*]}"
echo "Running 2 jobs at a time, waiting for both to finish"
echo "=========================================="

# Get start time
START_TIME=$(date +%s)

# Collect all job commands into an array
declare -a ALL_JOBS=()

# ===============================
# Alignment Weight Variants
# ===============================
echo "Collecting Alignment Weight Variants..."
for dataset in "${DATASET_LIST[@]}"; do
    for alignment_weight in "${ALIGNMENT_WEIGHT_VARIANTS[@]}"; do
        alignment_safe=$(convert_to_safe_name "$alignment_weight")
        model_name="navi_${dataset,,}_align${alignment_safe}"
        log_suffix="align${alignment_safe}"
        
        ALL_JOBS+=("train_navi_model \
            \"$dataset\" \
            \"$alignment_weight\" \
            \"$DEFAULT_HV_WEIGHT\" \
            \"$DEFAULT_VALUE_RATIO\" \
            \"$DEFAULT_LOW_ENTROPY_TAU\" \
            \"$DEFAULT_HIGH_ENTROPY_TAU\" \
            \"full\" \
            \"$model_name\" \
            \"$log_suffix\" \
            \"42\" \
            \"$DEFAULT_ENTROPY_THRESHOLD_METHOD\" \
            \"$DEFAULT_LOW_THRESHOLD_PERCENTILE\" \
            \"$DEFAULT_HIGH_THRESHOLD_PERCENTILE\" \
            \"$DEFAULT_GRADIENT_ACCUMULATION_STEPS\" \
            \"2\" \
            \"full\" \
            \"$GPU_IDX\"")
    done
done

# ===============================
# HV Weight & Value Ratio Variants
# ===============================
echo "Collecting HV Weight & Value Ratio Variants..."
for dataset in "${DATASET_LIST[@]}"; do
    for hv_value_pair in "${HV_WEIGHT_VALUE_RATIO_VARIANTS[@]}"; do
        hv_weight=$(echo $hv_value_pair | cut -d' ' -f1)
        value_ratio=$(echo $hv_value_pair | cut -d' ' -f2)
        hv_safe=$(convert_to_safe_name "$hv_weight")
        vr_safe=$(convert_to_safe_name "$value_ratio")
        model_name="navi_${dataset,,}_hv${hv_safe}_vr${vr_safe}"
        log_suffix="hv${hv_safe}_vr${vr_safe}"
        
        ALL_JOBS+=("train_navi_model \
            \"$dataset\" \
            \"$DEFAULT_ALIGNMENT_WEIGHT\" \
            \"$hv_weight\" \
            \"$value_ratio\" \
            \"$DEFAULT_LOW_ENTROPY_TAU\" \
            \"$DEFAULT_HIGH_ENTROPY_TAU\" \
            \"full\" \
            \"$model_name\" \
            \"$log_suffix\" \
            \"42\" \
            \"$DEFAULT_ENTROPY_THRESHOLD_METHOD\" \
            \"$DEFAULT_LOW_THRESHOLD_PERCENTILE\" \
            \"$DEFAULT_HIGH_THRESHOLD_PERCENTILE\" \
            \"$DEFAULT_GRADIENT_ACCUMULATION_STEPS\" \
            \"2\" \
            \"full\" \
            \"$GPU_IDX\"")
    done
done

# ===============================
# Entropy Threshold Variants
# ===============================
if [ -n "${ENTROPY_THRESHOLD_VARIANTS+x}" ] && [ ${#ENTROPY_THRESHOLD_VARIANTS[@]} -gt 0 ]; then
    echo "Collecting Entropy Threshold Variants..."
    for dataset in "${DATASET_LIST[@]}"; do
        for threshold_variant in "${ENTROPY_THRESHOLD_VARIANTS[@]}"; do
            method=$(echo $threshold_variant | cut -d' ' -f1)
            low_pct=$(echo $threshold_variant | cut -d' ' -f2)
            high_pct=$(echo $threshold_variant | cut -d' ' -f3)
            
            threshold_suffix="ethresh${low_pct}_${high_pct}"
            model_name="navi_${dataset,,}_${threshold_suffix}"
            
            ALL_JOBS+=("train_navi_model \
                \"$dataset\" \
                \"$DEFAULT_ALIGNMENT_WEIGHT\" \
                \"$DEFAULT_HV_WEIGHT\" \
                \"$DEFAULT_VALUE_RATIO\" \
                \"$DEFAULT_LOW_ENTROPY_TAU\" \
                \"$DEFAULT_HIGH_ENTROPY_TAU\" \
                \"full\" \
                \"$model_name\" \
                \"$threshold_suffix\" \
                \"42\" \
                \"$method\" \
                \"$low_pct\" \
                \"$high_pct\" \
                \"$DEFAULT_GRADIENT_ACCUMULATION_STEPS\" \
                \"2\" \
                \"full\" \
                \"$GPU_IDX\"")
        done
    done
fi

# ===============================
# Temperature Variants
# ===============================
echo "Collecting Temperature Variants..."
for dataset in "${DATASET_LIST[@]}"; do
    for temp_variant in "${TEMPERATURE_VARIANTS[@]}"; do
        high_tau=$(echo $temp_variant | cut -d' ' -f1)
        low_tau=$(echo $temp_variant | cut -d' ' -f2)
        
        high_tau_safe=$(convert_to_safe_name "$high_tau")
        low_tau_safe=$(convert_to_safe_name "$low_tau")
        temp_suffix="tau${high_tau_safe}_${low_tau_safe}"
        model_name="navi_${dataset,,}_${temp_suffix}"
        
        ALL_JOBS+=("train_navi_model \
            \"$dataset\" \
            \"$DEFAULT_ALIGNMENT_WEIGHT\" \
            \"$DEFAULT_HV_WEIGHT\" \
            \"$DEFAULT_VALUE_RATIO\" \
            \"$low_tau\" \
            \"$high_tau\" \
            \"full\" \
            \"$model_name\" \
            \"$temp_suffix\" \
            \"42\" \
            \"$DEFAULT_ENTROPY_THRESHOLD_METHOD\" \
            \"$DEFAULT_LOW_THRESHOLD_PERCENTILE\" \
            \"$DEFAULT_HIGH_THRESHOLD_PERCENTILE\" \
            \"$DEFAULT_GRADIENT_ACCUMULATION_STEPS\" \
            \"2\" \
            \"full\" \
            \"$GPU_IDX\"")
    done
done

# ===============================
# Gradient Accumulation Steps Variants
# ===============================
echo "Collecting Gradient Accumulation Steps Variants..."
for dataset in "${DATASET_LIST[@]}"; do
    for gradient_accumulation_steps in "${GRADIENT_ACCUMULATION_VARIANTS[@]}"; do
        model_name="navi_${dataset,,}_ga${gradient_accumulation_steps}"
        log_suffix="ga${gradient_accumulation_steps}"
        
        ALL_JOBS+=("train_navi_model \
            \"$dataset\" \
            \"$DEFAULT_ALIGNMENT_WEIGHT\" \
            \"$DEFAULT_HV_WEIGHT\" \
            \"$DEFAULT_VALUE_RATIO\" \
            \"$DEFAULT_LOW_ENTROPY_TAU\" \
            \"$DEFAULT_HIGH_ENTROPY_TAU\" \
            \"full\" \
            \"$model_name\" \
            \"$log_suffix\" \
            \"42\" \
            \"$DEFAULT_ENTROPY_THRESHOLD_METHOD\" \
            \"$DEFAULT_LOW_THRESHOLD_PERCENTILE\" \
            \"$DEFAULT_HIGH_THRESHOLD_PERCENTILE\" \
            \"$gradient_accumulation_steps\" \
            \"2\" \
            \"full\" \
            \"$GPU_IDX\"")
    done
done

# ===============================
# Header Encoder Training Variants
# ===============================
echo "Collecting Header Encoder Training Variants..."
for dataset in "${DATASET_LIST[@]}"; do
    model_name="navi_${dataset,,}_hefrozen"
    log_suffix="hefrozen"
    
    ALL_JOBS+=("train_navi_model \
        \"$dataset\" \
        \"$DEFAULT_ALIGNMENT_WEIGHT\" \
        \"$DEFAULT_HV_WEIGHT\" \
        \"$DEFAULT_VALUE_RATIO\" \
        \"$DEFAULT_LOW_ENTROPY_TAU\" \
        \"$DEFAULT_HIGH_ENTROPY_TAU\" \
        \"full\" \
        \"$model_name\" \
        \"$log_suffix\" \
        \"42\" \
        \"$DEFAULT_ENTROPY_THRESHOLD_METHOD\" \
        \"$DEFAULT_LOW_THRESHOLD_PERCENTILE\" \
        \"$DEFAULT_HIGH_THRESHOLD_PERCENTILE\" \
        \"$DEFAULT_GRADIENT_ACCUMULATION_STEPS\" \
        \"2\" \
        \"frozen\" \
        \"$GPU_IDX\"")

    model_name="navi_${dataset,,}_hepartial"
    log_suffix="hepartial"
    
    ALL_JOBS+=("train_navi_model \
        \"$dataset\" \
        \"$DEFAULT_ALIGNMENT_WEIGHT\" \
        \"$DEFAULT_HV_WEIGHT\" \
        \"$DEFAULT_VALUE_RATIO\" \
        \"$DEFAULT_LOW_ENTROPY_TAU\" \
        \"$DEFAULT_HIGH_ENTROPY_TAU\" \
        \"full\" \
        \"$model_name\" \
        \"$log_suffix\" \
        \"42\" \
        \"$DEFAULT_ENTROPY_THRESHOLD_METHOD\" \
        \"$DEFAULT_LOW_THRESHOLD_PERCENTILE\" \
        \"$DEFAULT_HIGH_THRESHOLD_PERCENTILE\" \
        \"$DEFAULT_GRADIENT_ACCUMULATION_STEPS\" \
        \"2\" \
        \"partial\" \
        \"$GPU_IDX\"")
done

# # ===============================
# # Ablation Variants
# # ===============================
# echo "Collecting Ablation Variants..."
# for dataset in "${DATASET_LIST[@]}"; do
#     for ablation in "${ABLATION_VARIANTS[@]}"; do
#         model_name="navi_${dataset,,}_${ablation}"
        
#         ALL_JOBS+=("train_navi_model \
#             \"$dataset\" \
#             \"$DEFAULT_ALIGNMENT_WEIGHT\" \
#             \"$DEFAULT_HV_WEIGHT\" \
#             \"$DEFAULT_VALUE_RATIO\" \
#             \"$DEFAULT_LOW_ENTROPY_TAU\" \
#             \"$DEFAULT_HIGH_ENTROPY_TAU\" \
#             \"$ablation\" \
#             \"$model_name\" \
#             \"$ablation\" \
#             \"42\" \
#             \"$DEFAULT_ENTROPY_THRESHOLD_METHOD\" \
#             \"$DEFAULT_LOW_THRESHOLD_PERCENTILE\" \
#             \"$DEFAULT_HIGH_THRESHOLD_PERCENTILE\" \
#             \"$DEFAULT_GRADIENT_ACCUMULATION_STEPS\" \
#             \"2\" \
#             \"full\" \
#             \"$GPU_IDX\"")
#     done
# done

# ===============================
# Epoch 3 Variant
# ===============================
echo "Collecting Epoch 3 Training..."
for dataset in "${DATASET_LIST[@]}"; do
    for epoch in "${EPOCH_VARIANTS[@]}"; do
        model_name="navi_${dataset,,}_epoch${epoch}"
        log_suffix="epoch${epoch}"
        
        ALL_JOBS+=("train_navi_model \
            \"$dataset\" \
            \"$DEFAULT_ALIGNMENT_WEIGHT\" \
            \"$DEFAULT_HV_WEIGHT\" \
            \"$DEFAULT_VALUE_RATIO\" \
            \"$DEFAULT_LOW_ENTROPY_TAU\" \
            \"$DEFAULT_HIGH_ENTROPY_TAU\" \
            \"full\" \
            \"$model_name\" \
            \"$log_suffix\" \
            \"42\" \
            \"$DEFAULT_ENTROPY_THRESHOLD_METHOD\" \
            \"$DEFAULT_LOW_THRESHOLD_PERCENTILE\" \
            \"$DEFAULT_HIGH_THRESHOLD_PERCENTILE\" \
            \"$DEFAULT_GRADIENT_ACCUMULATION_STEPS\" \
            \"3\" \
            \"full\" \
            \"$GPU_IDX\"")
    done
done

# ===============================
# Execute jobs in batches of 2
# ===============================
echo ""
echo "Total jobs collected: ${#ALL_JOBS[@]}"
echo "Executing in batches of 2..."
echo ""

total_jobs=${#ALL_JOBS[@]}
batch_num=1

for ((i=0; i<total_jobs; i+=2)); do
    job1="${ALL_JOBS[$i]}"
    
    if [ $((i+1)) -lt $total_jobs ]; then
        # We have 2 jobs
        job2="${ALL_JOBS[$((i+1))]}"
        echo "Batch $batch_num: Running jobs $((i+1)) and $((i+2)) of $total_jobs"
        run_batch_of_2 "$job1" "$job2"
    else
        # Only 1 job left (odd number)
        echo "Batch $batch_num: Running final job $((i+1)) of $total_jobs"
        eval "$job1"
        echo "  Final job completed."
        echo "---"
    fi
    
    batch_num=$((batch_num + 1))
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

echo "Alignment Weight Variants:"
for dataset in "${DATASET_LIST[@]}"; do
    for alignment_weight in "${ALIGNMENT_WEIGHT_VARIANTS[@]}"; do
        alignment_safe=$(convert_to_safe_name "$alignment_weight")
        echo "- navi_${dataset,,}_align${alignment_safe}: ./models/navi_${dataset,,}_align${alignment_safe}"
    done
done

echo ""
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

if [ -n "${ENTROPY_THRESHOLD_VARIANTS+x}" ] && [ ${#ENTROPY_THRESHOLD_VARIANTS[@]} -gt 0 ]; then
    echo ""
    echo "Entropy Threshold Variants:"
    for dataset in "${DATASET_LIST[@]}"; do
        for threshold_variant in "${ENTROPY_THRESHOLD_VARIANTS[@]}"; do
            low_pct=$(echo $threshold_variant | cut -d' ' -f2)
            high_pct=$(echo $threshold_variant | cut -d' ' -f3)
            echo "- navi_${dataset,,}_ethresh${low_pct}_${high_pct}: ./models/navi_${dataset,,}_ethresh${low_pct}_${high_pct}"
        done
    done
fi

echo ""
echo "Temperature Variants:"
for dataset in "${DATASET_LIST[@]}"; do
    for temp_variant in "${TEMPERATURE_VARIANTS[@]}"; do
        high_tau=$(echo $temp_variant | cut -d' ' -f1)
        low_tau=$(echo $temp_variant | cut -d' ' -f2)
        high_tau_safe=$(convert_to_safe_name "$high_tau")
        low_tau_safe=$(convert_to_safe_name "$low_tau")
        echo "- navi_${dataset,,}_tau${high_tau_safe}_${low_tau_safe}: ./models/navi_${dataset,,}_tau${high_tau_safe}_${low_tau_safe}"
    done
done

echo ""
echo "Negative Set Size Steps Variants:"
for dataset in "${DATASET_LIST[@]}"; do
    for gradient_accumulation_steps in "${GRADIENT_ACCUMULATION_VARIANTS[@]}"; do
        echo "- navi_${dataset,,}_ga${gradient_accumulation_steps}: ./models/navi_${dataset,,}_ga${gradient_accumulation_steps}"
    done
done

# echo ""
# echo "Ablation Variants:"
# for dataset in "${DATASET_LIST[@]}"; do
#     for ablation in "${ABLATION_VARIANTS[@]}"; do
#         echo "- navi_${dataset,,}_${ablation}: ./models/navi_${dataset,,}_${ablation}"
#     done
# done

echo ""
echo "Epoch 3 Training:"
for dataset in "${DATASET_LIST[@]}"; do
    echo "- navi_${dataset,,}_epoch3: ./models/navi_${dataset,,}_epoch3"
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