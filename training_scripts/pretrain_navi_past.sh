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
DEFAULT_HIGH_ENTROPY_TAU=0.07

# Seeds for variance analysis
SEEDS=(0 1 2)

# Hyperparameter variants (excluding default)
ALIGNMENT_WEIGHT_VARIANTS=(0.25 1.0 2.0 4.0)
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
    "percentile 10 90"
    "percentile 40 60"
    "percentile 50 50"
)

# Temperature variants for sensitivity analysis
# Format: high_entropy_tau low_entropy_tau
TEMPERATURE_VARIANTS=(
    "0.07 0.07"
    "0.07 0.10"
    "0.07 0.13"
    "0.10 0.10"
    "0.10 0.13"
    "0.13 0.13"
)

# Ablation variants
ABLATION_VARIANTS=("woSSI" "woMSM" "woESA")

# Gradient accumulation steps variants
GRADIENT_ACCUMULATION_VARIANTS=(1 2 4)

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

# Global arrays to track jobs per GPU (indexed by GPU ID)
declare -a GPU_0_PIDS=()
declare -a GPU_1_PIDS=()
MAX_JOBS_PER_GPU=2

# Global job queues for each GPU (stores job commands as strings)
declare -a GPU_0_QUEUE=()
declare -a GPU_1_QUEUE=()
GPU_0_QUEUE_INDEX=0
GPU_1_QUEUE_INDEX=0

# Function to get running job count for a GPU
get_gpu_job_count() {
    local gpu_idx=$1
    local count=0
    if [ "$gpu_idx" -eq 0 ]; then
        for pid in "${GPU_0_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                count=$((count + 1))
            fi
        done
    else
        for pid in "${GPU_1_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                count=$((count + 1))
            fi
        done
    fi
    echo "$count"
}

# Function to clean up finished jobs from GPU tracking
cleanup_gpu_jobs() {
    local gpu_idx=$1
    if [ "$gpu_idx" -eq 0 ]; then
        local new_pids=()
        for pid in "${GPU_0_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        GPU_0_PIDS=("${new_pids[@]}")
    else
        local new_pids=()
        for pid in "${GPU_1_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        GPU_1_PIDS=("${new_pids[@]}")
    fi
}

# Function to wait for GPU slot and launch job
# Usage: launch_job_on_gpu gpu_idx job_command
launch_job_on_gpu() {
    local gpu_idx=$1
    shift
    # Join all remaining arguments into a single command string
    local job_cmd="$*"
    
    # Wait until GPU has a free slot
    while [ $(get_gpu_job_count "$gpu_idx") -ge $MAX_JOBS_PER_GPU ]; do
        cleanup_gpu_jobs "$gpu_idx"
        sleep 1
    done
    
    # Launch the job
    eval "$job_cmd" &
    local pid=$!
    
    # Track the job
    if [ "$gpu_idx" -eq 0 ]; then
        GPU_0_PIDS+=("$pid")
    else
        GPU_1_PIDS+=("$pid")
    fi
    
    return 0
}

# Function to wait for all jobs on a specific GPU
wait_for_gpu_jobs() {
    local gpu_idx=$1
    if [ "$gpu_idx" -eq 0 ]; then
        for pid in "${GPU_0_PIDS[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
        GPU_0_PIDS=()
    else
        for pid in "${GPU_1_PIDS[@]}"; do
            wait "$pid" 2>/dev/null || true
        done
        GPU_1_PIDS=()
    fi
}

# Function to wait for all jobs on all GPUs
wait_for_all_gpu_jobs() {
    wait_for_gpu_jobs 0
    wait_for_gpu_jobs 1
}

# Function to add job to queue (non-blocking)
# Usage: add_job_to_queue gpu_idx job_command
add_job_to_queue() {
    local gpu_idx=$1
    shift
    local job_cmd="$*"
    
    if [ "$gpu_idx" -eq 0 ]; then
        GPU_0_QUEUE+=("$job_cmd")
    else
        GPU_1_QUEUE+=("$job_cmd")
    fi
}

# Function to process queues until all jobs are launched (non-blocking, processes what it can)
process_queues_once() {
    local jobs_launched=0
    
    # Process GPU 0 queue
    cleanup_gpu_jobs 0
    local gpu0_count=$(get_gpu_job_count 0)
    while [ $gpu0_count -lt $MAX_JOBS_PER_GPU ] && [ $GPU_0_QUEUE_INDEX -lt ${#GPU_0_QUEUE[@]} ]; do
        local job_cmd="${GPU_0_QUEUE[$GPU_0_QUEUE_INDEX]}"
        eval "$job_cmd" &
        local pid=$!
        GPU_0_PIDS+=("$pid")
        GPU_0_QUEUE_INDEX=$((GPU_0_QUEUE_INDEX + 1))
        jobs_launched=$((jobs_launched + 1))
        gpu0_count=$(get_gpu_job_count 0)
    done
    
    # Process GPU 1 queue
    cleanup_gpu_jobs 1
    local gpu1_count=$(get_gpu_job_count 1)
    while [ $gpu1_count -lt $MAX_JOBS_PER_GPU ] && [ $GPU_1_QUEUE_INDEX -lt ${#GPU_1_QUEUE[@]} ]; do
        local job_cmd="${GPU_1_QUEUE[$GPU_1_QUEUE_INDEX]}"
        eval "$job_cmd" &
        local pid=$!
        GPU_1_PIDS+=("$pid")
        GPU_1_QUEUE_INDEX=$((GPU_1_QUEUE_INDEX + 1))
        jobs_launched=$((jobs_launched + 1))
        gpu1_count=$(get_gpu_job_count 1)
    done
    
    echo $jobs_launched
}

# Function to wait for all queued jobs to complete
wait_for_queues_to_complete() {
    # Keep processing queues until all jobs are launched and completed
    while true; do
        # Try to launch more jobs from queues
        local jobs_launched=$(process_queues_once)
        
        # Check if all queues are processed and all jobs are done
        local total_queued=$(( ${#GPU_0_QUEUE[@]} + ${#GPU_1_QUEUE[@]} ))
        local total_processed=$((GPU_0_QUEUE_INDEX + GPU_1_QUEUE_INDEX))
        local total_running=$(( $(get_gpu_job_count 0) + $(get_gpu_job_count 1) ))
        
        # If all jobs are processed and none running, we're done
        if [ $total_processed -ge $total_queued ] && [ $total_running -eq 0 ]; then
            break
        fi
        
        # Sleep briefly before next check
        sleep 1
    done
    
    # Final cleanup
    wait_for_all_gpu_jobs
    # Reset queue indices for potential reuse
    GPU_0_QUEUE_INDEX=0
    GPU_1_QUEUE_INDEX=0
    GPU_0_QUEUE=()
    GPU_1_QUEUE=()
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
echo "Starting NAVI Model Training"
echo "Date: $(date)"
echo "GPU Assignment: Movie -> GPU 0, Product -> GPU 1"
echo "Datasets: ${DATASET_LIST[*]}"
echo "Running jobs in parallel (one per GPU)"
echo "=========================================="

# Get start time
START_TIME=$(date +%s)

echo "Using dynamic queue-based job scheduler..."
echo "Jobs will be interleaved across all phases for optimal GPU utilization"

# ===============================
# Default Configuration with 3 Epochs
# ===============================
echo "Training Default NAVI Models (3 epochs) on both domains..."
for i in "${!DATASET_LIST[@]}"; do
    dataset="${DATASET_LIST[$i]}"
    gpu_idx=$i  # Movie (index 0) -> GPU 0, Product (index 1) -> GPU 1
    
    model_name="navi_${dataset,,}_default_3epoch"
    job_cmd="train_navi_model \
        \"$dataset\" \
        \"$DEFAULT_ALIGNMENT_WEIGHT\" \
        \"$DEFAULT_HV_WEIGHT\" \
        \"$DEFAULT_VALUE_RATIO\" \
        \"$DEFAULT_LOW_ENTROPY_TAU\" \
        \"$DEFAULT_HIGH_ENTROPY_TAU\" \
        \"full\" \
        \"$model_name\" \
        \"default_3epoch\" \
        \"42\" \
        \"quartile\" \
        \"25\" \
        \"75\" \
        \"2\" \
        \"3\" \
        \"full\" \
        \"$gpu_idx\""
    
    add_job_to_queue "$gpu_idx" "$job_cmd"
done


# ===============================
# Seed Variance
# ===============================
echo "Training Default NAVI Models with seed variance..."
for i in "${!DATASET_LIST[@]}"; do
    dataset="${DATASET_LIST[$i]}"
    gpu_idx=$i  # Movie (index 0) -> GPU 0, Product (index 1) -> GPU 1
    
    for seed in "${SEEDS[@]}"; do
        job_cmd="train_navi_model \
            \"$dataset\" \
            \"$DEFAULT_ALIGNMENT_WEIGHT\" \
            \"$DEFAULT_HV_WEIGHT\" \
            \"$DEFAULT_VALUE_RATIO\" \
            \"$DEFAULT_LOW_ENTROPY_TAU\" \
            \"$DEFAULT_HIGH_ENTROPY_TAU\" \
            \"full\" \
            \"navi_${dataset,,}_seed${seed}\" \
            \"default\" \
            \"$seed\" \
            \"quartile\" \
            \"25\" \
            \"75\" \
            \"2\" \
            \"2\" \
            \"full\" \
            \"$gpu_idx\""
        
        add_job_to_queue "$gpu_idx" "$job_cmd"
    done
done

# ===============================
# Gradient Accumulation Steps Variants
# ===============================
# echo "Training Gradient Accumulation Steps Variants..."
# for i in "${!DATASET_LIST[@]}"; do
#     dataset="${DATASET_LIST[$i]}"
#     gpu_idx=$i  # Movie (index 0) -> GPU 0, Product (index 1) -> GPU 1
    
#     for gradient_accum in "${GRADIENT_ACCUMULATION_VARIANTS[@]}"; do
#         ga_suffix="ga${gradient_accum}"
#         model_name="navi_${dataset,,}_${ga_suffix}"
        
#         job_cmd="train_navi_model \
#             \"$dataset\" \
#             \"$DEFAULT_ALIGNMENT_WEIGHT\" \
#             \"$DEFAULT_HV_WEIGHT\" \
#             \"$DEFAULT_VALUE_RATIO\" \
#             \"$DEFAULT_LOW_ENTROPY_TAU\" \
#             \"$DEFAULT_HIGH_ENTROPY_TAU\" \
#             \"full\" \
#             \"$model_name\" \
#             \"$ga_suffix\" \
#             \"42\" \
#             \"quartile\" \
#             \"25\" \
#             \"75\" \
#             \"$gradient_accum\" \
#             \"2\" \
#             \"full\" \
#             \"$gpu_idx\""
        
#         launch_job_on_gpu "$gpu_idx" "$job_cmd"
#     done
# done
# # Wait for all jobs to complete
# wait_for_all_gpu_jobs

# ===============================
# Header Encoder Training Variations
# ===============================
# echo "Training Header Encoder Variants..."
# for i in "${!DATASET_LIST[@]}"; do
#     dataset="${DATASET_LIST[$i]}"
#     gpu_idx=$i  # Movie (index 0) -> GPU 0, Product (index 1) -> GPU 1
    
#     # Frozen header encoder
#     job1="train_navi_model \
#         \"$dataset\" \
#         \"$DEFAULT_ALIGNMENT_WEIGHT\" \
#         \"$DEFAULT_HV_WEIGHT\" \
#         \"$DEFAULT_VALUE_RATIO\" \
#         \"$DEFAULT_LOW_ENTROPY_TAU\" \
#         \"$DEFAULT_HIGH_ENTROPY_TAU\" \
#         \"full\" \
#         \"navi_${dataset,,}_hefrozen\" \
#         \"hefrozen\" \
#         \"42\" \
#         \"quartile\" \
#         \"25\" \
#         \"75\" \
#         \"2\" \
#         \"2\" \
#         \"frozen\" \
#         \"$gpu_idx\""
#     launch_job_on_gpu "$gpu_idx" "$job1"
    
#     # Partial header encoder (only layer_2 trainable)
#     job2="train_navi_model \
#         \"$dataset\" \
#         \"$DEFAULT_ALIGNMENT_WEIGHT\" \
#         \"$DEFAULT_HV_WEIGHT\" \
#         \"$DEFAULT_VALUE_RATIO\" \
#         \"$DEFAULT_LOW_ENTROPY_TAU\" \
#         \"$DEFAULT_HIGH_ENTROPY_TAU\" \
#         \"full\" \
#         \"navi_${dataset,,}_hepartial\" \
#         \"hepartial\" \
#         \"42\" \
#         \"quartile\" \
#         \"25\" \
#         \"75\" \
#         \"2\" \
#         \"2\" \
#         \"partial\" \
#         \"$gpu_idx\""
#     launch_job_on_gpu "$gpu_idx" "$job2"
# done
# # Wait for all jobs to complete
# wait_for_all_gpu_jobs

# ===============================
# Hyperparameter Variants
# ===============================
echo "Training Hyperparameter Variants..."

# # Alignment Weight Variants (COMMENTED OUT)
# echo "Training Alignment Weight Variants..."
# for dataset in "${DATASET_LIST[@]}"; do
#     for alignment_weight in "${ALIGNMENT_WEIGHT_VARIANTS[@]}"; do
#         alignment_safe=$(convert_to_safe_name "$alignment_weight")
#         train_navi_model \
#             "$dataset" \
#             "$alignment_weight" \
#             "$DEFAULT_HV_WEIGHT" \
#             "$DEFAULT_VALUE_RATIO" \
#             "$DEFAULT_LOW_ENTROPY_TAU" \
#             "$DEFAULT_HIGH_ENTROPY_TAU" \
#             "full" \
#             "navi_${dataset,,}_align${alignment_safe}" \
#             "align${alignment_safe}" \
#             "42"
#     done
# done

# # HV Weight & Value Ratio Variants (COMMENTED OUT)
# echo "Training HV Weight & Value Ratio Variants..."
# for dataset in "${DATASET_LIST[@]}"; do
#     for hv_value_pair in "${HV_WEIGHT_VALUE_RATIO_VARIANTS[@]}"; do
#         hv_weight=$(echo $hv_value_pair | cut -d' ' -f1)
#         value_ratio=$(echo $hv_value_pair | cut -d' ' -f2)
#         hv_safe=$(convert_to_safe_name "$hv_weight")
#         vr_safe=$(convert_to_safe_name "$value_ratio")
#         train_navi_model \
#             "$dataset" \
#             "$DEFAULT_ALIGNMENT_WEIGHT" \
#             "$hv_weight" \
#             "$value_ratio" \
#             "$DEFAULT_LOW_ENTROPY_TAU" \
#             "$DEFAULT_HIGH_ENTROPY_TAU" \
#             "full" \
#             "navi_${dataset,,}_hv${hv_safe}_vr${vr_safe}" \
#             "hv${hv_safe}_vr${vr_safe}" \
#             "42"
#     done
# done

# Entropy Threshold Sensitivity Analysis
# echo "Training Entropy Threshold Sensitivity Analysis Variants..."
# for i in "${!DATASET_LIST[@]}"; do
#     dataset="${DATASET_LIST[$i]}"
#     gpu_idx=$i  # Movie (index 0) -> GPU 0, Product (index 1) -> GPU 1
    
#     for threshold_variant in "${ENTROPY_THRESHOLD_VARIANTS[@]}"; do
#         method=$(echo $threshold_variant | cut -d' ' -f1)
#         low_pct=$(echo $threshold_variant | cut -d' ' -f2)
#         high_pct=$(echo $threshold_variant | cut -d' ' -f3)
        
#         threshold_suffix="ethresh${low_pct}_${high_pct}"
#         model_name="navi_${dataset,,}_${threshold_suffix}"
        
#         job_cmd="train_navi_model \
#             \"$dataset\" \
#             \"$DEFAULT_ALIGNMENT_WEIGHT\" \
#             \"$DEFAULT_HV_WEIGHT\" \
#             \"$DEFAULT_VALUE_RATIO\" \
#             \"$DEFAULT_LOW_ENTROPY_TAU\" \
#             \"$DEFAULT_HIGH_ENTROPY_TAU\" \
#             \"full\" \
#             \"$model_name\" \
#             \"$threshold_suffix\" \
#             \"42\" \
#             \"$method\" \
#             \"$low_pct\" \
#             \"$high_pct\" \
#             \"2\" \
#             \"2\" \
#             \"full\" \
#             \"$gpu_idx\""
        
#         launch_job_on_gpu "$gpu_idx" "$job_cmd"
#     done
# done
# # Wait for all jobs to complete
# wait_for_all_gpu_jobs

# Temperature Sensitivity Analysis
echo "Training Temperature Sensitivity Analysis Variants..."
for i in "${!DATASET_LIST[@]}"; do
    dataset="${DATASET_LIST[$i]}"
    gpu_idx=$i  # Movie (index 0) -> GPU 0, Product (index 1) -> GPU 1
    
    for temp_variant in "${TEMPERATURE_VARIANTS[@]}"; do
        high_tau=$(echo $temp_variant | cut -d' ' -f1)
        low_tau=$(echo $temp_variant | cut -d' ' -f2)
        
        high_tau_safe=$(convert_to_safe_name "$high_tau")
        low_tau_safe=$(convert_to_safe_name "$low_tau")
        temp_suffix="tau${high_tau_safe}_${low_tau_safe}"
        model_name="navi_${dataset,,}_${temp_suffix}"
        
        job_cmd="train_navi_model \
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
            \"quartile\" \
            \"25\" \
            \"75\" \
            \"2\" \
            \"2\" \
            \"full\" \
            \"$gpu_idx\""
        
        add_job_to_queue "$gpu_idx" "$job_cmd"
    done
done


# ===============================
# Ablation Variants (COMMENTED OUT)
# ===============================
# echo "Phase 5: Training Ablation Variants..."
# for dataset in "${DATASET_LIST[@]}"; do
#     for ablation in "${ABLATION_VARIANTS[@]}"; do
#         train_navi_model \
#             "$dataset" \
#             "$DEFAULT_ALIGNMENT_WEIGHT" \
#             "$DEFAULT_HV_WEIGHT" \
#             "$DEFAULT_VALUE_RATIO" \
#             "$DEFAULT_LOW_ENTROPY_TAU" \
#             "$DEFAULT_HIGH_ENTROPY_TAU" \
#             "$ablation" \
#             "navi_${dataset,,}_${ablation}" \
#             "$ablation" \
#             "42"
#     done
# done



# Wait for all queued jobs to complete
echo "All jobs added to queue. Processing 4 at a time (2 per GPU)..."
wait_for_queues_to_complete

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
echo "Default Configuration (3 epochs):"
for dataset in "${DATASET_LIST[@]}"; do
    echo "- navi_${dataset,,}_default_3epoch: ./models/navi_${dataset,,}_default_3epoch"
done

echo ""
echo "Gradient Accumulation Variants:"
for dataset in "${DATASET_LIST[@]}"; do
    for gradient_accum in "${GRADIENT_ACCUMULATION_VARIANTS[@]}"; do
        echo "- navi_${dataset,,}_ga${gradient_accum}: ./models/navi_${dataset,,}_ga${gradient_accum}"
    done
done

echo ""
echo "Header Encoder Training Variations:"
for dataset in "${DATASET_LIST[@]}"; do
    echo "- navi_${dataset,,}_hefrozen: ./models/navi_${dataset,,}_hefrozen (header encoder frozen)"
    echo "- navi_${dataset,,}_hepartial: ./models/navi_${dataset,,}_hepartial (header encoder partially fine-tuned)"
done

echo ""
echo "Seed Variance:"
for dataset in "${DATASET_LIST[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "- navi_${dataset,,}_seed${seed}: ./models/navi_${dataset,,}_seed${seed}"
    done
done

echo ""
echo "Hyperparameter Variants:"
echo "Entropy Threshold Variants:"
for dataset in "${DATASET_LIST[@]}"; do
    for threshold_variant in "${ENTROPY_THRESHOLD_VARIANTS[@]}"; do
        low_pct=$(echo $threshold_variant | cut -d' ' -f2)
        high_pct=$(echo $threshold_variant | cut -d' ' -f3)
        echo "- navi_${dataset,,}_ethresh${low_pct}_${high_pct}: ./models/navi_${dataset,,}_ethresh${low_pct}_${high_pct}"
    done
done

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
echo "Completed training all models."
echo ""
echo "Log files are in logs/ directory with pattern: navi_<dataset>_<variant>.log"
echo ""
echo "To monitor all logs:"
echo "tail -f logs/navi_*.log"
echo ""
echo "To check GPU usage:"
echo "nvidia-smi"




