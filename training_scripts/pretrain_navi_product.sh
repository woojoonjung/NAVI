#!/bin/bash

# Usage Example:
# bash training_scripts/pretrain_navi_product.sh

# NAVI Model Training Script (Single GPU, Product)
# Refined version of pretrain_navi_product_past.sh
# - Updated default hyperparameters (see below)
# - Negative set size ablation via batch size (not grad accumulation)
# - Runs 3 jobs in parallel, waits, then runs next 3

set -e  # Exit on any error

# ===============================
# Configuration
# ===============================
DATASET_LIST=("Product")
GPU_IDX=1

# Default parameters
DEFAULT_ALIGNMENT_WEIGHT=0.0125
DEFAULT_HV_WEIGHT=0.8
DEFAULT_VALUE_RATIO=0.5
DEFAULT_LOW_ENTROPY_TAU=0.1
DEFAULT_HIGH_ENTROPY_TAU=0.02
DEFAULT_ENTROPY_THRESHOLD_METHOD="percentile"
DEFAULT_LOW_THRESHOLD_PERCENTILE=5
DEFAULT_HIGH_THRESHOLD_PERCENTILE=95
DEFAULT_GRADIENT_ACCUMULATION_STEPS=1
DEFAULT_BATCH_SIZE=32

# Ablation variants
ABLATION_VARIANTS=("woSSI" "woMSM" "woESA")

# Hyperparameter variants (excluding default)
# Negative set size ablation via batch size
# BATCH_SIZE_VARIANTS=(16 32 64)

# Entropy threshold variants for sensitivity analysis
# Format: method low_percentile high_percentile
ENTROPY_THRESHOLD_VARIANTS=(
    "percentile 10 90"
    "percentile 25 75"
    "percentile 50 50"
)

# Temperature variants for sensitivity analysis
# Format: high_entropy_tau low_entropy_tau
TEMPERATURE_VARIANTS=(
    "0.04 0.05"
    "0.01 0.05"
    "0.02 0.025"
    "0.02 0.1"
)

ALIGNMENT_WEIGHT_VARIANTS=(0.001 0.1 1.0)
HV_WEIGHT_VALUE_RATIO_VARIANTS=(
    "0.8 0.25"
    "0.8 0.75"
    "0.4 0.5"
)

# Seed variants for variance analysis (default is 42)
SEED_VARIANTS=(0 1)

# ===============================
# Create necessary directories
# ===============================
mkdir -p logs
mkdir -p models

# ===============================
# Validation
# ===============================
validate_setup() {
    echo "Validating setup..."
    local errors=0

    # Check if datasets exist
    for dataset in "${DATASET_LIST[@]}"; do
        local train_path="./data/cleaned/${dataset}/train"
        local val_path="./data/cleaned/${dataset}/validation"

        if [ ! -d "$train_path" ]; then
            echo "  ✗ Training data not found: $train_path"
            errors=$((errors + 1))
        fi

        if [ ! -d "$val_path" ]; then
            echo "  ✗ Validation data not found: $val_path"
            errors=$((errors + 1))
        fi
    done

    # Check if Python training script exists
    if [ ! -f "training/train_navi.py" ]; then
        echo "  ✗ Training script not found: training/train_navi.py"
        errors=$((errors + 1))
    fi

    if [ $errors -eq 0 ]; then
        echo "  ✓ Setup validation passed"
    else
        echo "  ✗ Setup validation failed with $errors error(s)"
        return 1
    fi
}

# ===============================
# Helper Functions
# ===============================
convert_to_safe_name() {
    echo "$1" | sed 's/\./p/g'
}

# Run 3 jobs in parallel and wait for all to complete
run_batch_of_3() {
    local job1_cmd="$1"
    local job2_cmd="$2"
    local job3_cmd="$3"

    echo "Starting batch of 3 jobs..."

    eval "$job1_cmd" &
    local pid1=$!
    echo "  Launched job 1 (PID: $pid1)"

    eval "$job2_cmd" &
    local pid2=$!
    echo "  Launched job 2 (PID: $pid2)"

    eval "$job3_cmd" &
    local pid3=$!
    echo "  Launched job 3 (PID: $pid3)"

    wait "$pid1" 2>/dev/null
    local exit1=$?
    wait "$pid2" 2>/dev/null
    local exit2=$?
    wait "$pid3" 2>/dev/null
    local exit3=$?

    local total_exits=$((exit1 | exit2 | exit3))
    if [ $total_exits -eq 0 ]; then
        echo "  ✓ Batch completed successfully. All 3 jobs finished."
    else
        echo "  ⚠️ Batch completed with errors:"
        [ $exit1 -ne 0 ] && echo "    - Job 1 (PID: $pid1) failed with exit code $exit1"
        [ $exit2 -ne 0 ] && echo "    - Job 2 (PID: $pid2) failed with exit code $exit2"
        [ $exit3 -ne 0 ] && echo "    - Job 3 (PID: $pid3) failed with exit code $exit3"
    fi
    echo "---"

    return $total_exits
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
    local batch_size=${14:-$DEFAULT_BATCH_SIZE}
    local gradient_accumulation_steps=${15:-$DEFAULT_GRADIENT_ACCUMULATION_STEPS}
    local num_epochs=${16:-2}
    local header_encoder_mode=${17:-"full"}
    local gpu_idx=${18:-0}

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
    echo "Batch Size: ${batch_size}"
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
        --batch_size "${batch_size}" \
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
echo "Default Batch Size: ${DEFAULT_BATCH_SIZE}"
echo "Default Gradient Accumulation Steps: ${DEFAULT_GRADIENT_ACCUMULATION_STEPS}"
echo "Running 3 jobs at a time, waiting for all to finish"
echo "=========================================="

validate_setup || exit 1
echo ""

START_TIME=$(date +%s)
declare -a ALL_JOBS=()

# # ===============================
# # Default Model Training (Run First)
# # ===============================
# echo "=========================================="
# echo "Step 1: Training Default Model"
# echo "=========================================="
# for dataset in "${DATASET_LIST[@]}"; do
#     model_name="navi_${dataset,,}"
#     log_suffix="default"
    
#     echo "Training default model for ${dataset}..."
#     train_navi_model \
#         "$dataset" \
#         "$DEFAULT_ALIGNMENT_WEIGHT" \
#         "$DEFAULT_HV_WEIGHT" \
#         "$DEFAULT_VALUE_RATIO" \
#         "$DEFAULT_LOW_ENTROPY_TAU" \
#         "$DEFAULT_HIGH_ENTROPY_TAU" \
#         "full" \
#         "$model_name" \
#         "$log_suffix" \
#         "42" \
#         "$DEFAULT_ENTROPY_THRESHOLD_METHOD" \
#         "$DEFAULT_LOW_THRESHOLD_PERCENTILE" \
#         "$DEFAULT_HIGH_THRESHOLD_PERCENTILE" \
#         "$DEFAULT_BATCH_SIZE" \
#         "$DEFAULT_GRADIENT_ACCUMULATION_STEPS" \
#         "2" \
#         "full" \
#         "$GPU_IDX"
# done
# echo ""
# echo "Default model training completed. Starting hyperparameter variants..."
# echo ""

# ===============================
# Hyperparameter Variants
# ===============================

# ===============================
# Ablation Variants
# ===============================
echo "Collecting Ablation Variants..."
for dataset in "${DATASET_LIST[@]}"; do
    for ablation in "${ABLATION_VARIANTS[@]}"; do
        model_name="navi_${dataset,,}_${ablation}"
        ALL_JOBS+=("train_navi_model \
            \"$dataset\" \
            \"$DEFAULT_ALIGNMENT_WEIGHT\" \
            \"$DEFAULT_HV_WEIGHT\" \
            \"$DEFAULT_VALUE_RATIO\" \
            \"$DEFAULT_LOW_ENTROPY_TAU\" \
            \"$DEFAULT_HIGH_ENTROPY_TAU\" \
            \"$ablation\" \
            \"$model_name\" \
            \"$ablation\" \
            \"42\" \
            \"$DEFAULT_ENTROPY_THRESHOLD_METHOD\" \
            \"$DEFAULT_LOW_THRESHOLD_PERCENTILE\" \
            \"$DEFAULT_HIGH_THRESHOLD_PERCENTILE\" \
            \"$DEFAULT_BATCH_SIZE\" \
            \"$DEFAULT_GRADIENT_ACCUMULATION_STEPS\" \
            \"2\" \
            \"full\" \
            \"$GPU_IDX\"")
    done
done

# ===============================
# Negative Set Size Ablation (via batch size variants)
# ===============================
# echo "Collecting Batch Size Variants (Negative set size ablation)..."
# for dataset in "${DATASET_LIST[@]}"; do
#     for bs in "${BATCH_SIZE_VARIANTS[@]}"; do
#         model_name="navi_${dataset,,}_bs${bs}"
#         log_suffix="bs${bs}"

#         ALL_JOBS+=("train_navi_model \
#             \"$dataset\" \
#             \"$DEFAULT_ALIGNMENT_WEIGHT\" \
#             \"$DEFAULT_HV_WEIGHT\" \
#             \"$DEFAULT_VALUE_RATIO\" \
#             \"$DEFAULT_LOW_ENTROPY_TAU\" \
#             \"$DEFAULT_HIGH_ENTROPY_TAU\" \
#             \"full\" \
#             \"$model_name\" \
#             \"$log_suffix\" \
#             \"42\" \
#             \"$DEFAULT_ENTROPY_THRESHOLD_METHOD\" \
#             \"$DEFAULT_LOW_THRESHOLD_PERCENTILE\" \
#             \"$DEFAULT_HIGH_THRESHOLD_PERCENTILE\" \
#             \"$bs\" \
#             \"$DEFAULT_GRADIENT_ACCUMULATION_STEPS\" \
#             \"2\" \
#             \"full\" \
#             \"$GPU_IDX\"")
#     done
# done

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
                \"$DEFAULT_BATCH_SIZE\" \
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
            \"$DEFAULT_BATCH_SIZE\" \
            \"$DEFAULT_GRADIENT_ACCUMULATION_STEPS\" \
            \"2\" \
            \"full\" \
            \"$GPU_IDX\"")
    done
done

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
            \"$DEFAULT_BATCH_SIZE\" \
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
            \"$DEFAULT_BATCH_SIZE\" \
            \"$DEFAULT_GRADIENT_ACCUMULATION_STEPS\" \
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
        \"$DEFAULT_BATCH_SIZE\" \
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
        \"$DEFAULT_BATCH_SIZE\" \
        \"$DEFAULT_GRADIENT_ACCUMULATION_STEPS\" \
        \"2\" \
        \"partial\" \
        \"$GPU_IDX\"")
done

# ===============================
# Epoch 3 Variant
# ===============================
echo "Collecting Epoch 3 Training..."
for dataset in "${DATASET_LIST[@]}"; do
    model_name="navi_${dataset,,}_epoch3"
    log_suffix="epoch3"
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
        \"$DEFAULT_BATCH_SIZE\" \
        \"$DEFAULT_GRADIENT_ACCUMULATION_STEPS\" \
        \"3\" \
        \"full\" \
        \"$GPU_IDX\"")
done

# ===============================
# Seed Variance Variants (Default Configuration)
# ===============================
echo "Collecting Seed Variance Variants..."
for dataset in "${DATASET_LIST[@]}"; do
    for seed in "${SEED_VARIANTS[@]}"; do
        model_name="navi_${dataset,,}_seed${seed}"
        log_suffix="seed${seed}"
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
            \"$seed\" \
            \"$DEFAULT_ENTROPY_THRESHOLD_METHOD\" \
            \"$DEFAULT_LOW_THRESHOLD_PERCENTILE\" \
            \"$DEFAULT_HIGH_THRESHOLD_PERCENTILE\" \
            \"$DEFAULT_BATCH_SIZE\" \
            \"$DEFAULT_GRADIENT_ACCUMULATION_STEPS\" \
            \"2\" \
            \"full\" \
            \"$GPU_IDX\"")
    done
done

# ===============================
# Execute jobs in batches of 3
# ===============================
echo ""
echo "Total jobs collected: ${#ALL_JOBS[@]}"
echo "Executing in batches of 3..."
echo ""

total_jobs=${#ALL_JOBS[@]}
batch_num=1
JOBS_PER_BATCH=3

failed_batches=0
for ((i=0; i<total_jobs; i+=JOBS_PER_BATCH)); do
    job1="${ALL_JOBS[$i]}"
    job2="${ALL_JOBS[$((i+1))]:-}"
    job3="${ALL_JOBS[$((i+2))]:-}"

    if [ -n "${job3}" ]; then
        echo "Batch $batch_num: Running jobs $((i+1)), $((i+2)), and $((i+3)) of $total_jobs"
        if ! run_batch_of_3 "$job1" "$job2" "$job3"; then
            failed_batches=$((failed_batches + 1))
        fi
    elif [ -n "${job2}" ]; then
        echo "Batch $batch_num: Running jobs $((i+1)) and $((i+2)) of $total_jobs"
        eval "$job1" &
        pid1=$!
        eval "$job2" &
        pid2=$!
        wait "$pid1" 2>/dev/null
        exit1=$?
        wait "$pid2" 2>/dev/null
        exit2=$?
        if [ $((exit1 | exit2)) -ne 0 ]; then
            failed_batches=$((failed_batches + 1))
            echo "  ⚠️ Batch completed with errors"
        else
            echo "  ✓ Batch completed successfully"
        fi
        echo "---"
    else
        echo "Batch $batch_num: Running final job $((i+1)) of $total_jobs"
        eval "$job1"
        if [ $? -ne 0 ]; then
            failed_batches=$((failed_batches + 1))
            echo "  ⚠️ Job failed"
        else
            echo "  ✓ Job completed"
        fi
        echo "---"
    fi

    batch_num=$((batch_num + 1))
done

if [ $failed_batches -gt 0 ]; then
    echo "⚠️ Warning: $failed_batches batch(es) failed. Check logs for details."
fi

# ===============================
# Summary
# ===============================
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

# echo "Default Model:"
# for dataset in "${DATASET_LIST[@]}"; do
#     echo "- navi_${dataset,,}_default: ./models/navi_${dataset,,}_default"
# done

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
echo "Batch Size Variants (Negative set size ablation):"
for dataset in "${DATASET_LIST[@]}"; do
    for bs in "${BATCH_SIZE_VARIANTS[@]}"; do
        echo "- navi_${dataset,,}_bs${bs}: ./models/navi_${dataset,,}_bs${bs}"
    done
done

echo ""
echo "Header Encoder Variants:"
for dataset in "${DATASET_LIST[@]}"; do
    echo "- navi_${dataset,,}_hefrozen: ./models/navi_${dataset,,}_hefrozen"
    echo "- navi_${dataset,,}_hepartial: ./models/navi_${dataset,,}_hepartial"
done

echo ""
echo "Ablation Variants:"
for dataset in "${DATASET_LIST[@]}"; do
    for ablation in "${ABLATION_VARIANTS[@]}"; do
        echo "- navi_${dataset,,}_${ablation}: ./models/navi_${dataset,,}_${ablation}"
    done
done

echo ""
echo "Epoch 3 Training:"
for dataset in "${DATASET_LIST[@]}"; do
    echo "- navi_${dataset,,}_epoch3: ./models/navi_${dataset,,}_epoch3"
done

echo ""
echo "Seed Variance Variants:"
for dataset in "${DATASET_LIST[@]}"; do
    for seed in "${SEED_VARIANTS[@]}"; do
        echo "- navi_${dataset,,}_seed${seed}: ./models/navi_${dataset,,}_seed${seed}"
    done
done

echo ""
echo "Completed training all models."
echo ""
echo "Log files are in logs/ directory with pattern: navi_<dataset>_<variant>_seed<seed>.log"
echo "To monitor all logs:"
echo "tail -f logs/navi_*.log"
echo ""
echo "To check GPU usage:"
echo "nvidia-smi"