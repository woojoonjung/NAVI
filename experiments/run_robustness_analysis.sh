#!/bin/bash

# Run robustness analysis experiments

set -e

# Configuration
DATA_DIR="${DATA_DIR:-data}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/robustness_analysis/robustness_results}"
DOMAINS=("cleaned/Movie" "cleaned/Product")
MODELS=("bert" "tapas" "haetae" "navi" "woSSI" "woMSM" "woESA" "woGHA" "woGHC")

echo "=== Robustness Analysis Experiment ==="
echo "Data Directory: $DATA_DIR"
echo "Artifacts Directory: $ARTIFACTS_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Domains: ${DOMAINS[*]}"
echo "Models: ${MODELS[*]}"
echo ""

python experiments/robustness_analysis/robustness_exp.py \
    --data_dir "$DATA_DIR" \
    --artifacts_dir "$ARTIFACTS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --domains "${DOMAINS[@]}" \
    --models "${MODELS[@]}" \
    --n_samples 100 \
    --n_permutations 5 \
    --synonym_map "$ARTIFACTS_DIR/schema_noise/synonym_map.json" \
    --embedding_types meanpooled cls \
    --skip_si \
    --skip_downstream \
    --run_masked_prediction

echo ""
echo "=== Experiment Complete ==="
