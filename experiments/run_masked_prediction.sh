#!/bin/bash

# Run masked prediction experiments

set -e

# Configuration
DOMAIN="${1:-Movie}"  # Default to Movie if not provided
MODEL_TYPE="${2:-baselines}"  # Default to baselines if not provided

echo "=== Masked Prediction Experiment ==="
echo "Domain: $DOMAIN"
echo "Model Type: $MODEL_TYPE"
echo ""

python experiments/masked_prediction/masked_prediction.py \
    --model "$MODEL_TYPE" \
    --domain "$DOMAIN"

echo ""
echo "=== Experiment Complete ==="
