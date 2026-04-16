#!/bin/bash

# Run row classification experiments

set -e

# Configuration
MODE="${1:-lm_encoders}"  # Default to lm_encoders if not provided
DOMAIN="${2:-Movie}"  # Default to Movie if not provided
EMBEDDING_TYPE="${3:-cls}"  # Default to cls if not provided
# Remaining args (e.g. --cm2_checkpoint /path --cm2_table_flag 0) are forwarded
EXTRA_ARGS=("${@:4}")

echo "=== Row Classification Experiment ==="
echo "Mode: $MODE"
echo "Domain: $DOMAIN"
echo "Embedding Type: $EMBEDDING_TYPE"
echo ""

python experiments/downstream_tasks/row_classification.py \
    --mode "$MODE" \
    --domain "$DOMAIN" \
    --embedding_type "$EMBEDDING_TYPE" \
    "${EXTRA_ARGS[@]}"

echo ""
echo "=== Experiment Complete ==="
