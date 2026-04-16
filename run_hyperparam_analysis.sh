#!/bin/bash

# Run hyperparameter sweep experiments: all ./models/navi_{domain}* checkpoints.
# Writes JSON logs per domain, then LaTeX + TXT summary tables (paired by RUN_ID).

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_section() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:$PYTHONPATH}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
export RUN_ID
echo "Run ID (shared across JSON + tables): $RUN_ID"

DOMAINS=("Movie" "Product")

print_section "Starting Hyperparameter Analysis (all navi_{domain}* under ./models)"

# print_section "STEP 1: Masked prediction"
# for domain in "${DOMAINS[@]}"; do
#     echo -e "${YELLOW}Processing $domain domain...${NC}"
#     python experiments/masked_prediction/masked_prediction.py \
#         --model hyperparam_sensitivity \
#         --domain "$domain" \
#         --max_loaded_variants 4 \
#         --run_id "$RUN_ID"
#     echo -e "${GREEN}✓ Masked prediction completed for $domain${NC}"
# done

print_section "STEP 2: Row classification"
for domain in "${DOMAINS[@]}"; do
    echo -e "${YELLOW}Processing $domain domain...${NC}"
    python experiments/downstream_tasks/row_classification.py \
        --mode hyperparam_sensitivity \
        --domain "$domain" \
        --embedding_type cls \
        --max_loaded_variants 4 \
        --run_id "$RUN_ID"
    echo -e "${GREEN}✓ Row classification completed for $domain${NC}"
done

print_section "STEP 3: LaTeX + TXT tables"
python experiments/export_hyperparam_tables.py --run_id "$RUN_ID"

print_section "All experiments completed"
echo -e "${GREEN}✓ Hyperparameter analysis finished${NC}"
echo ""
echo "JSON logs: experiments/logs/*_${RUN_ID}.json"
echo "Tables:    experiments/logs/hyperparam_tables_${RUN_ID}_masked_prediction.{tex,txt}"
echo "           experiments/logs/hyperparam_tables_${RUN_ID}_row_classification.{tex,txt}"
