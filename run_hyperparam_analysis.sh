#!/bin/bash

# Script to run hyperparameter analysis experiments
# Runs masked_prediction, row_clustering, and row_classification
# for align, ethresh, and tau variants

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print section headers
print_section() {
    echo ""
    echo "=========================================="
    echo "$1"
    echo "=========================================="
    echo ""
}

# Function to run experiment with error handling
run_experiment() {
    local script=$1
    local model_type=$2
    local domain=$3
    local embedding_type=${4:-"cls"}
    
    print_section "Running: $script --model $model_type --domain $domain"
    
    if [ "$script" = "experiments/row_clustering.py" ]; then
        python "$script" --model "$model_type" --domain "$domain" --embedding_type "$embedding_type"
    else
        python "$script" --model "$model_type" --domain "$domain"
    fi
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Successfully completed: $script --model $model_type --domain $domain${NC}"
    else
        echo -e "${RED}✗ Failed: $script --model $model_type --domain $domain${NC}"
        exit 1
    fi
}

# Main execution
print_section "Starting Hyperparameter Analysis Experiments"

# Domains to evaluate
DOMAINS=("Movie" "Product")

# Step 1: Masked Prediction for all variants
print_section "STEP 1: Masked Prediction Analysis"
for domain in "${DOMAINS[@]}"; do
    echo -e "${YELLOW}Processing $domain domain...${NC}"
    # Use hyperparam_sensitivity which includes align, ethresh, and tau variants
    run_experiment "experiments/masked_prediction.py" "hyperparam_sensitivity" "$domain"
done

# Step 2: Row Clustering for all variants
print_section "STEP 2: Row Clustering Analysis"
for domain in "${DOMAINS[@]}"; do
    echo -e "${YELLOW}Processing $domain domain...${NC}"
    # Use hyperparam_sensitivity which includes align, ethresh, and tau variants
    run_experiment "experiments/row_clustering.py" "hyperparam_sensitivity" "$domain" "cls"
done

# Step 3: Row Classification for all variants
print_section "STEP 3: Row Classification Analysis"
for domain in "${DOMAINS[@]}"; do
    echo -e "${YELLOW}Processing $domain domain...${NC}"
    # Use hyperparam_sensitivity which includes align, ethresh, and tau variants
    run_experiment "experiments/row_classification.py" "hyperparam_sensitivity" "$domain" "cls"
done

print_section "All Experiments Completed Successfully!"
echo -e "${GREEN}✓ All hyperparameter analysis experiments finished${NC}"
echo ""
echo "Results saved in:"
echo "  - experiments/logs/masked_prediction_hyperparam_sensitivity_*.json"
echo "  - experiments/logs/row_clustering_hyperparam_sensitivity_*.json"
echo "  - experiments/logs/row_classification_hyperparam_sensitivity_*.json"
