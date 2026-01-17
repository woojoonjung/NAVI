#!/bin/bash

# Domain Consistency Experiment Pipeline
# This script runs the domain consistency experiment pipeline:
# 1. Get header embeddings  
# 2. Run domain consistency experiment
# Note: Canonical sets are expected to already exist in artifacts/lexvar/

set -e  # Exit on any error

# Configuration
DATA_DIR="data"
ARTIFACTS_DIR="artifacts/lexvar"
DOMAINS=("cleaned/Movie" "cleaned/Product")
MODELS=("bert" "tapas" "haetae" "navi")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if required directories exist
check_directories() {
    log "Checking required directories..."
    
    if [ ! -d "$DATA_DIR" ]; then
        error "Data directory '$DATA_DIR' not found!"
        exit 1
    fi
    
    for domain in "${DOMAINS[@]}"; do
        domain_path="$DATA_DIR/$domain"
        validation_path="$domain_path/validation"
        
        if [ ! -d "$domain_path" ]; then
            error "Domain directory '$domain_path' not found!"
            exit 1
        fi
        
        if [ ! -d "$validation_path" ]; then
            error "Validation directory '$validation_path' not found!"
            exit 1
        fi
    done
    
    success "All required directories found"
}

# Check if canonical sets exist
check_canonical_sets() {
    log "Checking for canonical sets..."
    
    local missing_canonical=()
    
    for domain in "${DOMAINS[@]}"; do
        # Extract domain name from path (e.g., 'cleaned/Movie' -> 'movie')
        domain_name=$(echo "$domain" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]')
        canonical_file="$ARTIFACTS_DIR/canonical_sets_${domain_name}.json"
        
        if [ ! -f "$canonical_file" ]; then
            missing_canonical+=("$canonical_file")
        fi
    done
    
    if [ ${#missing_canonical[@]} -gt 0 ]; then
        error "Canonical sets not found: ${missing_canonical[*]}"
        error "Please ensure the curated canonical sets files exist in $ARTIFACTS_DIR:"
        error "  - canonical_sets_movie.json"
        error "  - canonical_sets_product.json"
        exit 1
    else
        success "Canonical sets found"
    fi
}

# Step 1: Get header embeddings
get_header_embeddings() {
    log "Step 1: Getting header embeddings..."
    
    python experiments/domain_consistency_get_header_embeddings.py \
        --data_dir "$DATA_DIR" \
        --artifacts_dir "$ARTIFACTS_DIR" \
        --domains "${DOMAINS[@]}" \
        --models "${MODELS[@]}"
    
    if [ $? -eq 0 ]; then
        success "Header embeddings generated successfully"
    else
        error "Failed to generate header embeddings"
        exit 1
    fi
}

# Step 2: Run domain consistency experiment
run_domain_consistency_exp() {
    log "Step 2: Running domain consistency experiment..."
    
    python experiments/domain_consistency_exp.py \
        --artifacts_dir "$ARTIFACTS_DIR" \
        --domains "${DOMAINS[@]}" \
        --models "${MODELS[@]}"
    
    if [ $? -eq 0 ]; then
        success "Domain consistency experiment completed successfully"
    else
        error "Failed to run domain consistency experiment"
        exit 1
    fi
}

# Main execution
main() {
    log "Starting Domain Consistency Experiment Pipeline"
    log "=============================================="
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help|-h)
                echo "Usage: $0 [--help]"
                echo ""
                echo "Options:"
                echo "  --help, -h         Show this help message"
                echo ""
                echo "This script runs the domain consistency experiment pipeline:"
                echo "1. Check required directories (data/cleaned/{domain}/validation)"
                echo "2. Check for existing canonical sets files"
                echo "3. Get header embeddings for all models"
                echo "4. Run domain consistency experiment"
                echo ""
                echo "Configuration:"
                echo "  Data directory: $DATA_DIR"
                echo "  Artifacts directory: $ARTIFACTS_DIR"
                echo "  Domains: ${DOMAINS[*]}"
                echo "  Models: ${MODELS[*]}"
                echo ""
                echo "Note: Canonical sets must exist in $ARTIFACTS_DIR:"
                echo "  - canonical_sets_movie.json"
                echo "  - canonical_sets_product.json"
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    # Create artifacts directory
    mkdir -p "$ARTIFACTS_DIR"
    
    # Run pipeline steps
    check_directories
    check_canonical_sets
    get_header_embeddings
    run_domain_consistency_exp
    
    log "=============================================="
    success "Domain Consistency Experiment Pipeline completed successfully!"
    log "Results saved in: $ARTIFACTS_DIR"
    log "Check the following for results:"
    log "  - CSV files: results_clustering_{model}_{domain}.csv"
    log "  - Plots: plots/{model}_{domain}_umap_top5.png"
    log "  - Actor plots (Movie only): plots/{model}_movie_actor_clustering.png"
}

# Run main function with all arguments
main "$@"