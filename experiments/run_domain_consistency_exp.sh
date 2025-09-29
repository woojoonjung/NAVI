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
DOMAINS=("Movie_top100" "Product_top100")
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
        if [ ! -d "$DATA_DIR/$domain" ]; then
            error "Domain directory '$DATA_DIR/$domain' not found!"
            exit 1
        fi
    done
    
    success "All required directories found"
}

# Check if cleaned datasets exist, if not run preprocessing
check_cleaned_datasets() {
    log "Checking for cleaned datasets..."
    
    local missing_datasets=()
    
    for domain in "${DOMAINS[@]}"; do
        local cleaned_dir="$DATA_DIR/Quarter_${domain}_cleaned"
        if [ ! -d "$cleaned_dir" ]; then
            missing_datasets+=("$cleaned_dir")
        fi
    done
    
    if [ ${#missing_datasets[@]} -gt 0 ]; then
        warning "Cleaned datasets not found: ${missing_datasets[*]}"
        log "Running preprocessing pipeline to create cleaned datasets..."
        
        # Run preprocessing
        python dataset/preprocess.py \
            --movie_dir "$DATA_DIR/Movie_top100" \
            --product_dir "$DATA_DIR/Product_top100" \
            --product_max_rows 5000 \
            --heldout_start 450 \
            --heldout_end 459
        
        # Create symlinks for Quarter_* directories (expected by experiments)
        for domain in "${DOMAINS[@]}"; do
            local cleaned_dir="$DATA_DIR/Quarter_${domain}_cleaned"
            local source_dir="$DATA_DIR/${domain}_cleaned"
            
            if [ -d "$source_dir" ] && [ ! -d "$cleaned_dir" ]; then
                log "Creating symlink: $cleaned_dir -> $source_dir"
                ln -sf "../${domain}_cleaned" "$cleaned_dir"
            fi
        done
        
        success "Preprocessing completed and symlinks created"
    else
        success "Cleaned datasets found"
    fi
}

# Check if canonical sets exist
check_canonical_sets() {
    log "Checking for canonical sets..."
    
    local missing_canonical=()
    
    for domain in "${DOMAINS[@]}"; do
        local canonical_file="$ARTIFACTS_DIR/canonical_final_Quarter_${domain}_cleaned.json"
        if [ ! -f "$canonical_file" ]; then
            missing_canonical+=("$canonical_file")
        fi
    done
    
    if [ ${#missing_canonical[@]} -gt 0 ]; then
        error "Canonical sets not found: ${missing_canonical[*]}"
        error "Please run the canonical sets generation first:"
        error "python experiments/domain_consistency_get_canonical_sets.py \\"
        error "    --data_dir $DATA_DIR \\"
        error "    --artifacts_dir $ARTIFACTS_DIR \\"
        error "    --domains ${DOMAINS[@]/#/Quarter_} \\"
        error "    --cluster_method semantic \\"
        error "    --run_curation"
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
        --domains "${DOMAINS[@]/#/Quarter_}" \
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
        --domains "${DOMAINS[@]/#/Quarter_}" \
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
                echo "1. Check and prepare cleaned datasets"
                echo "2. Check for existing canonical sets"
                echo "3. Get header embeddings for all models"
                echo "4. Run domain consistency experiment"
                echo ""
                echo "Note: Canonical sets must be generated separately first using:"
                echo "python experiments/domain_consistency_get_canonical_sets.py \\"
                echo "    --data_dir data \\"
                echo "    --artifacts_dir artifacts/lexvar \\"
                echo "    --domains Quarter_Movie_top100_cleaned Quarter_Product_top100_cleaned \\"
                echo "    --cluster_method semantic \\"
                echo "    --run_curation"
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
    check_cleaned_datasets
    check_canonical_sets
    get_header_embeddings
    run_domain_consistency_exp
    
    log "=============================================="
    success "Domain Consistency Experiment Pipeline completed successfully!"
    log "Results saved in: $ARTIFACTS_DIR"
    log "Check the generated files for quantitative results"
}

# Run main function with all arguments
main "$@"