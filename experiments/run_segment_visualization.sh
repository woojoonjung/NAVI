#!/bin/bash

# Example script to run segment embedding visualization
# This demonstrates the complete pipeline from data extraction to visualization

set -e

# Configuration
MODEL_PATH="./models/navi_movie/epoch_2"
OUTPUT_DIR="./artifacts/segment_visualization"
N_TABLES=6
ROWS_PER_TABLE=169
SEED=42

echo "=== Segment Embedding Visualization Pipeline ==="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Tables: $N_TABLES, Rows per table: $ROWS_PER_TABLE"
echo "Seed: $SEED"
echo ""

# Step 1: Extract segment embeddings
echo "Step 1: Extracting segment embeddings..."
python experiments/visualization_get_segment_embeddings.py \
    --model_path "$MODEL_PATH" \
    --output_path "$OUTPUT_DIR/segments" \
    --n_tables $N_TABLES \
    --rows_per_table $ROWS_PER_TABLE \
    --random_state $SEED

# Step 2: Create visualizations
echo ""
echo "Step 2: Creating visualizations..."
python experiments/visualization_plot_segments.py \
    --input "$OUTPUT_DIR/segments/segments.json" \
    --outdir "$OUTPUT_DIR/plots" \
    --model-name "Navi" \
    --umap-n-neighbors 30 \
    --umap-min-dist 0.05 \
    --tsne-perplexity 30 \
    --seed $SEED \
    --preprocessing l2_normalize

echo ""
echo "=== Pipeline Complete ==="
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - $OUTPUT_DIR/plots/umap_entropy_genre.png"
echo "  - $OUTPUT_DIR/plots/umap_entropy_genre.svg"
echo "  - $OUTPUT_DIR/plots/tsne_entropy_genre.png"
echo "  - $OUTPUT_DIR/plots/tsne_entropy_genre.svg"
echo "  - $OUTPUT_DIR/plots/density_core_periphery_umap.png"
echo "  - $OUTPUT_DIR/plots/density_core_periphery_tsne.png"
echo "  - $OUTPUT_DIR/plots/coords_umap.csv"
echo "  - $OUTPUT_DIR/plots/coords_tsne.csv"
echo "  - $OUTPUT_DIR/plots/metrics.txt"
