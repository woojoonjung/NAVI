#!/bin/bash
# Run the complete segment-wise retrieval experiment

set -e

REPO_ROOT="/home/susik/navi_private"
LLM_USECASE_DIR="$REPO_ROOT/llm_usecase"
EXPERIMENTS_DIR="$REPO_ROOT/experiments"

cd "$REPO_ROOT"

# Activate conda environment
source ~/.bashrc 2>/dev/null || true
conda activate navi 2>/dev/null || true

export PYTHONPATH="$LLM_USECASE_DIR:$PYTHONPATH"

echo "=== Segment-wise Retrieval Experiment ==="
echo ""

# Step 1: Generate queries (already done, but can regenerate)
echo "Step 1: Generating queries..."
python3 "$EXPERIMENTS_DIR/generate_segment_queries.py" \
    --queries_per_domain 10 \
    --output "$EXPERIMENTS_DIR/segment_retrieval_queries.jsonl"

# Step 2: Build indexes (run in background, check if they exist first)
echo ""
echo "Step 2: Building segment-level indexes..."

# Check if indexes exist
if [ ! -f "$LLM_USECASE_DIR/index/product_bert_segments/index.faiss" ]; then
    echo "Building BERT product index..."
    python3 "$LLM_USECASE_DIR/scripts/build_index.py" \
        --data_dir data/cleaned/Product/test \
        --model_type bert \
        --dataset wdc_product \
        --output_dir "$LLM_USECASE_DIR/index/product_bert_segments" \
        > "$LLM_USECASE_DIR/logs/build_product_bert.log" 2>&1 &
    BERT_PRODUCT_PID=$!
else
    echo "BERT product index already exists"
    BERT_PRODUCT_PID=""
fi

if [ ! -f "$LLM_USECASE_DIR/index/product_navi_segments/index.faiss" ]; then
    echo "Building NAVI product index..."
    python3 "$LLM_USECASE_DIR/scripts/build_index.py" \
        --data_dir data/cleaned/Product/test \
        --model_type ours \
        --dataset wdc_product \
        --output_dir "$LLM_USECASE_DIR/index/product_navi_segments" \
        > "$LLM_USECASE_DIR/logs/build_product_navi.log" 2>&1 &
    NAVI_PRODUCT_PID=$!
else
    echo "NAVI product index already exists"
    NAVI_PRODUCT_PID=""
fi

if [ ! -f "$LLM_USECASE_DIR/index/movie_bert_segments/index.faiss" ]; then
    echo "Building BERT movie index..."
    python3 "$LLM_USECASE_DIR/scripts/build_index.py" \
        --data_dir data/cleaned/Movie/test \
        --model_type bert \
        --dataset wdc_movie \
        --output_dir "$LLM_USECASE_DIR/index/movie_bert_segments" \
        > "$LLM_USECASE_DIR/logs/build_movie_bert.log" 2>&1 &
    BERT_MOVIE_PID=$!
else
    echo "BERT movie index already exists"
    BERT_MOVIE_PID=""
fi

if [ ! -f "$LLM_USECASE_DIR/index/movie_navi_segments/index.faiss" ]; then
    echo "Building NAVI movie index..."
    python3 "$LLM_USECASE_DIR/scripts/build_index.py" \
        --data_dir data/cleaned/Movie/test \
        --model_type ours \
        --dataset wdc_movie \
        --output_dir "$LLM_USECASE_DIR/index/movie_navi_segments" \
        > "$LLM_USECASE_DIR/logs/build_movie_navi.log" 2>&1 &
    NAVI_MOVIE_PID=$!
else
    echo "NAVI movie index already exists"
    NAVI_MOVIE_PID=""
fi

# Wait for all indexing to complete
echo ""
echo "Waiting for indexes to be built..."
for pid in "$BERT_PRODUCT_PID" "$NAVI_PRODUCT_PID" "$BERT_MOVIE_PID" "$NAVI_MOVIE_PID"; do
    if [ -n "$pid" ]; then
        wait $pid
        echo "Index building completed (PID: $pid)"
    fi
done

echo ""
echo "All indexes built!"

# Step 3: Run retrieval for product domain
echo ""
echo "Step 3: Running retrieval experiments..."

# Filter queries by domain
python3 -c "
import json
queries = [json.loads(l) for l in open('$EXPERIMENTS_DIR/segment_retrieval_queries.jsonl')]
product_queries = [q for q in queries if q['domain'] == 'product']
movie_queries = [q for q in queries if q['domain'] == 'movie']
with open('$EXPERIMENTS_DIR/segment_retrieval_queries_product.jsonl', 'w') as f:
    for q in product_queries:
        f.write(json.dumps(q) + '\n')
with open('$EXPERIMENTS_DIR/segment_retrieval_queries_movie.jsonl', 'w') as f:
    for q in movie_queries:
        f.write(json.dumps(q) + '\n')
print(f'Filtered: {len(product_queries)} product, {len(movie_queries)} movie queries')
"

# Product domain
echo "Running product domain retrieval..."
python3 "$EXPERIMENTS_DIR/segment_retrieval.py" \
    --queries_file "$EXPERIMENTS_DIR/segment_retrieval_queries_product.jsonl" \
    --product_data data/cleaned/Product/test/WDC_product_for_cls.jsonl \
    --movie_data data/cleaned/Movie/test/WDC_movie_for_cls.jsonl \
    --bert_index_dir "$LLM_USECASE_DIR/index/product_bert_segments" \
    --navi_index_dir "$LLM_USECASE_DIR/index/product_navi_segments" \
    --output_file "$EXPERIMENTS_DIR/segment_retrieval_results_product.json" \
    --top_k 5 \
    --model_type both

# Movie domain  
echo "Running movie domain retrieval..."
python3 "$EXPERIMENTS_DIR/segment_retrieval.py" \
    --queries_file "$EXPERIMENTS_DIR/segment_retrieval_queries_movie.jsonl" \
    --product_data data/cleaned/Product/test/WDC_product_for_cls.jsonl \
    --movie_data data/cleaned/Movie/test/WDC_movie_for_cls.jsonl \
    --bert_index_dir "$LLM_USECASE_DIR/index/movie_bert_segments" \
    --navi_index_dir "$LLM_USECASE_DIR/index/movie_navi_segments" \
    --output_file "$EXPERIMENTS_DIR/segment_retrieval_results_movie.json" \
    --top_k 5 \
    --model_type both

# Step 4: Format results
echo ""
echo "Step 4: Formatting results..."

# Combine results
python3 -c "
import json
import sys

product_results = json.load(open('$EXPERIMENTS_DIR/segment_retrieval_results_product.json'))
movie_results = json.load(open('$EXPERIMENTS_DIR/segment_retrieval_results_movie.json'))

# Filter by domain
product_only = [r for r in product_results if r['domain'] == 'product']
movie_only = [r for r in movie_results if r['domain'] == 'movie']

all_results = product_only + movie_only

with open('$EXPERIMENTS_DIR/segment_retrieval_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f'Combined {len(product_only)} product and {len(movie_only)} movie results')
"

# Format output
python3 "$EXPERIMENTS_DIR/format_segment_results.py" \
    --results_file "$EXPERIMENTS_DIR/segment_retrieval_results.json" \
    --output_file "$EXPERIMENTS_DIR/segment_retrieval_results.txt" \
    --model both

echo ""
echo "=== Experiment Complete ==="
echo "Results saved to: $EXPERIMENTS_DIR/segment_retrieval_results.txt"
