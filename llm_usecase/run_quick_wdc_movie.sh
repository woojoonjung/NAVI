#!/bin/bash

# WDC Movie RAG Pipeline
# This script runs the complete RAG pipeline for WDC Movie data

set -e

# Configuration
QUERY_FILE="data/qa/qas_wdc_movie.jsonl"
DATA_DIR="data"
INDEX_DIR="index"
OUTPUT_DIR="runs"
TOP_K=3

echo "=== WDC Movie RAG Pipeline ==="
echo "Query file: $QUERY_FILE"
echo "Data directory: $DATA_DIR"
echo "Index directory: $INDEX_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Top-K: $TOP_K"
echo

# Check if query file exists
if [ ! -f "$QUERY_FILE" ]; then
    echo "Error: Query file $QUERY_FILE not found!"
    echo "Please generate QA pairs first using:"
    echo "python scripts/generate_qa.py --dataset wdc_movie --target_count 30 --rng_seed 42"
    exit 1
fi

# Check if indices exist
if [ ! -d "$INDEX_DIR/wdc_movie_bert" ] || [ ! -d "$INDEX_DIR/wdc_movie_ours" ]; then
    echo "Error: Indices not found!"
    echo "Please build indices first using:"
    echo "python scripts/build_index.py --data_dir data/tables_cleaned --model_type bert --output_dir index/wdc_movie_bert"
    echo "python scripts/build_index.py --data_dir data/tables_cleaned --model_type ours --output_dir index/wdc_movie_ours"
    exit 1
fi

echo "=== Step 1: Retrieval ==="
echo "Running BERT retrieval..."
mkdir -p "$OUTPUT_DIR/wdc_movie_rag_bert"
python scripts/retrieve.py \
    --query_file "$QUERY_FILE" \
    --index_dir "$INDEX_DIR/wdc_movie_bert" \
    --model_type bert \
    --top_k $TOP_K \
    --enable_struct_parse \
    --llm_provider openai \
    --llm_model gpt-4o-mini > "$OUTPUT_DIR/wdc_movie_rag_bert/retrieved.jsonl"

echo "Running Atlas retrieval..."
mkdir -p "$OUTPUT_DIR/wdc_movie_rag_ours"
python scripts/retrieve.py \
    --query_file "$QUERY_FILE" \
    --index_dir "$INDEX_DIR/wdc_movie_ours" \
    --model_type ours \
    --top_k $TOP_K \
    --enable_struct_parse \
    --llm_provider openai \
    --llm_model gpt-4o-mini \
    --use_multi_segment_retrieval > "$OUTPUT_DIR/wdc_movie_rag_ours/retrieved.jsonl"

echo "=== Step 2: Answer Generation ==="
echo "Generating answers for BERT..."
python scripts/answer.py \
    --query_file "$QUERY_FILE" \
    --retrieved_file "$OUTPUT_DIR/wdc_movie_rag_bert/retrieved.jsonl" \
    --data_dir "$DATA_DIR" \
    --paradigm rag_bert \
    --output_dir "$OUTPUT_DIR/wdc_movie_rag_bert" \
    --llm_provider openai \
    --llm_model gpt-4o-mini

echo "Generating answers for Atlas..."
python scripts/answer.py \
    --query_file "$QUERY_FILE" \
    --retrieved_file "$OUTPUT_DIR/wdc_movie_rag_ours/retrieved.jsonl" \
    --data_dir "$DATA_DIR" \
    --paradigm rag_ours \
    --output_dir "$OUTPUT_DIR/wdc_movie_rag_ours" \
    --llm_provider openai \
    --llm_model gpt-4o-mini

echo "=== Step 3: Evaluation ==="
echo "Evaluating BERT predictions..."
python scripts/evaluate.py \
    --predictions_file "$OUTPUT_DIR/wdc_movie_rag_bert/predictions.jsonl" \
    --ground_truth_file "$QUERY_FILE" \
    --output_file "$OUTPUT_DIR/wdc_movie_rag_bert/evaluation.json"

echo "Evaluating Atlas predictions..."
python scripts/evaluate.py \
    --predictions_file "$OUTPUT_DIR/wdc_movie_rag_ours/predictions.jsonl" \
    --ground_truth_file "$QUERY_FILE" \
    --output_file "$OUTPUT_DIR/wdc_movie_rag_ours/evaluation.json"

echo "=== Pipeline Complete ==="
echo "Results saved to:"
echo "  BERT: $OUTPUT_DIR/wdc_movie_rag_bert/"
echo "  Atlas: $OUTPUT_DIR/wdc_movie_rag_ours/"
echo
echo "To view results:"
echo "  cat $OUTPUT_DIR/wdc_movie_rag_bert/evaluation.json"
echo "  cat $OUTPUT_DIR/wdc_movie_rag_ours/evaluation.json"
