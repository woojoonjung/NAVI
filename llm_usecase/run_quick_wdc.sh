#!/bin/bash

# Exit on error
set -e

# Define variables
DATA_DIR="data"
INDEX_DIR="index"
RUNS_DIR="runs"
PROMPTS_DIR="prompts"
LOGS_DIR="logs"
LLM_PROVIDER="openai"
LLM_MODEL="gpt-4o-mini"

# Create necessary directories
mkdir -p $RUNS_DIR/zero_shot $RUNS_DIR/wdc_rag_bert $RUNS_DIR/wdc_rag_ours
mkdir -p $LOGS_DIR

echo "=== LLM Case Study: Product QA Evaluation with WDC Data (Quick) ==="
echo "Comparing three paradigms: Zero-shot, RAG-BERT, RAG-Atlas"
echo "Dataset: WDC Product Data (Cleaned) - Using existing indexes"
echo

# Check if indexes exist
if [ ! -f "$INDEX_DIR/wdc_product_bert/index.faiss" ]; then
    echo "Error: BERT index not found at $INDEX_DIR/wdc_product_bert/index.faiss"
    echo "Please run the full pipeline first or build indexes manually"
    exit 1
fi

if [ ! -f "$INDEX_DIR/wdc_product_ours/index.faiss" ]; then
    echo "Error: Atlas index not found at $INDEX_DIR/wdc_product_ours/index.faiss"
    echo "Please run the full pipeline first or build indexes manually"
    exit 1
fi

echo "=== Step 1: Retrieval with BERT ==="
python scripts/retrieve.py \
    --index_dir $INDEX_DIR/wdc_product_bert \
    --query_file $DATA_DIR/qa/qas_product.jsonl \
    --model_type bert \
    --top_k 10 \
    --enable_struct_parse \
    --llm_provider openai \
    --llm_model gpt-4o-mini > $RUNS_DIR/wdc_rag_bert/retrieved.jsonl 2> $LOGS_DIR/retrieve_wdc_bert.log

echo
echo "=== Step 2: Retrieval with Atlas (Multi-Segment) ==="
python scripts/retrieve.py \
    --index_dir $INDEX_DIR/wdc_product_ours \
    --query_file $DATA_DIR/qa/qas_product.jsonl \
    --model_type ours \
    --top_k 10 \
    --enable_struct_parse \
    --use_multi_segment_retrieval \
    --llm_provider openai \
    --llm_model gpt-4o-mini > $RUNS_DIR/wdc_rag_ours/retrieved.jsonl 2> $LOGS_DIR/retrieve_wdc_ours.log

echo
echo "=== Step 3: Zero-shot Answering ==="
python scripts/answer.py \
    --paradigm zero_shot \
    --query_file $DATA_DIR/qa/qas_product.jsonl \
    --output_dir $RUNS_DIR/zero_shot \
    --llm_provider $LLM_PROVIDER \
    --llm_model $LLM_MODEL \
    --prompts_dir $PROMPTS_DIR \
    > $LOGS_DIR/answer_zero_shot_wdc.log 2>&1

echo
echo "=== Step 4: RAG-BERT Answering ==="
python scripts/answer.py \
    --paradigm rag_bert \
    --query_file $DATA_DIR/qa/qas_product.jsonl \
    --retrieved_file $RUNS_DIR/wdc_rag_bert/retrieved.jsonl \
    --data_dir $DATA_DIR \
    --output_dir $RUNS_DIR/wdc_rag_bert \
    --llm_provider $LLM_PROVIDER \
    --llm_model $LLM_MODEL \
    --prompts_dir $PROMPTS_DIR \
    > $LOGS_DIR/answer_wdc_rag_bert.log 2>&1

echo
echo "=== Step 5: RAG-Atlas Answering ==="
python scripts/answer.py \
    --paradigm rag_ours \
    --query_file $DATA_DIR/qa/qas_product.jsonl \
    --retrieved_file $RUNS_DIR/wdc_rag_ours/retrieved.jsonl \
    --data_dir $DATA_DIR \
    --output_dir $RUNS_DIR/wdc_rag_ours \
    --llm_provider $LLM_PROVIDER \
    --llm_model $LLM_MODEL \
    --prompts_dir $PROMPTS_DIR \
    > $LOGS_DIR/answer_wdc_rag_ours.log 2>&1

echo
echo "=== Experiment Summary ==="
echo "Results saved in:"
echo "  - Zero-shot: $RUNS_DIR/zero_shot/"
echo "  - RAG-BERT: $RUNS_DIR/wdc_rag_bert/"
echo "  - RAG-Atlas: $RUNS_DIR/wdc_rag_ours/"
echo "  - Logs: $LOGS_DIR/"
echo "  - Analysis Report: $RUNS_DIR/analysis_report_wdc.html"
echo
echo "Dataset: WDC Product Data (6,212 cleaned products)"
echo "Models: BERT vs Atlas (Custom Table Embeddings)"
echo "Paradigms: Zero-shot vs RAG-BERT vs RAG-Atlas"
echo
echo "Quick WDC experiment finished successfully!"
