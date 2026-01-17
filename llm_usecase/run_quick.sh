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
LLM_MODEL="gpt-4.1-nano"

# Create necessary directories
mkdir -p $RUNS_DIR/zero_shot $RUNS_DIR/movielens_rag_bert $RUNS_DIR/movielens_rag_ours
mkdir -p $LOGS_DIR

echo "=== LLM Case Study: Movie QA Evaluation with Movielens Data (Quick) ==="
echo "Comparing three paradigms: Zero-shot, RAG-BERT, RAG-Atlas"
echo "Dataset: Movielens Movies (Cleaned) - Using existing indexes"
echo

# Check if indexes exist
if [ ! -f "$INDEX_DIR/movielens_bert/index.faiss" ]; then
    echo "Error: BERT index not found at $INDEX_DIR/movielens_bert/index.faiss"
    echo "Please run the full pipeline first or build indexes manually"
    exit 1
fi

if [ ! -f "$INDEX_DIR/movielens_ours/index.faiss" ]; then
    echo "Error: Atlas index not found at $INDEX_DIR/movielens_ours/index.faiss"
    echo "Please run the full pipeline first or build indexes manually"
    exit 1
fi

echo "=== Step 1: Retrieval with BERT ==="
python scripts/retrieve.py \
    --index_dir $INDEX_DIR/movielens_bert \
    --query_file $DATA_DIR/qa/qas.jsonl \
    --model_type bert \
    --top_k 10 \
    --enable_struct_parse \
    --llm_provider openai \
    --llm_model gpt-4o-mini > $RUNS_DIR/movielens_rag_bert/retrieved.jsonl 2> $LOGS_DIR/retrieve_movielens_bert.log

echo
echo "=== Step 2: Retrieval with Atlas ==="
python scripts/retrieve.py \
    --index_dir $INDEX_DIR/movielens_ours \
    --query_file $DATA_DIR/qa/qas.jsonl \
    --model_type ours \
    --top_k 10 \
    --enable_struct_parse \
    --llm_provider openai \
    --llm_model gpt-4o-mini > $RUNS_DIR/movielens_rag_ours/retrieved.jsonl 2> $LOGS_DIR/retrieve_movielens_ours.log

echo
echo "=== Step 3: Zero-shot Answering ==="
python scripts/answer.py \
    --paradigm zero_shot \
    --query_file $DATA_DIR/qa/qas.jsonl \
    --output_dir $RUNS_DIR/zero_shot \
    --llm_provider $LLM_PROVIDER \
    --llm_model $LLM_MODEL \
    --prompts_dir $PROMPTS_DIR \
    > $LOGS_DIR/answer_zero_shot.log 2>&1

echo
echo "=== Step 4: RAG-BERT Answering ==="
python scripts/answer.py \
    --paradigm rag_bert \
    --query_file $DATA_DIR/qa/qas.jsonl \
    --retrieved_file $RUNS_DIR/movielens_rag_bert/retrieved.jsonl \
    --data_dir $DATA_DIR \
    --output_dir $RUNS_DIR/movielens_rag_bert \
    --llm_provider $LLM_PROVIDER \
    --llm_model $LLM_MODEL \
    --prompts_dir $PROMPTS_DIR \
    > $LOGS_DIR/answer_movielens_rag_bert.log 2>&1

echo
echo "=== Step 5: RAG-Atlas Answering ==="
python scripts/answer.py \
    --paradigm rag_ours \
    --query_file $DATA_DIR/qa/qas.jsonl \
    --retrieved_file $RUNS_DIR/movielens_rag_ours/retrieved.jsonl \
    --data_dir $DATA_DIR \
    --output_dir $RUNS_DIR/movielens_rag_ours \
    --llm_provider $LLM_PROVIDER \
    --llm_model $LLM_MODEL \
    --prompts_dir $PROMPTS_DIR \
    > $LOGS_DIR/answer_movielens_rag_ours.log 2>&1

# echo
# echo "=== Step 6: Evaluation ==="
# echo "Evaluating Zero-shot..."
# python scripts/evaluate.py \
#     --predictions_file $RUNS_DIR/zero_shot/predictions.jsonl \
#     --labels_file $DATA_DIR/qa/qas.jsonl \
#     --analyze_by_type \
#     --output_file $RUNS_DIR/zero_shot/evaluation.json \
#     > $LOGS_DIR/eval_zero_shot.log 2>&1

# echo "Evaluating RAG-BERT..."
# python scripts/evaluate.py \
#     --predictions_file $RUNS_DIR/movielens_rag_bert/predictions.jsonl \
#     --labels_file $DATA_DIR/qa/qas.jsonl \
#     --analyze_by_type \
#     --output_file $RUNS_DIR/movielens_rag_bert/evaluation.json \
#     > $LOGS_DIR/eval_movielens_rag_bert.log 2>&1

# echo "Evaluating RAG-Atlas..."
# python scripts/evaluate.py \
#     --predictions_file $RUNS_DIR/movielens_rag_ours/predictions.jsonl \
#     --labels_file $DATA_DIR/qa/qas.jsonl \
#     --analyze_by_type \
#     --output_file $RUNS_DIR/movielens_rag_ours/evaluation.json \
#     > $LOGS_DIR/eval_movielens_rag_ours.log 2>&1

# echo
# echo "=== Step 7: Analysis ==="
# python scripts/analyze_results.py \
#     --results_dir $RUNS_DIR \
#     --output_file $RUNS_DIR/analysis_report.html \
#     > $LOGS_DIR/analyze_results.log 2>&1

echo
echo "=== Experiment Summary ==="
echo "Results saved in:"
echo "  - Zero-shot: $RUNS_DIR/zero_shot/"
echo "  - RAG-BERT: $RUNS_DIR/movielens_rag_bert/"
echo "  - RAG-Atlas: $RUNS_DIR/movielens_rag_ours/"
echo "  - Logs: $LOGS_DIR/"
echo "  - Analysis Report: $RUNS_DIR/analysis_report.html"
echo
echo "Dataset: Movielens Movies (45,390 cleaned movies)"
echo "Models: BERT vs Atlas (Custom Table Embeddings)"
echo "Paradigms: Zero-shot vs RAG-BERT vs RAG-Atlas"
echo
echo "Quick experiment finished successfully!"
