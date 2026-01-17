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
mkdir -p $INDEX_DIR/movielens_bert $INDEX_DIR/movielens_ours
mkdir -p $RUNS_DIR/zero_shot $RUNS_DIR/movielens_rag_bert $RUNS_DIR/movielens_rag_ours
mkdir -p $LOGS_DIR

echo "=== LLM Case Study: Movie QA Evaluation with Movielens Data ==="
echo "Comparing three paradigms: Zero-shot, RAG-BERT, RAG-Atlas"
echo "Dataset: Movielens Movies (Cleaned)"
echo

# Check if data files exist
if [ ! -f "$DATA_DIR/qa/qas.jsonl" ]; then
    echo "Error: QA data not found at $DATA_DIR/qa/qas.jsonl"
    exit 1
fi

if [ ! -f "$DATA_DIR/tables_cleaned/Movielens_movie_for_cls_cleaned.jsonl" ]; then
    echo "Error: Cleaned Movielens data not found at $DATA_DIR/tables_cleaned/Movielens_movie_for_cls_cleaned.jsonl"
    exit 1
fi

echo "=== Step 1: Building BERT Index for Movielens ==="
python scripts/build_index.py \
    --data_dir $DATA_DIR \
    --model_type bert \
    --output_dir $INDEX_DIR/movielens_bert \
    > $LOGS_DIR/build_movielens_bert_index.log 2>&1

echo
echo "=== Step 2: Building Atlas Index for Movielens ==="
python scripts/build_index.py \
    --data_dir $DATA_DIR \
    --model_type ours \
    --output_dir $INDEX_DIR/movielens_ours \
    > $LOGS_DIR/build_movielens_ours_index.log 2>&1

echo
echo "=== Step 3: Retrieval with BERT ==="
python scripts/retrieve.py \
    --index_dir $INDEX_DIR/movielens_bert \
    --query_file $DATA_DIR/qa/qas.jsonl \
    --model_type bert \
    --top_k 5 > $RUNS_DIR/movielens_rag_bert/retrieved.jsonl 2> $LOGS_DIR/retrieve_movielens_bert.log

echo
echo "=== Step 4: Retrieval with Atlas ==="
python scripts/retrieve.py \
    --index_dir $INDEX_DIR/movielens_ours \
    --query_file $DATA_DIR/qa/qas.jsonl \
    --model_type ours \
    --top_k 5 > $RUNS_DIR/movielens_rag_ours/retrieved.jsonl 2> $LOGS_DIR/retrieve_movielens_ours.log

echo
echo "=== Step 5: Zero-shot Answering ==="
python scripts/answer.py \
    --paradigm zero_shot \
    --query_file $DATA_DIR/qa/qas.jsonl \
    --output_dir $RUNS_DIR/zero_shot \
    --llm_provider $LLM_PROVIDER \
    --llm_model $LLM_MODEL \
    --prompts_dir $PROMPTS_DIR \
    > $LOGS_DIR/answer_zero_shot.log 2>&1

echo
echo "=== Step 6: RAG-BERT Answering ==="
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
echo "=== Step 7: RAG-Atlas Answering ==="
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

echo
echo "=== Step 8: Evaluation ==="
echo "Evaluating Zero-shot..."
python scripts/evaluate.py \
    --predictions_file $RUNS_DIR/zero_shot/predictions.jsonl \
    --labels_file $DATA_DIR/qa/qas.jsonl \
    --analyze_by_type \
    --output_file $RUNS_DIR/zero_shot/evaluation.json \
    > $LOGS_DIR/eval_zero_shot.log 2>&1

echo "Evaluating RAG-BERT..."
python scripts/evaluate.py \
    --predictions_file $RUNS_DIR/movielens_rag_bert/predictions.jsonl \
    --labels_file $DATA_DIR/qa/qas.jsonl \
    --analyze_by_type \
    --output_file $RUNS_DIR/movielens_rag_bert/evaluation.json \
    > $LOGS_DIR/eval_movielens_rag_bert.log 2>&1

echo "Evaluating RAG-Atlas..."
python scripts/evaluate.py \
    --predictions_file $RUNS_DIR/movielens_rag_ours/predictions.jsonl \
    --labels_file $DATA_DIR/qa/qas.jsonl \
    --analyze_by_type \
    --output_file $RUNS_DIR/movielens_rag_ours/evaluation.json \
    > $LOGS_DIR/eval_movielens_rag_ours.log 2>&1

echo
echo "=== Experiment Summary ==="
echo "Results saved in:"
echo "  - Zero-shot: $RUNS_DIR/zero_shot/"
echo "  - RAG-BERT: $RUNS_DIR/movielens_rag_bert/"
echo "  - RAG-Atlas: $RUNS_DIR/movielens_rag_ours/"
echo "  - Logs: $LOGS_DIR/"
echo
echo "Dataset: Movielens Movies (45,390 cleaned movies)"
echo "Models: BERT vs Atlas (Custom Table Embeddings)"
echo "Paradigms: Zero-shot vs RAG-BERT vs RAG-Atlas"
echo
echo "Experiment finished successfully!"
