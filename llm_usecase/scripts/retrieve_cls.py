#!/usr/bin/env python3

import os
import sys
import json
import argparse
import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModel
from dotenv import load_dotenv

# Add the project root to the path
sys.path.append('/home/work/.default/woojun/navi')

def load_dotenv():
    """Load environment variables from .env file if it exists."""
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        from dotenv import load_dotenv
        load_dotenv(env_path)

def load_index(index_dir):
    """Load FAISS index and metadata."""
    index_path = os.path.join(index_dir, 'index.faiss')
    metadata_path = os.path.join(index_dir, 'metadata.json')
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Index files not found in {index_dir}")
    
    index = faiss.read_index(index_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return index, metadata

def load_model_and_tokenizer(model_type, device):
    """Load model and tokenizer based on model type."""
    if model_type == 'bert':
        model_path = '/home/work/.default/woojun/navi/models/bert_quarter_product/epoch_2'
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained(model_path, local_files_only=True)
        model.to(device)
        model.eval()
        print(f"Loaded BERT model from {model_path}", file=sys.stderr)
    elif model_type == 'ours':
        model_path = '/home/work/.default/woojun/navi/models/full_Quarter_Product_HVB_hv0p8_align0p5_vr0p5/epoch_2'
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained(model_path, local_files_only=True)
        model.to(device)
        model.eval()
        print(f"Loaded Navi model from {model_path}", file=sys.stderr)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, tokenizer

def get_cls_embedding(text, model, tokenizer, device):
    """Get CLS embedding for a given text."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        if hasattr(model, 'bert'):
            # BERT model
            hidden_states = model.bert(**inputs).last_hidden_state
        else:
            # Navi model
            hidden_states = model(**inputs).last_hidden_state
        
        # Use CLS token embedding
        cls_embedding = hidden_states[:, 0, :].detach().cpu().numpy()
    
    return cls_embedding

def main():
    parser = argparse.ArgumentParser(description='Retrieve documents using CLS embeddings')
    parser.add_argument('--index_dir', type=str, required=True, help='Directory containing FAISS index')
    parser.add_argument('--query_file', type=str, required=True, help='Path to query file (JSONL)')
    parser.add_argument('--model_type', type=str, required=True, choices=['bert', 'ours'], help='Model type')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top documents to retrieve')
    parser.add_argument('--output_file', type=str, help='Output file path (optional)')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", file=sys.stderr)
    
    # Load index and metadata
    print(f"Loading index from {args.index_dir}", file=sys.stderr)
    index, metadata = load_index(args.index_dir)
    print(f"Index loaded with {index.ntotal} vectors", file=sys.stderr)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_type, device)
    
    # Load queries
    with open(args.query_file, 'r') as f:
        queries = [json.loads(line) for line in f]
    
    print(f"Processing {len(queries)} queries", file=sys.stderr)
    
    # Process each query
    for query in tqdm(queries, desc="Processing queries"):
        questions = query['question']
        if not isinstance(questions, list):
            questions = [questions]
        
        for question_idx, question_text in enumerate(questions):
            # Get CLS embedding for the question
            query_embedding = get_cls_embedding(question_text, model, tokenizer, device)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search
            distances, indices = index.search(query_embedding, args.top_k)
            
            # Prepare retrieved documents
            retrieved_docs = []
            for i in range(len(distances[0])):
                doc_idx = indices[0][i]
                doc_metadata = metadata[doc_idx]
                retrieved_docs.append({
                    'table_id': doc_metadata['table_id'],
                    'row_id': doc_metadata['row_id'],
                    'content': doc_metadata['full_row'],
                    'score': float(distances[0][i])
                })
            
            # Create output
            variant_query_id = f"{query['id']}_{question_idx}"
            output = {
                'query_id': variant_query_id,
                'original_query_id': query['id'],
                'question_idx': question_idx,
                'question_text': question_text,
                'retrieved_docs': retrieved_docs
            }
            
            # Output to stdout or file
            if args.output_file:
                with open(args.output_file, 'a') as f:
                    f.write(json.dumps(output) + '\n')
            else:
                print(json.dumps(output))

if __name__ == '__main__':
    main()
