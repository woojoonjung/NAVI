#!/usr/bin/env python3

import os
import sys
import json
import argparse
import torch
import numpy as np
import faiss
from transformers import BertTokenizer, BertForMaskedLM

# Add the project root to the path
sys.path.append('/home/work/.default/woojun/navi')
from model.navi import NaviForMaskedLM

def load_model_and_tokenizer(model_type, device):
    """Load model and tokenizer based on type"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    if model_type == 'bert':
        model_path = '/home/work/.default/woojun/navi/models/bert_quarter_product/epoch_2'
        model = BertForMaskedLM.from_pretrained(model_path, local_files_only=True)
        model.to(device)
        model.eval()
        print(f"Loaded BERT from {model_path}", file=sys.stderr)
    elif model_type == 'ours':
        model_path = '/home/work/.default/woojun/navi/models/full_Quarter_Product_HVB_hv0p8_align0p5_vr0p5/epoch_2'
        model = NaviForMaskedLM(model_path=model_path)
        model.to(device)
        model.eval()
        print(f"Loaded Navi from {model_path}", file=sys.stderr)
    else:
        raise ValueError("model_type must be 'bert' or 'ours'")
    
    return model, tokenizer

def get_cls_embedding(question_text, model, tokenizer, model_type, device):
    """Get CLS embedding for a question"""
    if model_type == 'bert':
        # BERT CLS embedding
        inputs = tokenizer(question_text, return_tensors='pt', truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True
            )
            # Use CLS token embedding
            embedding = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
    
    elif model_type == 'ours':
        # Navi CLS embedding - match original retrieve.py approach
        qtext = f"Question: {question_text} [SEP]"
        inputs = tokenizer(qtext, return_tensors='pt', truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        batch_size, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        segment_ids = torch.zeros_like(input_ids, device=device)
        header_strings = [["question"]]
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                header_strings=header_strings,
                segment_ids=segment_ids
            )
            # Use CLS token embedding (Navi returns tuple: (contextualized_embeddings, logits))
            embedding = outputs[0][:, 0, :].cpu().numpy()
    
    return embedding

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_dir', type=str, required=True)
    parser.add_argument('--query_file', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True, choices=['bert', 'ours'])
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", file=sys.stderr)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_type, device)
    
    # Load FAISS index
    index_path = os.path.join(args.index_dir, 'index.faiss')
    index = faiss.read_index(index_path)
    print(f"Loaded FAISS index with {index.ntotal} vectors", file=sys.stderr)
    
    # Load metadata
    metadata_path = os.path.join(args.index_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"Loaded {len(metadata)} metadata entries", file=sys.stderr)
    
    # Load queries
    with open(args.query_file, 'r') as f:
        queries = [json.loads(line) for line in f]
    print(f"Loaded {len(queries)} queries", file=sys.stderr)
    
    # Process each query
    for query in queries:
        query_id = query['id']
        questions = query['question'] if isinstance(query['question'], list) else [query['question']]
        
        for question_idx, question_text in enumerate(questions):
            variant_query_id = f"{query_id}_{question_idx}"
            
            # Get CLS embedding
            query_embedding = get_cls_embedding(question_text, model, tokenizer, args.model_type, device)
            
            # Search
            distances, indices = index.search(query_embedding, args.top_k)
            
            # Format results
            retrieved_docs = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                doc_metadata = metadata[idx]
                retrieved_docs.append({
                    "table_id": None,
                    "row_id": doc_metadata['row_id'],
                    "content": doc_metadata['full_row'],
                    "score": float(dist)
                })
            
            # Output result
            result = {
                "query_id": variant_query_id,
                "original_query_id": query_id,
                "question_idx": question_idx,
                "question_text": question_text,
                "retrieved_docs": retrieved_docs,
                "structured_query": {}
            }
            
            print(json.dumps(result))

if __name__ == '__main__':
    main()
