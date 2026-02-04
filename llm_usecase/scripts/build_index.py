#!/usr/bin/env python3

import os
import sys
import json
import argparse
import faiss
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModel
import sys
import glob

# Add the project root to the path
sys.path.append('/home/work/.default/woojun/navi')
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # Add repo root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # Add llm_usecase

from dataprep import NaviDataset, BertDataset
from model.navi import NaviForMaskedLM

def find_epoch_path(base_path, epoch=2):
    """Find epoch directory using glob pattern"""
    pattern = os.path.join(base_path, f"*epoch_{epoch}")
    matches = glob.glob(pattern)
    if not matches:
        pattern = os.path.join(base_path, f"epoch_{epoch}")
        if os.path.exists(pattern):
            return pattern
    if matches:
        matches.sort(key=len, reverse=True)
        return matches[0]
    return None

def extract_header_value_embeddings_from_positions(embeddings, header_positions, value_positions):
    """
    Extracts header and value embeddings from contextualized embeddings based on their token positions.
    This is a utility function adapted for inference, assuming a batch size of 1.
    """
    hidden_size = embeddings.size(-1)
    
    # Assuming batch size is 1, so we access the first element
    embeddings = embeddings.squeeze(0) # From (1, seq_len, hidden) to (seq_len, hidden)
    
    header_pos_dict = header_positions[0] if isinstance(header_positions, list) else header_positions
    value_pos_dict = value_positions[0] if isinstance(value_positions, list) else value_positions

    header_keys = list(header_pos_dict.keys())
    num_headers = len(header_keys)
    
    header_embeds = torch.zeros((1, num_headers, hidden_size), device=embeddings.device)
    val_embeds = torch.zeros((1, num_headers, hidden_size), device=embeddings.device)
    
    for i, header_name in enumerate(header_keys):
        # Extract header embedding
        h_token_indices = header_pos_dict.get(header_name, [])
        if h_token_indices:
            header_tokens = embeddings[h_token_indices]
            header_embeds[0, i] = header_tokens.mean(dim=0)
            
        # Extract value embedding
        v_token_indices = value_pos_dict.get(header_name, [])
        if v_token_indices:
            value_tokens = embeddings[v_token_indices]
            val_embeds[0, i] = value_tokens.mean(dim=0)
            
    return header_embeds, val_embeds

def main(args):
    # Map dataset names to data files
    dataset_files = {
        'wdc_product': 'WDC_product_for_cls.jsonl',
        'wdc_movie': 'WDC_movie_for_cls.jsonl',
        'amazon_product': 'Amazon_product_for_cls_cleaned.jsonl'
    }
    
    # Load table data
    # Try tables_cleaned first, then direct path
    table_file = os.path.join(args.data_dir, 'tables_cleaned', dataset_files[args.dataset])
    if not os.path.exists(table_file):
        table_file = os.path.join(args.data_dir, dataset_files[args.dataset])
    with open(table_file, 'r') as f:
        table_data = [json.loads(line) for line in f]
    if getattr(args, 'max_rows', None) is not None and args.max_rows > 0:
        table_data = table_data[: args.max_rows]
        print(f"Using first {len(table_data)} rows (--max_rows={args.max_rows})", file=sys.stderr)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", file=sys.stderr)
    
    # Map dataset to model paths (relative to repo root)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_paths = {
        'wdc_product': {
            'bert': os.path.join(repo_root, 'models', 'bert_product', 'epoch_2'),
            'ours': os.path.join(repo_root, 'models', 'navi_product')
        },
        'wdc_movie': {
            'bert': os.path.join(repo_root, 'models', 'bert_movie', 'epoch_2'),
            'ours': os.path.join(repo_root, 'models', 'navi_movie')
        },
        'amazon_product': {
            'bert': os.path.join(repo_root, 'models', 'bert_product', 'epoch_2'),
            'ours': os.path.join(repo_root, 'models', 'navi_product')
        }
    }
    
    if args.model_type == 'bert':
        # Load fine-tuned BERT model
        model_path = model_paths[args.dataset]['bert']
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained(model_path, local_files_only=True)
        model.to(device)
        model.eval()
        print(f"Loaded fine-tuned BERT from {model_path}", file=sys.stderr)
        print("BERT: Using segment-level indexing (header: value pairs)", file=sys.stderr)
        
        # For BERT segment-level indexing, we don't need the dataset
        # We'll encode each segment directly
        dataset = None
        
    elif args.model_type == 'ours':
        # Load Navi model - find epoch_2 directory
        base_path = model_paths[args.dataset]['ours']
        model_path = find_epoch_path(base_path, epoch=2)
        if not model_path:
            raise FileNotFoundError(f"NAVI model epoch_2 not found at {base_path}")
        model = NaviForMaskedLM(model_path=model_path)
        model.to(device)
        model.eval()
        print(f"Loaded Navi model from {model_path}", file=sys.stderr)
        print("NAVI: Using segment-level indexing (with segment embeddings)", file=sys.stderr)
        
        # Create Navi dataset
        dataset = NaviDataset(table_data, tokenizer_name='bert-base-uncased', max_length=512, ablation_mode="full")
        
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Use CPU FAISS index for now (GPU FAISS requires additional setup)
    index = faiss.IndexFlatL2(model.config.hidden_size)
    print(f"Created FAISS index with dimension {model.config.hidden_size}", file=sys.stderr)
    
    metadata = []
    embeddings = []
    
    print("Generating embeddings...", file=sys.stderr)
    with torch.no_grad():
        if args.model_type == 'bert':
            # For BERT: Segment-level indexing - encode each "header: value" pair individually
            for i in tqdm(range(len(table_data)), desc="Processing rows"):
                row_dict = table_data[i]
                row_id = row_dict.get('row_id', i)
                
                for header, value in row_dict.items():
                    # Skip non-data fields
                    if header == 'row_id':
                        continue
                    
                    # Skip empty values
                    if not value or str(value).strip() == '' or str(value) == 'None':
                        continue
                    
                    # Create text representation: "header: value"
                    segment_text = f"{header}: {value}"
                    
                    # Tokenize and encode
                    inputs = tokenizer(segment_text, return_tensors='pt', truncation=True, max_length=512)
                    input_ids = inputs['input_ids'].to(device)
                    attention_mask = inputs['attention_mask'].to(device)
                    
                    # Get BERT embeddings
                    outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
                    # Use CLS token embedding
                    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    
                    embeddings.append(embedding.flatten())
                    metadata.append({
                        'row_id': row_id,
                        'header': header,
                        'value': str(value),
                        'full_row': table_data[i]
                    })
        elif args.model_type == 'ours':
            # For NAVI: Use dataset for segment-level indexing
            for i in tqdm(range(len(dataset)), desc="Processing rows"):
                # Get data from dataset
                data = dataset[i]
                
                # Get row_id from original data
                row_id = table_data[i].get('row_id', i)
                
                # For NAVI: Segment-level indexing using segment embeddings
                input_ids = data['input_ids'].unsqueeze(0).to(device)
                attention_mask = data['attention_mask'].unsqueeze(0).to(device)
                position_ids = data['position_ids'].unsqueeze(0).to(device)
                segment_ids = data['segment_ids'].unsqueeze(0).to(device)
                header_strings = data.get('header_strings', [])
                header_positions = data.get('header_positions', {})
                value_positions = data.get('value_positions', {})
                
                if not header_strings or not header_positions or not value_positions:
                    print(f"Warning: Row {i} missing segment information, skipping", file=sys.stderr)
                    continue
                
                # Get Navi contextualized embeddings
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                    header_strings=[header_strings] if isinstance(header_strings, list) and all(isinstance(k, str) for k in header_strings) else header_strings
                )
                contextualized_embeddings = outputs[0]  # (1, seq_len, hidden)
                
                # Extract segment components
                E_univ, _ = model.bert.embeddings.header_encoder(
                    [header_strings] if isinstance(header_strings, list) and all(isinstance(k, str) for k in header_strings) else header_strings
                )
                H_ctx, V_ctx = extract_header_value_embeddings_from_positions(
                    contextualized_embeddings,
                    header_positions,
                    value_positions
                )
                
                # Create segment embeddings
                segment_embeddings = model.create_segment_embeddings(E_univ, H_ctx, V_ctx)  # (1, num_segments, hidden)
                segment_embeddings_np = segment_embeddings.squeeze(0).cpu().numpy()  # (num_segments, hidden)
                
                # Index each segment separately
                for seg_idx, header_name in enumerate(header_strings):
                    segment_embedding = segment_embeddings_np[seg_idx]  # (hidden,)
                    segment_value = table_data[i].get(header_name, '')
                    
                    embeddings.append(segment_embedding.flatten())
                    metadata.append({
                        'row_id': row_id,
                        'header': header_name,
                        'value': str(segment_value),
                        'full_row': table_data[i]  # Keep full row for retrieval
                    })
    
    # Convert to numpy array
    embeddings = np.array(embeddings).astype('float32')
    print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}", file=sys.stderr)
    
    # Print summary
    if args.model_type == 'bert':
        unique_rows = len(set(m['row_id'] for m in metadata))
        print(f"BERT: Indexed {len(embeddings)} segments (segment-level) from {unique_rows} rows", file=sys.stderr)
    else:
        unique_rows = len(set(m['row_id'] for m in metadata))
        print(f"NAVI: Indexed {len(embeddings)} segments (segment-level) from {unique_rows} rows", file=sys.stderr)
    
    # Add embeddings to FAISS index
    print("Creating FAISS index...", file=sys.stderr)
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, os.path.join(args.output_dir, 'index.faiss'))
    
    # Save metadata
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    
    # Save row-to-segments mapping (useful for retrieval)
    row_to_segments = {}
    for idx, meta in enumerate(metadata):
        row_id = meta['row_id']
        if row_id not in row_to_segments:
            row_to_segments[row_id] = []
        row_to_segments[row_id].append({
            'index': idx,
            'header': meta['header'],
            'value': meta['value']
        })
    with open(os.path.join(args.output_dir, 'row_to_segments.json'), 'w') as f:
        json.dump(row_to_segments, f)
    print(f"Saved row-to-segments mapping with {len(row_to_segments)} rows", file=sys.stderr)
    
    print(f"Index saved to {args.output_dir}", file=sys.stderr)
    print(f"Index contains {index.ntotal} vectors", file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True, choices=['bert', 'ours'])
    parser.add_argument('--dataset', type=str, required=True, choices=['wdc_product', 'wdc_movie', 'amazon_product'])
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)