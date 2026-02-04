#!/usr/bin/env python3
"""
Segment-wise retrieval experiment with [MASK] tokens.
Uses mean-pooling of "header: value" pairs from full query input.
"""

import os
import sys
import json
import argparse
import faiss
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM
from pathlib import Path
from collections import defaultdict
import glob

# Add paths (repo root and llm_usecase for imports)
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, 'llm_usecase'))
sys.path.append('/home/work/.default/woojun/navi')
from model.navi import NaviForMaskedLM
from scripts.dataprep import NaviDataset

def extract_header_value_embeddings_from_positions(embeddings, header_positions, value_positions):
    """
    Extract header and value embeddings from contextualized embeddings based on token positions.
    Mean-pool header tokens and value tokens separately.
    """
    hidden_size = embeddings.size(-1)
    embeddings = embeddings.squeeze(0)  # (seq_len, hidden)
    
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

def encode_masked_query_bert(query_dict: dict, model, tokenizer, device, target_attribute: str):
    """
    Encode masked query using BERT with mean-pooling approach.
    Encode full query, then mean-pool header and value tokens for the masked segment.
    """
    # Create text representation of full query
    query_parts = []
    for header, value in query_dict.items():
        if value == "[MASK]":
            query_parts.append(f"{header}: {tokenizer.mask_token}")
        else:
            query_parts.append(f"{header}: {value}")
    
    query_text = " ".join(query_parts)
    
    # Tokenize and encode
    inputs = tokenizer(query_text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.bert(**inputs)
        contextualized_embeddings = outputs.last_hidden_state  # (1, seq_len, hidden)
    
    # Find positions of target attribute's header and value
    # Tokenize just the target part to find positions
    target_part = f"{target_attribute}: {tokenizer.mask_token}"
    target_tokens = tokenizer.tokenize(target_part)
    target_token_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    
    # Find these tokens in the full sequence
    input_ids = inputs['input_ids'][0].cpu().numpy()
    mask_token_id = tokenizer.mask_token_id
    
    # Find mask token position
    mask_positions = np.where(input_ids == mask_token_id)[0]
    if len(mask_positions) == 0:
        # Fallback: use CLS token
        return contextualized_embeddings[:, 0, :].cpu().numpy()
    
    mask_pos = mask_positions[0]
    
    # Find header tokens (tokens before the colon before [MASK])
    # Simple approach: tokens before mask position
    header_start = max(0, mask_pos - len(target_tokens))
    header_end = mask_pos
    
    # Extract embeddings
    header_tokens_emb = contextualized_embeddings[:, header_start:header_end, :]  # (1, num_tokens, hidden)
    mask_token_emb = contextualized_embeddings[:, mask_pos:mask_pos+1, :]  # (1, 1, hidden)
    
    # Mean-pool header tokens and combine with mask token
    if header_tokens_emb.size(1) > 0:
        header_mean = header_tokens_emb.mean(dim=1)  # (1, hidden)
        segment_embedding = (header_mean + mask_token_emb.squeeze(1)) / 2  # Average
    else:
        segment_embedding = mask_token_emb.squeeze(1)
    
    return segment_embedding.cpu().numpy()

def encode_masked_query_navi(query_dict: dict, model: NaviForMaskedLM, device, target_attribute: str):
    """
    Encode masked query using NAVI with mean-pooling approach.
    Create NaviDataset from full query, then mean-pool header and value tokens.
    """
    # Create NaviDataset from query
    dataset = NaviDataset([query_dict])
    data_item = dataset[0]
    
    allowed_keys = ['input_ids', 'attention_mask', 'position_ids', 'segment_ids', 'header_strings']
    inputs = {
        k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in data_item.items()
        if k in allowed_keys
    }
    
    if 'header_strings' in inputs and isinstance(inputs['header_strings'], list):
        if all(isinstance(k, str) for k in inputs['header_strings']):
            inputs['header_strings'] = [inputs['header_strings']]
    
    # Get contextualized embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        contextualized_embeddings = outputs[0]  # (1, seq_len, hidden)
    
    # Extract header and value positions
    header_positions = data_item.get('header_positions')
    value_positions = data_item.get('value_positions')
    header_strings = data_item.get('header_strings')
    
    if not all([header_positions, value_positions, header_strings]):
        # Fallback to CLS
        return contextualized_embeddings[:, 0, :].cpu().numpy()
    
    # Extract segment embeddings using mean-pooling
    H_ctx, V_ctx = extract_header_value_embeddings_from_positions(
        contextualized_embeddings,
        header_positions,
        value_positions
    )
    
    # Find index of target attribute
    try:
        target_idx = header_strings.index(target_attribute)
        # Mean-pool header and value for target segment
        header_emb = H_ctx[0, target_idx, :]  # (hidden,)
        value_emb = V_ctx[0, target_idx, :]   # (hidden,)
        segment_embedding = (header_emb + value_emb) / 2  # Mean-pool
        return segment_embedding.unsqueeze(0).cpu().numpy()  # (1, hidden)
    except ValueError:
        # Target attribute not found, use CLS
        return contextualized_embeddings[:, 0, :].cpu().numpy()

def retrieve_segments(query_embedding, index, metadata, row_to_segments, top_k=5):
    """
    Retrieve segments from FAISS index and aggregate by row_id.
    Returns top-k unique rows.
    """
    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)
    
    # Search
    distances, indices = index.search(query_embedding, top_k * 10)  # Get more to account for duplicates
    
    # Aggregate by row_id
    row_scores = defaultdict(lambda: {'score': float('inf'), 'segments': []})
    
    for i in range(len(indices[0])):
        seg_idx = indices[0][i]
        if seg_idx < 0 or seg_idx >= len(metadata):
            continue
        
        seg_meta = metadata[seg_idx]
        row_id = seg_meta['row_id']
        score = float(distances[0][i])
        
        # Keep best score for each row
        if score < row_scores[row_id]['score']:
            row_scores[row_id]['score'] = score
        row_scores[row_id]['segments'].append({
            'header': seg_meta['header'],
            'value': seg_meta['value'],
            'score': score
        })
    
    # Sort by score and return top-k
    sorted_rows = sorted(row_scores.items(), key=lambda x: x[1]['score'])[:top_k]
    
    return sorted_rows

def load_ground_truth_row(row_id: str, data_pool: list) -> dict:
    """Load ground truth row from data pool."""
    for row in data_pool:
        if str(row.get('row_id', '')) == str(row_id):
            return row
    return {}

def main():
    parser = argparse.ArgumentParser(description='Segment-wise retrieval experiment')
    parser.add_argument('--queries_file', type=str, required=True,
                       help='Path to queries JSONL file')
    parser.add_argument('--product_data', type=str,
                       default='data/cleaned/Product/test/WDC_product_for_cls.jsonl',
                       help='Path to product data')
    parser.add_argument('--movie_data', type=str,
                       default='data/cleaned/Movie/test/WDC_movie_for_cls.jsonl',
                       help='Path to movie data')
    parser.add_argument('--bert_index_dir', type=str, required=True,
                       help='BERT segment index directory')
    parser.add_argument('--navi_index_dir', type=str, required=True,
                       help='NAVI segment index directory')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output results file')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of top results to retrieve')
    parser.add_argument('--model_type', type=str, choices=['bert', 'navi', 'both'], default='both',
                       help='Which model(s) to run')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", file=sys.stderr)
    
    # Load queries
    print(f"Loading queries from {args.queries_file}...", file=sys.stderr)
    queries = []
    with open(args.queries_file, 'r') as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))
    print(f"Loaded {len(queries)} queries", file=sys.stderr)
    
    # Load data pools
    print("Loading data pools...", file=sys.stderr)
    product_data = []
    with open(args.product_data, 'r') as f:
        for line in f:
            if line.strip():
                product_data.append(json.loads(line))
    
    movie_data = []
    with open(args.movie_data, 'r') as f:
        for line in f:
            if line.strip():
                movie_data.append(json.loads(line))
    
    # Create data pool lookup
    data_pools = {
        'product': {str(row.get('row_id', '')): row for row in product_data},
        'movie': {str(row.get('row_id', '')): row for row in movie_data}
    }
    
    results = []
    
    # Load models and indexes - determine domain from first query
    domain = queries[0]['domain'] if queries else 'product'
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.model_type in ['bert', 'both']:
        print("Loading BERT model and index...", file=sys.stderr)
        if domain == 'product':
            bert_model_path = os.path.join(repo_root, 'models', 'bert_product', 'epoch_2')
        else:
            bert_model_path = os.path.join(repo_root, 'models', 'bert_movie', 'epoch_2')
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert_model = BertForMaskedLM.from_pretrained(bert_model_path, local_files_only=True)
        bert_model.to(device)
        bert_model.eval()
        
        bert_index = faiss.read_index(os.path.join(args.bert_index_dir, 'index.faiss'))
        with open(os.path.join(args.bert_index_dir, 'metadata.json'), 'r') as f:
            bert_metadata = json.load(f)
        with open(os.path.join(args.bert_index_dir, 'row_to_segments.json'), 'r') as f:
            bert_row_to_segments = json.load(f)
    
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
    
    if args.model_type in ['navi', 'both']:
        print("Loading NAVI model and index...", file=sys.stderr)
        if domain == 'product':
            navi_base_path = os.path.join(repo_root, 'models', 'navi_product')
        else:
            navi_base_path = os.path.join(repo_root, 'models', 'navi_movie')
        navi_model_path = find_epoch_path(navi_base_path, epoch=2)
        if not navi_model_path:
            raise FileNotFoundError(f"NAVI model not found at {navi_base_path}")
        navi_model = NaviForMaskedLM(model_path=navi_model_path)
        navi_model.to(device)
        navi_model.eval()
        
        navi_index = faiss.read_index(os.path.join(args.navi_index_dir, 'index.faiss'))
        with open(os.path.join(args.navi_index_dir, 'metadata.json'), 'r') as f:
            navi_metadata = json.load(f)
        with open(os.path.join(args.navi_index_dir, 'row_to_segments.json'), 'r') as f:
            navi_row_to_segments = json.load(f)
    
    # Process queries
    print(f"\nProcessing {len(queries)} {domain} queries...", file=sys.stderr)
    with torch.no_grad():
        for query in tqdm(queries, desc="Retrieving"):
            query_id = query['query_id']
            structured_query = query['structured_query']
            target_attribute = query['target_attribute']
            gt_row_id = query['ground_truth_row_id']
            
            # Get ground truth row
            gt_row = data_pools[domain].get(str(gt_row_id), {})
            
            result = {
                'query_id': query_id,
                'domain': domain,
                'query': structured_query,
                'target_attribute': target_attribute,
                'ground_truth_row': gt_row,
                'ground_truth_value': query['ground_truth_value']
            }
            
            # BERT retrieval
            if args.model_type in ['bert', 'both']:
                bert_embedding = encode_masked_query_bert(
                    structured_query, bert_model, bert_tokenizer, device, target_attribute
                )
                bert_retrieved = retrieve_segments(
                    bert_embedding, bert_index, bert_metadata, bert_row_to_segments, args.top_k
                )
                
                # Convert to full rows
                bert_rows = []
                for row_id, row_data in bert_retrieved:
                    # Get full row from metadata (first segment has full_row)
                    for seg_meta in bert_metadata:
                        if seg_meta['row_id'] == row_id:
                            bert_rows.append({
                                'row_id': row_id,
                                'row': seg_meta.get('full_row', {}),
                                'score': row_data['score']
                            })
                            break
                
                result['bert_retrieved_rows'] = bert_rows
            
            # NAVI retrieval
            if args.model_type in ['navi', 'both']:
                navi_embedding = encode_masked_query_navi(
                    structured_query, navi_model, device, target_attribute
                )
                navi_retrieved = retrieve_segments(
                    navi_embedding, navi_index, navi_metadata, navi_row_to_segments, args.top_k
                )
                
                # Convert to full rows
                navi_rows = []
                for row_id, row_data in navi_retrieved:
                    # Get full row from metadata
                    for seg_meta in navi_metadata:
                        if seg_meta['row_id'] == row_id:
                            navi_rows.append({
                                'row_id': row_id,
                                'row': seg_meta.get('full_row', {}),
                                'score': row_data['score']
                            })
                            break
                
                result['navi_retrieved_rows'] = navi_rows
            
            results.append(result)
    
    # Save results
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved results to {output_path}", file=sys.stderr)
    print(f"Processed {len(results)} queries", file=sys.stderr)

if __name__ == '__main__':
    main()
