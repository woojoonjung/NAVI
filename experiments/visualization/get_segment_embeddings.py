#!/usr/bin/env python3
"""
Extract segment embeddings from Navi model for entropy-aware visualization.

This script follows the exact data preprocessing procedure from row_clustering_genre.py
and extracts segment embeddings from NaviForMaskedLM models.
"""

import torch
import numpy as np
import json
import random
import os
import argparse
import gc
import glob
from datetime import datetime
from collections import Counter
from pathlib import Path
import pandas as pd

# Utility imports from existing experiment scripts
from experiments.visualization.utils import load_data

# Model imports
from transformers import BertTokenizer, BertConfig
from model.navi import NaviForMaskedLM
from config import config as app_config

_checkpoint_epoch = getattr(app_config, "CHECKPOINT_EPOCH", 4)

# Dataset imports
from dataset.dataset import NaviDataset, FieldEntropyAnalyzer

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Re-initialize tokenizer (local or online per RUN_ONLINE)
_bert_name = app_config.get_bert_name()
_kw = {"local_files_only": app_config.use_local_files_only()}
config = BertConfig.from_pretrained(_bert_name, **_kw)
tokenizer = BertTokenizer.from_pretrained(_bert_name, **_kw)


def extract_segment_embeddings_from_navi(row_json, model, device):
    """
    Extract segment embeddings for a NAVI model for a single row.
    
    Args:
        row_json: Dictionary containing the row data
        model: NaviForMaskedLM model
        device: Device to run on
        
    Returns:
        dict: Mapping from header to segment embedding (numpy array)
    """
    dataset = NaviDataset([row_json])
    data_item = dataset[0]
    
    allowed_keys = ['input_ids', 'attention_mask', 'position_ids', 'segment_ids', 'header_strings']
    inputs = {
        k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v) 
        for k, v in data_item.items() 
        if k in allowed_keys
    }
    
    if 'header_strings' in inputs and isinstance(inputs['header_strings'], list) and all(isinstance(k, str) for k in inputs['header_strings']):
        inputs['header_strings'] = [inputs['header_strings']]
    
    with torch.no_grad():
        outputs = model(**inputs)
        contextualized_embeddings = outputs[0]
    
    header_positions = data_item.get('header_positions')
    value_positions = data_item.get('value_positions')
    header_strings = data_item.get('header_strings')
    
    if not all([header_positions, value_positions, header_strings]):
        raise ValueError("Missing header_positions, value_positions, or header_strings for segment embedding.")
    
    header_strings_for_encoder = [header_strings] if isinstance(header_strings, list) and all(isinstance(k, str) for k in header_strings) else header_strings
    
    with torch.no_grad():
        E_univ, _ = model.bert.embeddings.header_encoder(header_strings_for_encoder)
        H_ctx, V_ctx = extract_header_value_embeddings_from_positions(contextualized_embeddings, header_positions, value_positions)
        segment_embeddings = model.create_segment_embeddings(E_univ, H_ctx, V_ctx).squeeze(0)
    
    return {header: emb.cpu().numpy() for header, emb in zip(header_strings, segment_embeddings)}


def extract_header_value_embeddings_from_positions(embeddings, header_positions, value_positions):
    """
    Extract header and value embeddings from contextualized embeddings using positions.
    
    Args:
        embeddings: Contextualized embeddings (1, seq_len, hidden_size)
        header_positions: Dict mapping header names to token positions
        value_positions: Dict mapping header names to value token positions
        
    Returns:
        tuple: (header_embeddings, value_embeddings) both (1, num_headers, hidden_size)
    """
    device = embeddings.device
    hidden_size = embeddings.size(-1)
    
    # Get headers in consistent order
    headers = list(header_positions.keys())
    num_headers = len(headers)
    
    header_embeds = []
    val_embeds = []
    
    for header in headers:
        # Extract header embedding (mean pool over header tokens)
        header_pos = header_positions[header]
        if header_pos:
            header_tokens = embeddings[0, header_pos, :]  # (num_tokens, hidden_size)
            header_emb = header_tokens.mean(dim=0, keepdim=True)  # (1, hidden_size)
        else:
            header_emb = torch.zeros(1, hidden_size, device=device)
        header_embeds.append(header_emb)
        
        # Extract value embedding (mean pool over value tokens)
        val_pos = value_positions[header]
        if val_pos:
            val_tokens = embeddings[0, val_pos, :]  # (num_tokens, hidden_size)
            val_emb = val_tokens.mean(dim=0, keepdim=True)  # (1, hidden_size)
        else:
            val_emb = torch.zeros(1, hidden_size, device=device)
        val_embeds.append(val_emb)
    
    # Stack to (1, num_headers, hidden_size)
    header_embeds = torch.stack(header_embeds, dim=1)  # (1, num_headers, hidden_size)
    val_embeds = torch.stack(val_embeds, dim=1)  # (1, num_headers, hidden_size)
    
    return header_embeds, val_embeds


def extract_universal_header_embeddings(header_strings, model):
    """
    Extract universal header embeddings for anchor visualization.
    
    Args:
        header_strings: List of unique header strings
        model: NaviForMaskedLM model
        
    Returns:
        dict: Mapping from header to universal embedding (numpy array)
    """
    with torch.no_grad():
        E_univ, _ = model.bert.embeddings.header_encoder([header_strings])
        E_univ = E_univ.squeeze(0)  # (num_headers, hidden_size)
    
    return {header: emb.cpu().numpy() for header, emb in zip(header_strings, E_univ)}


def load_tables_from_validation_dir(validation_dir, n_tables=6, rows_per_table=200, random_state=42):
    """
    Load and sample tables from validation directory.
    Each JSON file in the directory represents one table.
    
    Args:
        validation_dir: Path to validation directory containing JSON files
        n_tables: Number of tables to sample
        rows_per_table: Number of rows per table
        random_state: Random seed for reproducibility
        
    Returns:
        dict: Mapping from table_id (filename stem) to list of sampled rows
    """
    
    random.seed(random_state)
    np.random.seed(random_state)
    
    validation_path = Path(validation_dir)
    if not validation_path.exists():
        raise ValueError(f"Validation directory not found: {validation_dir}")
    
    # Get all JSON files in the directory
    json_files = sorted(list(validation_path.glob('*.json')))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {validation_dir}")
    
    print(f"Found {len(json_files)} tables in {validation_dir}")
    
    # Load all tables with their row counts
    table_data = {}
    for json_file in json_files:
        table_id = json_file.stem  # Use filename without extension as table_id
        rows = []
        
        if json_file.stat().st_size == 0:
            print(f"  Skipping empty file: {table_id}")
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        rows.append(row)
                    except json.JSONDecodeError:
                        continue
            
            if rows:
                table_data[table_id] = rows
                print(f"  Table {table_id}: {len(rows)} rows")
        except Exception as e:
            print(f"  Error loading {table_id}: {e}")
            continue
    
    if not table_data:
        raise ValueError(f"No valid tables found in {validation_dir}")
    
    print(f"Successfully loaded {len(table_data)} tables")
    
    # Filter tables with enough rows
    valid_tables = {tid: rows for tid, rows in table_data.items() if len(rows) >= rows_per_table}
    
    if len(valid_tables) < n_tables:
        print(f"Warning: Only {len(valid_tables)} tables have >= {rows_per_table} rows.")
        if len(valid_tables) == 0:
            print("No tables have enough rows. Using all available data with reduced rows_per_table.")
            # Use all tables but reduce rows_per_table
            valid_tables = table_data
            rows_per_table = min(rows_per_table, min(len(rows) for rows in table_data.values()) if table_data else 0)
            print(f"Adjusted rows_per_table to {rows_per_table}")
        else:
            print("Using all available valid tables.")
            n_tables = len(valid_tables)
    
    # Sample tables
    selected_table_ids = random.sample(list(valid_tables.keys()), min(n_tables, len(valid_tables)))
    
    # Sample rows from each selected table
    sampled_data = {}
    total_rows = 0
    for table_id in selected_table_ids:
        table_rows = valid_tables[table_id]
        # Sample up to rows_per_table, but use all if fewer available
        sample_size = min(rows_per_table, len(table_rows))
        sampled_rows = random.sample(table_rows, sample_size)
        sampled_data[table_id] = sampled_rows
        total_rows += len(sampled_rows)
        print(f"Selected table {table_id}: sampled {len(sampled_rows)} rows")
    
    print(f"Total sampled rows: {total_rows}")
    return sampled_data


def compute_entropy_categorization(sampled_data):
    """
    Compute entropy categorization for each table using FieldEntropyAnalyzer.
    Uses 10th and 90th percentiles for low/high entropy categorization.
    
    Args:
        sampled_data: Dict mapping table_id to list of rows
        
    Returns:
        dict: Mapping from table_id to entropy categories
    """
    print("Computing entropy categorization...")
    
    # Create dataset with entropy analysis using percentile method (10th and 90th percentiles)
    all_rows = []
    table_mapping = {}
    
    for table_id, rows in sampled_data.items():
        for i, row in enumerate(rows):
            all_rows.append((table_id, row))
            table_mapping[len(all_rows) - 1] = table_id
    
    # Create NaviDataset with entropy analysis using 10th and 90th percentiles
    dataset = NaviDataset(
        all_rows, 
        compute_field_entropy=True,
        entropy_threshold_method="percentile",
        low_threshold_percentile=10,
        high_threshold_percentile=90
    )
    
    # Get field categories
    field_categories = dataset.get_field_categories()
    
    return field_categories


def extract_segments_with_entropy(sampled_data, model_path, field_categories, device):
    """
    Extract segment embeddings with entropy categorization.
    
    Args:
        sampled_data: Dict mapping table_id to list of rows
        model_path: Path to Navi model
        field_categories: Entropy categorization results
        device: Device to run on
        
    Returns:
        list: List of segment data dictionaries
    """
    print("Extracting segment embeddings...")
    
    # Load model
    model = NaviForMaskedLM(model_path)
    model = model.to(device)
    model.eval()
    
    segments = []
    all_headers = set()
    
    for table_id, rows in sampled_data.items():
        print(f"Processing table {table_id}...")
        
        # Get entropy categories for this table
        table_categories = field_categories.get(table_id, {})
        low_entropy_fields = table_categories.get('low_entropy', set())
        high_entropy_fields = table_categories.get('high_entropy', set())
        field_entropy = table_categories.get('field_entropy', {})
        
        for row_idx, row in enumerate(rows):
            try:
                # Process row as-is (no target column removal needed)
                processed_row = row.copy()
                
                # Extract segment embeddings
                segment_embeddings = extract_segment_embeddings_from_navi(processed_row, model, device)
                
                # Create segment data
                for header, embedding in segment_embeddings.items():
                    all_headers.add(header)
                    
                    # Determine entropy category
                    entropy_value = field_entropy.get(header, 0.0)
                    if header in low_entropy_fields:
                        entropy_group = 'low'
                    elif header in high_entropy_fields:
                        entropy_group = 'high'
                    else:
                        entropy_group = 'mid'
                    
                    segment_data = {
                        'table_id': table_id,
                        'row_id': row_idx,
                        'header': header,
                        'entropy': entropy_value,
                        'entropy_group': entropy_group,
                        'segment_embedding': embedding
                    }
                    segments.append(segment_data)
                    
            except Exception as e:
                print(f"Error processing row {row_idx} in table {table_id}: {e}")
                continue
    
    print(f"Extracted {len(segments)} segments from {len(all_headers)} unique headers")
    return segments, list(all_headers)


def save_segments_data(segments, all_headers, output_path):
    """
    Save segment data to files.
    
    Args:
        segments: List of segment data dictionaries
        all_headers: List of all unique headers
        output_path: Path to save data
    """
    print(f"Saving segment data to {output_path}...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save segments as JSON (with embeddings as lists)
    segments_json = []
    for seg in segments:
        seg_copy = seg.copy()
        seg_copy['segment_embedding'] = seg_copy['segment_embedding'].tolist()
        segments_json.append(seg_copy)
    
    with open(os.path.join(output_path, 'segments.json'), 'w') as f:
        json.dump(segments_json, f, indent=2)
    
    # Save metadata
    metadata = {
        'num_segments': len(segments),
        'num_headers': len(all_headers),
        'headers': all_headers,
        'entropy_groups': {
            'low': len([s for s in segments if s['entropy_group'] == 'low']),
            'mid': len([s for s in segments if s['entropy_group'] != 'low' and s['entropy_group'] != 'high']),
            'high': len([s for s in segments if s['entropy_group'] == 'high'])
        }
    }
    
    with open(os.path.join(output_path, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved {len(segments)} segments to {output_path}")


def find_epoch_path(base_path, epoch):
    """Find epoch directory using glob pattern"""
    pattern = os.path.join(base_path, f"*epoch_{epoch}")
    matches = glob.glob(pattern)
    if matches:
        # Prefer the most specific match (longest path)
        matches.sort(key=len, reverse=True)
        return matches[0]
    # Fallback to direct path
    direct_path = os.path.join(base_path, f"epoch_{epoch}")
    if os.path.exists(direct_path):
        return direct_path
    return None


def main():
    parser = argparse.ArgumentParser(description='Extract segment embeddings from Navi model')
    parser.add_argument('--model_path', type=str, required=True,
                       help=f'Path to Navi model (base directory or full path to epoch_{getattr(app_config, "CHECKPOINT_EPOCH", 4)})')
    parser.add_argument('--domain', type=str, default='cleaned/Movie',
                       help='Domain path (e.g., cleaned/Movie or cleaned/Product)')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Base data directory')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output directory for segment data')
    parser.add_argument('--n_tables', type=int, default=5,
                       help='Number of tables to sample')
    parser.add_argument('--rows_per_table', type=int, default=100,
                       help='Number of rows per table')
    parser.add_argument('--random_state', type=int, default=24,
                       help='Random seed for reproducibility')
    parser.add_argument('--ablation_mode', action='store_true', default=False,
                       help='Enable ablation mode for ablation model paths')
    
    args = parser.parse_args()
    
    # Find the actual checkpoint epoch directory
    # Handle both cases: base directory or full path to epoch_N
    ep = _checkpoint_epoch
    if os.path.isdir(args.model_path):
        if args.model_path.endswith(f'epoch_{ep}') or os.path.basename(args.model_path) == f'epoch_{ep}':
            if os.path.exists(os.path.join(args.model_path, 'model.safetensors')):
                actual_model_path = args.model_path
            else:
                raise ValueError(f"Model path {args.model_path} appears to be epoch_{ep} but model.safetensors not found")
        else:
            epoch_path = find_epoch_path(args.model_path, epoch=ep)
            if epoch_path and os.path.exists(os.path.join(epoch_path, 'model.safetensors')):
                actual_model_path = epoch_path
                print(f"Found epoch_{ep} directory: {actual_model_path}")
            else:
                direct_epoch_path = os.path.join(args.model_path, f'epoch_{ep}')
                if os.path.exists(direct_epoch_path) and os.path.exists(os.path.join(direct_epoch_path, 'model.safetensors')):
                    actual_model_path = direct_epoch_path
                else:
                    raise ValueError(f"Could not find epoch_{ep} directory with model.safetensors in {args.model_path}")
    elif os.path.isfile(args.model_path):
        # If it's a file, use the parent directory
        actual_model_path = os.path.dirname(args.model_path)
    else:
        raise ValueError(f"Model path not found: {args.model_path}")
    
    # Verify model files exist
    model_file = os.path.join(actual_model_path, 'model.safetensors')
    if not os.path.exists(model_file):
        raise ValueError(f"Model file not found at {model_file}")
    
    print(f"Extracting segment embeddings from {actual_model_path}")
    print(f"Domain: {args.domain}")
    print(f"Ablation mode: {args.ablation_mode}")
    print(f"Sampling {args.n_tables} tables with {args.rows_per_table} rows each")
    
    # Load tables from validation directory
    from pathlib import Path
    validation_dir = Path(args.data_dir) / args.domain / 'validation'
    
    if not validation_dir.exists():
        raise ValueError(f"Validation directory not found: {validation_dir}")
    
    print(f"Loading tables from {validation_dir}...")
    sampled_data = load_tables_from_validation_dir(
        validation_dir,
        n_tables=args.n_tables,
        rows_per_table=args.rows_per_table,
        random_state=args.random_state
    )
    
    # Compute entropy categorization (using 10th and 90th percentiles)
    print("Computing entropy categorization...")
    field_categories = compute_entropy_categorization(sampled_data)
    
    # Extract segment embeddings
    segments, all_headers = extract_segments_with_entropy(
        sampled_data, 
        actual_model_path, 
        field_categories, 
        device
    )
    
    # Adjust output path for ablation mode
    if args.ablation_mode:
        # Create ablation subdirectory if needed
        output_path = args.output_path
        if 'ablation' not in output_path:
            # Extract model name from path for better organization
            base_output = os.path.dirname(output_path) if os.path.dirname(output_path) else output_path
            output_name = os.path.basename(output_path)
            output_path = os.path.join(base_output, 'ablation', output_name)
    else:
        output_path = args.output_path
    
    # Save data
    save_segments_data(segments, all_headers, output_path)
    
    print("✅ Segment extraction complete!")


if __name__ == "__main__":
    main()
