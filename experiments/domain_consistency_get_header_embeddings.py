import argparse
import json
import logging
from pathlib import Path
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import normalize as l2_normalize
import pyarrow as pa
import pyarrow.parquet as pq
import glob
import os

# Model imports
from transformers import BertForMaskedLM, BertTokenizer, TapasForMaskedLM, TapasConfig, BertConfig, AutoTokenizer
from baselines.haetae.model import HAETAE
from model.navi import NaviForMaskedLM

# Dataset imports
from dataset.dataset import NaviDataset, BertDataset, TapasDataset, HaetaeDataset

# Utils
from experiments.experiment_utils import extract_header_value_embeddings_from_positions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def get_model_path(model_name: str, domain: str, ablation_mode: bool = False) -> str:
    """
    Determines the model path based on model name and mode.
    """
    if ablation_mode:
        # For ablation models, use the full model name directly
        return f'./models/{model_name}/epoch_2'
    else:
        # For regular models, use the simple mapping - fixed to use correct model directory names
        model_path_map = {
            'bert': f'./models/bert_{domain}/epoch_2',
            'tapas': f'./models/tapas_{domain}/epoch_2',
            'haetae': f'./models/haetae_{domain}/epoch_2',
            'navi': f'./models/navi_{domain}/epoch_2'
        }
        return model_path_map.get(model_name)


def get_base_model_name(model_name: str, ablation_mode: bool = False) -> str:
    """
    Determines the base model type for loading the appropriate model class.
    For ablation models, we need to infer the base model from the name.
    """
    if ablation_mode:
        # Infer base model from ablation model name
        # Check for new naming convention from pretrain_navi.sh: navi_{domain}_{ablation}
        if 'navi_' in model_name and ('woESA' in model_name or 'woSSI' in model_name or 'woMSM' in model_name):
            return 'navi'  # All ablation models appear to be NAVI variants
        # Check for old naming convention for backward compatibility
        elif 'woESA' in model_name or 'woSSI' in model_name or 'woMSM' in model_name or 'full_' in model_name:
            return 'navi'  # All ablation models appear to be NAVI variants
        else:
            raise ValueError(f"Cannot determine base model for ablation model: {model_name}")
    else:
        return model_name


def load_model(model_name: str, domain: str, ablation_mode: bool = False):
    """Loads a single model."""
    config = BertConfig.from_pretrained('bert-base-uncased')
    
    model_path = get_model_path(model_name, domain, ablation_mode)
    if not model_path:
        logging.error(f"Unknown model name: {model_name}")
        return None

    base_model_name = get_base_model_name(model_name, ablation_mode)
    
    # For NAVI models, use glob pattern to find the actual epoch directory
    if base_model_name == 'navi':
        base_path = model_path.rsplit('/epoch_2', 1)[0] if '/epoch_2' in model_path else model_path
        epoch_path = find_epoch_path(base_path, epoch=2)
        if not epoch_path:
            logging.error(f"NAVI model epoch_2 not found for {model_name} in domain {domain} at base path: {base_path}")
            return None
        model_path = epoch_path
    elif not Path(model_path).exists():
        logging.error(f"Model path not found for {model_name} in domain {domain}: {model_path}")
        return None
    
    logging.info(f"Loading model {model_name} (base: {base_model_name}) from {model_path}")
    
    if base_model_name == 'bert':
        model = BertForMaskedLM.from_pretrained(model_path, local_files_only=True)
    elif base_model_name == 'tapas':
        model = TapasForMaskedLM.from_pretrained(model_path, local_files_only=True)
    elif base_model_name == 'haetae':
        haetae_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
        model = HAETAE(config, haetae_tokenizer, model_path)
    elif base_model_name == 'navi':
        model = NaviForMaskedLM(model_path)
    else:
        raise ValueError(f"Unknown base model name: {base_model_name}")
    
    model.to(device)
    model.eval()
    return model


def get_contextualized_header_embeddings_for_rows(row_jsons: list[dict], row_metadata: list[tuple], 
                                                   header: str, model, model_name: str, 
                                                   ablation_mode: bool = False):
    """
    Computes contextualized embeddings for a header across multiple rows.
    
    OPTIMIZATION: Creates ONE dataset for all rows instead of one per row.
    This dramatically reduces tokenizer creation overhead (from O(rows) to O(1) per header).
    
    Args:
        row_jsons: List of row JSON dictionaries to process
        row_metadata: List of (table_id, row_idx) tuples corresponding to row_jsons
        header: The header name to extract embeddings for
        model: The model to use for inference
        model_name: Name of the model
        ablation_mode: Whether running in ablation mode
    
    Returns: 
        List of (table_id, row_idx, embedding) tuples (embedding may be None if header not found)
    """
    base_model_name = get_base_model_name(model_name, ablation_mode)
    results = []
    
    if not row_jsons:
        return results
    
    # Create ONE dataset with all rows for this header
    try:
        if base_model_name == 'navi':
            dataset = NaviDataset(row_jsons)
        elif base_model_name == 'bert':
            dataset = BertDataset(row_jsons, mode="masked_prediction")
        elif base_model_name == 'tapas':
            dataset = TapasDataset(row_jsons, mode="masked_prediction")
        elif base_model_name == 'haetae':
            dataset = HaetaeDataset(json_data=row_jsons)
        else:
            raise ValueError(f"Unknown base model name: {base_model_name}")
        
        # Process all rows from the dataset
        for idx, (table_id, row_idx) in enumerate(row_metadata):
            try:
                data_item = dataset[idx]
                
                allowed_keys = {
                    'navi': ['input_ids', 'attention_mask', 'position_ids', 'segment_ids', 'header_strings'],
                    'bert': ['input_ids', 'attention_mask'],
                    'tapas': ['input_ids', 'attention_mask', 'token_type_ids'],
                    'haetae': ['input_ids', 'attention_mask', 'key_positions']
                }[base_model_name]
                
                inputs = {k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else ([v] if isinstance(v, list) and k=='header_strings' else v)) 
                         for k, v in data_item.items() if k in allowed_keys}

                with torch.no_grad():
                    if base_model_name == 'navi':
                        outputs = model(**inputs)
                        last_hidden = outputs[0]
                    elif base_model_name == 'haetae':
                        outputs = model(**inputs)
                        last_hidden = outputs["hidden_states"]
                    else: # BERT, Tapas
                        outputs = model(**inputs, output_hidden_states=True)
                        last_hidden = outputs.hidden_states[-1]
                
                header_positions = data_item.get('header_positions') or data_item.get('key_positions')
                
                embedding = None
                if header_positions and header in header_positions:
                    header_embeds, _ = extract_header_value_embeddings_from_positions(
                        last_hidden, [header_positions], [{}])
                    
                    header_keys = list(header_positions.keys())
                    try:
                        header_idx = header_keys.index(header)
                        embedding = header_embeds[0, header_idx].cpu().numpy()
                    except ValueError:
                        pass
                
                results.append((table_id, row_idx, embedding))
            except Exception as e:
                logging.warning(f"Error processing row {idx} (table_id={table_id}, row_idx={row_idx}): {e}")
                results.append((table_id, row_idx, None))
                continue
    except Exception as e:
        logging.error(f"Error creating dataset for header '{header}': {e}")
        # Fallback: return None for all rows
        for table_id, row_idx in row_metadata:
            results.append((table_id, row_idx, None))
    
    return results


def flatten_nested_canonical_map(nested_map: dict) -> dict[str, str]:
    """
    Flattens a nested dictionary of canonical classes into a flat map
    from raw header to a dot-separated canonical path.
    """
    raw_to_canonical = {}

    def recurse(d, path_prefix):
        for key, value in d.items():
            current_path = f"{path_prefix}.{key}" if path_prefix else key
            if isinstance(value, list):
                for header in value:
                    raw_to_canonical[header] = current_path
            elif isinstance(value, dict):
                recurse(value, current_path)

    recurse(nested_map, '')
    return raw_to_canonical


def get_output_path(artifacts_dir: Path, model_name: str, domain: str, ablation_mode: bool = False) -> Path:
    """
    Determines the correct output path for embeddings based on model type.
    """
    if ablation_mode:
        # For ablation models, embeddings go in artifacts_dir/emb/ablation/model_name/domain/
        return artifacts_dir / 'emb' / 'ablation' / model_name / domain / 'header_embeds.parquet'
    else:
        # For regular models, embeddings go in artifacts_dir/emb/model_name/domain/
        return artifacts_dir / 'emb' / model_name / domain / 'header_embeds.parquet'


def compute_and_save_header_embeddings(domain: str, model_name: str, artifacts_dir: Path, data_dir: Path, rows_per_header: int = 16, ablation_mode: bool = False):
    logging.info(f"Starting header embedding generation for domain='{domain}', model='{model_name}', ablation_mode={ablation_mode}")
    
    # For ablation mode, we don't need to extract domain from model_name since it's already in the model name
    if ablation_mode:
        model = load_model(model_name, domain, ablation_mode=True)
    else:
        # Extract domain name for model loading (e.g., 'cleaned/Movie' -> 'movie')
        domain_name = domain.split('/')[-1].lower()
        model = load_model(model_name, domain_name, ablation_mode=False)
    
    if model is None: 
        return

    # Load the human-curated canonical sets file
    domain_artifact = domain.split('/')[-1].lower()  # 'cleaned/Movie' -> 'movie'
    canonical_path = artifacts_dir / f"canonical_sets_{domain_artifact}.json"
    if not canonical_path.exists():
        logging.error(f"Canonical sets file not found at {canonical_path}. Please ensure the curated canonical sets file exists.")
        return
    
    with open(canonical_path, 'r') as f:
        nested_canonical_map = json.load(f)
    
    # Flatten the nested canonical map to get raw header -> canonical mapping
    raw_to_canonical = flatten_nested_canonical_map(nested_canonical_map)

    table_data = {}
    header_index = {}
    domain_path = data_dir / domain / 'validation'
    
    for file_path in tqdm(domain_path.glob('*.json'), desc="Indexing tables"):
        table_id = file_path.stem
        if file_path.stat().st_size == 0: continue
        
        table_rows = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    table_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        table_data[table_id] = table_rows

        for i, row in enumerate(table_rows):
            for header in row.keys():
                if header not in header_index:
                    header_index[header] = []
                header_index[header].append((table_id, i))
    
    results = []
    unique_headers = sorted(list(header_index.keys()))

    for header in tqdm(unique_headers, desc=f"Computing embeddings for {model_name}"):
        locations = header_index[header]
        
        locs_by_table = {}
        for table_id, row_idx in locations:
            if table_id not in locs_by_table:
                locs_by_table[table_id] = []
            locs_by_table[table_id].append(row_idx)

        # Collect all rows to process together (one dataset per header instead of one per row)
        rows_to_process = []
        row_metadata = []
        
        for table_id, row_indices in locs_by_table.items():
            sample_indices = random.sample(row_indices, min(len(row_indices), rows_per_header))
            
            for row_idx in sample_indices:
                row_json = table_data[table_id][row_idx]
                rows_to_process.append(row_json)
                row_metadata.append((table_id, row_idx))
        
        # Process all rows for this header in one go (creates one dataset instead of many)
        if rows_to_process:
            embedding_results = get_contextualized_header_embeddings_for_rows(
                rows_to_process, row_metadata, header, model, model_name, ablation_mode
            )
            
            # Process results
            for table_id, row_idx, embedding in embedding_results:
                if embedding is not None:
                    # L2 normalize the individual embedding
                    embedding = l2_normalize(embedding.reshape(1, -1)).flatten()
                    
                    results.append({
                        'table_id': table_id,
                        'header': header,
                        'canonical': raw_to_canonical.get(header),
                        'embed': embedding
                    })
    
    output_path = get_output_path(artifacts_dir, model_name, domain, ablation_mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df = pd.DataFrame(results)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)
    logging.info(f"Saved {len(df)} header embeddings to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate contextualized header embeddings.")
    parser.add_argument('--data_dir', type=Path, default='data', help="Directory containing the domain data.")
    parser.add_argument('--artifacts_dir', type=Path, default='artifacts/lexvar', help="Directory to save artifacts.")
    parser.add_argument('--domains', type=str, nargs='+', default=['cleaned/Movie', 'cleaned/Product'], help="List of domains to process.")
    parser.add_argument('--models', type=str, nargs='+', default=['bert', 'tapas', 'haetae', 'navi'], help="Models to run.")
    parser.add_argument('--ablation_mode', action='store_true', help="Enable ablation mode for complex model names.")
    parser.add_argument('--ablation_models', type=str, nargs='+', help="List of ablation model names to run (only used when --ablation_mode is set).")
    
    args = parser.parse_args()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    if args.ablation_mode:
        if not args.ablation_models:
            logging.error("--ablation_models must be specified when --ablation_mode is set")
            return
        
        logging.info(f"Running in ablation mode with models: {args.ablation_models}")
        for domain in args.domains:
            for model_name in args.ablation_models:
                compute_and_save_header_embeddings(domain, model_name, args.artifacts_dir, args.data_dir, ablation_mode=True)
    else:
        logging.info(f"Running in regular mode with models: {args.models}")
        for domain in args.domains:
            for model_name in args.models:
                compute_and_save_header_embeddings(domain, model_name, args.artifacts_dir, args.data_dir, ablation_mode=False)


if __name__ == '__main__':
    main()