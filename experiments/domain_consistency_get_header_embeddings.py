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

# Model imports
from transformers import BertForMaskedLM, BertTokenizer, TapasForMaskedLM, TapasConfig, BertConfig
from baselines.haetae.model import HAETAE
from model.navi import NaviForMaskedLM

# Dataset imports
from dataset.dataset import NaviDataset, BertDataset, TapasDataset, HaetaeDataset

# Utils
from experiments.experiment_utils import extract_header_value_embeddings_from_positions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def get_model_path(model_name: str, domain: str, ablation_mode: bool = False) -> str:
    """
    Determines the model path based on model name and mode.
    """
    if ablation_mode:
        # For ablation models, use the full model name directly
        return f'./models/{model_name}/epoch_2'
    else:
        # For regular models, use the simple mapping
        model_path_map = {
            'bert': f'./models/bert_quarter_{domain}/epoch_2',
            'tapas': f'./models/tapas_quarter_{domain}/epoch_2',
            'haetae': f'./models/haetae_quarter_{domain}/epoch_2',
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
    if not model_path or not Path(model_path).exists():
        logging.error(f"Model path not found for {model_name} in domain {domain}: {model_path}")
        return None

    base_model_name = get_base_model_name(model_name, ablation_mode)
    
    logging.info(f"Loading model {model_name} (base: {base_model_name}) from {model_path}")
    
    if base_model_name == 'bert':
        model = BertForMaskedLM.from_pretrained(model_path, local_files_only=True)
    elif base_model_name == 'tapas':
        model = TapasForMaskedLM.from_pretrained(model_path, local_files_only=True)
    elif base_model_name == 'haetae':
        model = HAETAE(config, tokenizer, model_path)
    elif base_model_name == 'navi':
        model = NaviForMaskedLM(model_path)
    else:
        raise ValueError(f"Unknown base model name: {base_model_name}")
    
    model.to(device)
    model.eval()
    return model


def get_contextualized_header_embedding(row_json: dict, header: str, model, model_name: str, ablation_mode: bool = False):
    """
    Computes the contextualized embedding for a single header within a single row.
    """
    base_model_name = get_base_model_name(model_name, ablation_mode)
    
    if base_model_name == 'navi':
        dataset = NaviDataset([row_json])
    elif base_model_name == 'bert':
        dataset = BertDataset([row_json], mode="masked_prediction")
    elif base_model_name == 'tapas':
        dataset = TapasDataset([row_json], mode="masked_prediction")
    elif base_model_name == 'haetae':
        dataset = HaetaeDataset(json_data=[row_json])
    else:
        raise ValueError(f"Unknown base model name: {base_model_name}")

    data_item = dataset[0]
    
    allowed_keys = {
        'navi': ['input_ids', 'attention_mask', 'position_ids', 'segment_ids', 'header_strings'],
        'bert': ['input_ids', 'attention_mask'],
        'tapas': ['input_ids', 'attention_mask', 'token_type_ids'],
        'haetae': ['input_ids', 'attention_mask', 'key_positions']
    }[base_model_name]
    
    inputs = {k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else ([v] if isinstance(v, list) and k=='header_strings' else v)) for k, v in data_item.items() if k in allowed_keys}

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
    
    if not header_positions or header not in header_positions:
        return None

    header_embeds, _ = extract_header_value_embeddings_from_positions(
        last_hidden, [header_positions], [{}])

    header_keys = list(header_positions.keys())
    try:
        header_idx = header_keys.index(header)
        return header_embeds[0, header_idx].cpu().numpy()
    except ValueError:
        return None


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
        model = load_model(model_name, domain.split('_')[1].lower(), ablation_mode=False)
    
    if model is None: 
        return

    final_path = artifacts_dir / f"canonical_final_{domain}.json"
    if not final_path.exists():
        logging.error(f"Final canonical file not found at {final_path}. Please run curation first.")
        return
    
    with open(final_path, 'r') as f:
        canonical_map = json.load(f)
    raw_to_canonical = {raw: can for can, raws in canonical_map.items() for raw in raws}
    
    norm_map_path = artifacts_dir / f"normalized_map_{domain}.json"
    with open(norm_map_path, 'r') as f:
        norm_map = json.load(f)

    table_data = {}
    header_index = {}
    domain_path = data_dir / domain
    
    for file_path in tqdm(domain_path.glob('*.jsonl'), desc="Indexing tables"):
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

        for table_id, row_indices in locs_by_table.items():
            sample_indices = random.sample(row_indices, min(len(row_indices), rows_per_header))
            
            for row_idx in sample_indices:
                row_json = table_data[table_id][row_idx]
                embedding = get_contextualized_header_embedding(row_json, header, model, model_name, ablation_mode)
                
                if embedding is not None:
                    # L2 normalize the individual embedding
                    embedding = l2_normalize(embedding.reshape(1, -1)).flatten()
                    
                    results.append({
                        'table_id': table_id,
                        'header': header,
                        'canonical': raw_to_canonical.get(header),
                        'normalized': norm_map.get(header),
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
    parser.add_argument('--domains', type=str, nargs='+', default=['Quarter_Movie_top100_cleaned', 'Quarter_Product_top100_cleaned'], help="List of domains to process.")
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