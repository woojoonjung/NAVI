import argparse
import json
import logging
from pathlib import Path
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import kendalltau
from scipy.spatial import procrustes
from sklearn.metrics.pairwise import cosine_similarity

# Model imports
from transformers import BertForMaskedLM, BertTokenizer, TapasForMaskedLM, TapasConfig, BertConfig
from baselines.haetae.model import HAETAE
from model.navi import NaviForMaskedLM

# Dataset imports
from dataset.dataset import NaviDataset, BertDataset, TapasDataset, HaetaeDataset

# Utils
from experiments.experiment_utils import (
    get_cls_embedding, 
    get_meanpooled_embedding,
    get_meanpooled_segment_embedding, 
    extract_header_value_embeddings_from_positions
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def load_model(model_name, domain, config_suffix=""):
    """Loads a single model with optional config suffix for ablation variants."""
    config = BertConfig.from_pretrained('bert-base-uncased')
    
    # Base model path mapping
    model_path_map = {
        'bert': f'./models/bert_quarter_{domain}/epoch_2',
        'tapas': f'./models/tapas_quarter_{domain}/epoch_2',
        'haetae': f'./models/haetae_quarter_{domain}/epoch_2',
        'navi': f'./models/navi_{domain}/epoch_2'
    }
    
    # Add ablation variants if config_suffix is provided
    if config_suffix:
        # Map domain to the format used in ablation model names
        domain_mapping = {
            'movie': 'movie',
            'product': 'product'
        }
        domain_lower = domain_mapping.get(domain, domain.lower())
        
        # Add ablation variants using the new naming convention from pretrain_navi.sh
        ablation_variants = {
            'woSSI': f'./models/navi_{domain_lower}_woSSI/epoch_2',
            'woMSM': f'./models/navi_{domain_lower}_woMSM/epoch_2',
            'woESA': f'./models/navi_{domain_lower}_woESA/epoch_2',
            'full': f'./models/navi_{domain_lower}/epoch_2'  # Default NAVI model
        }
        model_path_map.update(ablation_variants)
    
    model_path = model_path_map.get(model_name)

    if not model_path or not Path(model_path).exists():
        logging.error(f"Model path not found for {model_name} in domain {domain} with config {config_suffix}: {model_path}")
        return None

    logging.info(f"Loading model {model_name} from {model_path}")
    if model_name == 'bert':
        model = BertForMaskedLM.from_pretrained(model_path, local_files_only=True)
    elif model_name == 'tapas':
        model = TapasForMaskedLM.from_pretrained(model_path, local_files_only=True)
    elif model_name == 'haetae':
        model = HAETAE(config, tokenizer, model_path)
    elif model_name in ['navi', 'woSSI', 'woMSM', 'woESA', 'full']:
        if model_name != 'navi':
            model = NaviForMaskedLM(model_path, ablation_mode=model_name)
        else:
            model = NaviForMaskedLM(model_path)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.to(device)
    model.eval()
    return model

def _get_dataset_for_row(row_json, model_name):
    if model_name in ['navi', 'woSSI', 'woMSM', 'woESA', 'full']:
        return NaviDataset([row_json])
    elif model_name == 'bert':
        return BertDataset([row_json], mode="masked_prediction")
    elif model_name == 'tapas':
        return TapasDataset([row_json], mode="masked_prediction")
    elif model_name == 'haetae':
        return HaetaeDataset(json_data=[row_json])
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def get_primary_row_embedding(row_json: dict, model, model_name: str):
    """
    Computes a row embedding using the primary strategy for the model.
    (Segment-based for NAVI, mean-pooled tokens for baselines).
    """
    dataset = _get_dataset_for_row(row_json, model_name)
    # if model_name in ['navi', 'woSSI', 'woMSM', 'woESA', 'full']:
    #     return get_meanpooled_segment_embedding(dataset, 0, model)
    # else:
    return get_meanpooled_embedding(dataset, 0, model)

def get_cls_row_embedding(row_json: dict, model, model_name: str):
    """Computes a row embedding using the [CLS] token."""
    dataset = _get_dataset_for_row(row_json, model_name)
    return get_cls_embedding(dataset, 0, model)

def get_segment_embeddings(row_json: dict, model: NaviForMaskedLM):
    """Extracts segment embeddings for a NAVI model for a single row."""
    dataset = NaviDataset([row_json])
    data_item = dataset[0]
    allowed_keys = ['input_ids', 'attention_mask', 'position_ids', 'segment_ids', 'header_strings']
    inputs = {k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v) for k, v in data_item.items() if k in allowed_keys}
    if 'header_strings' in inputs and isinstance(inputs['header_strings'], list) and all(isinstance(k, str) for k in inputs['header_strings']):
        inputs['header_strings'] = [inputs['header_strings']]
    with torch.no_grad():
        outputs = model(**inputs)
        contextualized_embeddings = outputs[0]
    header_positions, value_positions, header_strings = data_item.get('header_positions'), data_item.get('value_positions'), data_item.get('header_strings')
    header_strings_for_encoder = [header_strings] if isinstance(header_strings, list) and all(isinstance(k, str) for k in header_strings) else header_strings
    with torch.no_grad():
        E_univ, _ = model.bert.embeddings.header_encoder(header_strings_for_encoder)
        H_ctx, V_ctx = extract_header_value_embeddings_from_positions(contextualized_embeddings, header_positions, value_positions)
        segment_embeddings = model.create_segment_embeddings(E_univ, H_ctx, V_ctx).squeeze(0)
    return {header: emb.cpu().numpy() for header, emb in zip(header_strings, segment_embeddings)}

def kendall_tau_distance(order1, order2):
    if len(order1) != len(order2) or set(order1) != set(order2):
        raise ValueError("Input lists must be permutations of each other.")
    rank_map1 = {val: i for i, val in enumerate(order1)}
    ranks2 = [rank_map1[val] for val in order2]
    tau, _ = kendalltau(list(range(len(order1))), ranks2)
    return (1.0 - tau) / 2.0

def run_structural_consistency_experiment(domain: str, model_name: str, artifacts_dir: Path, data_dir: Path, 
                                        n_permutations: int = 5, n_samples: int = 100, config_suffix: str = ""):
    logging.info(f"Starting Structural Consistency experiment for domain='{domain}', model='{model_name}', config='{config_suffix}'")
    
    # Extract domain name for model loading
    domain_name = domain.split('_')[1].lower()
    model = load_model(model_name, domain_name, config_suffix)
    if model is None: 
        return

    all_rows = []
    domain_path = data_dir / domain
    for file_path in tqdm(domain_path.glob('*.jsonl'), desc="Loading data"):
        table_id = file_path.stem
        if file_path.stat().st_size == 0: continue
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    row = json.loads(line)
                    row['_table_id'], row['_row_idx'] = table_id, i
                    all_rows.append(row)
                except (json.JSONDecodeError, TypeError):
                    continue
    
    sampled_rows = random.sample(all_rows, min(len(all_rows), n_samples))
    logging.info(f"Running experiment on {len(sampled_rows)} sampled rows.")

    results = []
    for row_json in tqdm(sampled_rows, desc=f"Processing rows for {model_name}"):
        table_id, row_idx = row_json.pop('_table_id'), row_json.pop('_row_idx')
        if len(row_json.keys()) < 2: continue

        original_order = list(row_json.keys())
        z_primary = get_primary_row_embedding(row_json, model, model_name)
        z_cls = get_cls_row_embedding(row_json, model, model_name)
        S_orig = get_segment_embeddings(row_json, model) if model_name in ['navi', 'woSSI', 'woMSM', 'woESA', 'full'] else None

        for k in range(n_permutations):
            permuted_order = random.sample(original_order, len(original_order))
            permuted_row = {key: row_json[key] for key in permuted_order}
            k_tau_dist = kendall_tau_distance(original_order, permuted_order)

            # --- Primary Embedding Evaluation ---
            z_tilde_primary = get_primary_row_embedding(permuted_row, model, model_name)
            psi_cos_primary = 1.0 - cosine_similarity(z_primary.reshape(1, -1), z_tilde_primary.reshape(1, -1))[0, 0]
            procrustes_err = None
            if model_name in ['navi', 'woSSI', 'woMSM', 'woESA', 'full']:
                S_perm = get_segment_embeddings(permuted_row, model)
                S_orig_matrix = np.array([S_orig[h] for h in original_order])
                S_perm_reordered_matrix = np.array([S_perm[h] for h in original_order])
                _, _, procrustes_err = procrustes(S_orig_matrix, S_perm_reordered_matrix)
            
            results.append({
                'model': model_name, 'domain': domain, 'table_id': table_id, 'row_idx': row_idx,
                'permutation_k': k, 'embedding_type': 'mean_pooled_segment' if model_name in ['navi', 'woSSI', 'woMSM', 'woESA', 'full'] else 'mean_pooled_token',
                'psi_cos': psi_cos_primary, 'kendall_tau_distance': k_tau_dist,
                'procrustes_error': procrustes_err, 'num_columns': len(original_order),
                'config': config_suffix
            })

            # --- CLS Embedding Evaluation ---
            z_tilde_cls = get_cls_row_embedding(permuted_row, model, model_name)
            psi_cos_cls = 1.0 - cosine_similarity(z_cls.reshape(1, -1), z_tilde_cls.reshape(1, -1))[0, 0]
            results.append({
                'model': model_name, 'domain': domain, 'table_id': table_id, 'row_idx': row_idx,
                'permutation_k': k, 'embedding_type': 'cls',
                'psi_cos': psi_cos_cls, 'kendall_tau_distance': k_tau_dist,
                'procrustes_error': None, 'num_columns': len(original_order),
                'config': config_suffix
            })

    # Convert to DataFrame and compute aggregated results
    results_df = pd.DataFrame(results)
    
    # Map embedding types to the desired naming convention
    embedding_type_mapping = {
        'mean_pooled_segment': 'meanpooled',
        'mean_pooled_token': 'meanpooled',
        'cls': 'cls'
    }
    results_df['embedding_type_mapped'] = results_df['embedding_type'].map(embedding_type_mapping)
    
    # Extract domain name (product or movie)
    domain_short = domain.split('_')[1].lower()  # 'movie' or 'product'
    
    # Compute aggregated results
    aggregated_results = []
    
    # Group by embedding type and compute averages
    for emb_type, group in results_df.groupby('embedding_type_mapped'):
        avg_psi_cos = group['psi_cos'].mean()
        std_psi_cos = group['psi_cos'].std()
        count = len(group)
        
        aggregated_results.append({
            'model': model_name,
            'domain': domain_short,
            'embedding_type': emb_type,
            'avg_psi_cos': avg_psi_cos,
            'std_psi_cos': std_psi_cos,
            'count': count,
            'config': config_suffix
        })
    
    # Save aggregated results
    aggregated_df = pd.DataFrame(aggregated_results)
    config_str = f"_{config_suffix}" if config_suffix else ""
    output_path = artifacts_dir / f'{model_name}_{domain_short}_aggregated{config_str}.csv'
    aggregated_df.to_csv(output_path, index=False)
    logging.info(f"Saved aggregated structural consistency results to {output_path}")
    
    # Also save individual results if needed for debugging
    individual_output_path = artifacts_dir / f'individual_{model_name}_{domain_short}{config_str}.csv'
    results_df.to_csv(individual_output_path, index=False)
    logging.info(f"Saved individual results to {individual_output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run structural consistency experiments.")
    parser.add_argument('--data_dir', type=Path, default='data', help="Directory containing the domain data.")
    parser.add_argument('--artifacts_dir', type=Path, default='artifacts/structvar', help="Directory to save artifacts.")
    parser.add_argument('--domains', type=str, nargs='+', default=['Quarter_Movie_top100_cleaned', 'Quarter_Product_top100_cleaned'], help="List of domains to process.")
    parser.add_argument('--models', type=str, nargs='+', default=['bert', 'tapas', 'haetae', 'navi'], help="Models to run.")
    parser.add_argument('--ablation_models', type=str, nargs='+', default=['woSSI', 'woMSM', 'woESA', 'full'], help="Ablation models to run.")
    parser.add_argument('--config_suffix', type=str, default='hv0p4_align4p0_vr0p5', help="Configuration suffix for ablation models.")
    parser.add_argument('--n_samples', type=int, default=100, help="Number of rows to sample for the experiment.")
    parser.add_argument('--n_permutations', type=int, default=5, help="Number of permutations per row.")
    parser.add_argument('--run_ablation', action='store_true', help="Run ablation experiments with the specified config.")
    parser.add_argument('--save_individual', action='store_true', help="Also save individual row results (for debugging).")
    
    args = parser.parse_args()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    if args.run_ablation:
        # Run ablation experiments only
        for domain in args.domains:
            for model_name in args.ablation_models:
                run_structural_consistency_experiment(domain, model_name, args.artifacts_dir, args.data_dir,
                                                    n_permutations=args.n_permutations, n_samples=args.n_samples,
                                                    config_suffix=args.config_suffix)
    else:
        # Run regular experiments (baselines)
        for domain in args.domains:
            for model_name in args.models:
                run_structural_consistency_experiment(domain, model_name, args.artifacts_dir, args.data_dir,
                                                    n_permutations=args.n_permutations, n_samples=args.n_samples)

if __name__ == '__main__':
    main()