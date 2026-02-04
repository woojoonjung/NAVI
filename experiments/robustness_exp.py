#!/usr/bin/env python3
"""
Robustness to Lexical and Structural Perturbations Experiment

Evaluates table representation robustness using:
1. Sensitivity Index (SI) for order and lexical perturbations
2. Downstream performance (classification/imputation) under perturbations
3. Header clustering quality (B³-F1, NMI)
"""

import argparse
import json
import logging
import random
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import glob
import os

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
    extract_header_value_embeddings_from_positions,
    load_data,
    evaluate_masked_prediction,
    b_cubed_score
)
from experiments.schema_perturbations import (
    apply_synonym_replacement,
    apply_header_typos,
    apply_column_reordering,
    identify_low_entropy_headers
)
from experiments.row_classification import (
    run_repeated_classification,
    preprocess_wdc_movie,
    preprocess_wdc_product,
    stratified_sample,
    verify_target_columns_present
)
from experiments.domain_consistency_exp import run_clustering_experiment
from dataset.collator import CollatorForMaskedPrediction
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def find_epoch_path(base_path, epoch):
    """Find epoch directory using glob pattern"""
    pattern = os.path.join(base_path, f"*epoch_{epoch}")
    matches = glob.glob(pattern)
    if matches:
        matches.sort(key=len, reverse=True)
        return matches[0]
    direct_path = os.path.join(base_path, f"epoch_{epoch}")
    if os.path.exists(direct_path):
        return direct_path
    return None


def load_model(model_name, domain, config_suffix=""):
    """Loads a single model with optional config suffix for ablation variants."""
    config = BertConfig.from_pretrained('bert-base-uncased')
    
    model_path_map = {
        'bert': f'./models/bert_{domain}/epoch_2',
        'tapas': f'./models/tapas_{domain}/epoch_2',
        'haetae': f'./models/haetae_{domain}/epoch_2',
        'navi': f'./models/navi_{domain}'
    }
    
    if config_suffix or model_name in ['woSSI', 'woMSM', 'woESA', 'full']:
        domain_mapping = {
            'movie': 'movie',
            'product': 'product'
        }
        domain_lower = domain_mapping.get(domain, domain.lower())
        
        ablation_variants = {
            'woSSI': f'./models/navi_{domain_lower}_woSSI',
            'woMSM': f'./models/navi_{domain_lower}_woMSM',
            'woESA': f'./models/navi_{domain_lower}_woESA',
            'full': f'./models/navi_{domain_lower}'
        }
        model_path_map.update(ablation_variants)
    
    model_path = model_path_map.get(model_name)

    if not model_path:
        logging.error(f"Unknown model name: {model_name}")
        return None

    if model_name in ['navi', 'woSSI', 'woMSM', 'woESA', 'full']:
        epoch_path = find_epoch_path(model_path, epoch=2)
        if not epoch_path:
            logging.error(f"NAVI model epoch_2 not found for {model_name} in domain {domain} at base path: {model_path}")
            return None
        model_path = epoch_path
    elif not Path(model_path).exists():
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
        if model_name != 'navi' and model_name != 'full':
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
    """Computes a row embedding using the primary strategy for the model."""
    dataset = _get_dataset_for_row(row_json, model_name)
    return get_meanpooled_embedding(dataset, 0, model)


def get_cls_row_embedding(row_json: dict, model, model_name: str):
    """Computes a row embedding using the [CLS] token."""
    dataset = _get_dataset_for_row(row_json, model_name)
    return get_cls_embedding(dataset, 0, model)


def apply_synonym_to_single_row(row_json: dict, synonym_map: dict, domain: str, 
                                 low_entropy_headers: set, random_seed: int = 42) -> dict:
    """
    Apply synonym replacement to a single row.
    Simplified version that works on individual rows.
    """
    random.seed(random_seed)
    domain_synonyms = synonym_map.get(domain.lower(), {})
    
    new_row = {}
    for header, value in row_json.items():
        if header.startswith('_'):
            new_row[header] = value
            continue
        
        # Check if header should be replaced (matches low-entropy header)
        should_replace = header in low_entropy_headers
        
        if should_replace:
            # Try to find matching synonym key
            matched_key = None
            for key in domain_synonyms.keys():
                if header == key or header.startswith(key + ".") or header.endswith("." + key):
                    matched_key = key
                    break
            
            if matched_key and domain_synonyms[matched_key]:
                synonym = random.choice(domain_synonyms[matched_key])
                if header == matched_key:
                    new_row[synonym] = value
                elif header.startswith(matched_key + "."):
                    suffix = header[len(matched_key):]
                    new_row[synonym + suffix] = value
                elif header.endswith("." + matched_key):
                    prefix = header[:-len(matched_key) - 1]
                    new_row[prefix + "." + synonym] = value
                else:
                    new_row[header.replace(matched_key, synonym)] = value
            else:
                new_row[header] = value
        else:
            new_row[header] = value
    
    return new_row


def apply_typo_to_single_row(row_json: dict, domain: str, 
                              low_entropy_headers: set, random_seed: int = 42) -> dict:
    """
    Apply header typos to a single row.
    Simplified version that works on individual rows.
    """
    import string
    random.seed(random_seed)
    
    def corrupt_header(header: str, num_chars: int = None) -> str:
        if len(header) == 0:
            return header
        if num_chars is None:
            num_chars = random.randint(1, min(2, len(header)))
        
        header_list = list(header)
        for _ in range(num_chars):
            if len(header_list) == 0:
                break
            corruption_options = ['substitute']
            if len(header_list) < 50:
                corruption_options.append('insert')
            if len(header_list) > 1:
                corruption_options.append('delete')
            
            corruption_type = random.choice(corruption_options)
            if corruption_type == 'substitute':
                pos = random.randint(0, len(header_list) - 1)
                header_list[pos] = random.choice(string.ascii_letters + string.digits)
            elif corruption_type == 'insert':
                pos = random.randint(0, len(header_list))
                header_list.insert(pos, random.choice(string.ascii_letters + string.digits))
            elif corruption_type == 'delete':
                pos = random.randint(0, len(header_list) - 1)
                header_list.pop(pos)
        
        return ''.join(header_list)
    
    new_row = {}
    header_replacements = {}  # Track replacements for consistency
    
    for header, value in row_json.items():
        if header.startswith('_'):
            new_row[header] = value
            continue
        
        should_corrupt = header in low_entropy_headers
        
        if should_corrupt:
            if header not in header_replacements:
                header_replacements[header] = corrupt_header(header)
            new_row[header_replacements[header]] = value
        else:
            new_row[header] = value
    
    return new_row


def compute_lexical_si(domain: str, model_name: str, model, data_dir: Path, 
                       synonym_map: dict, n_samples: int = 100, 
                       embedding_type: str = 'meanpooled', random_seed: int = 42):
    """
    Compute Lexical Sensitivity Index (SI) for synonym and typo perturbations.
    
    SI = E_k[1 - cos(z, z_tilde^(k))] where k=2 (synonym + typo)
    """
    logging.info(f"Computing Lexical SI for {model_name} on {domain} (embedding_type={embedding_type})")
    
    random.seed(random_seed)
    
    # Load validation data
    domain_path = data_dir / domain / 'validation'
    all_rows = []
    for file_path in tqdm(domain_path.glob('*.json'), desc="Loading validation data"):
        table_id = file_path.stem
        if file_path.stat().st_size == 0:
            continue
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    row = json.loads(line)
                    row['_table_id'], row['_row_idx'] = table_id, i
                    all_rows.append(row)
                except (json.JSONDecodeError, TypeError):
                    continue
    
    sampled_rows = random.sample(all_rows, min(len(all_rows), n_samples))
    logging.info(f"Sampled {len(sampled_rows)} rows for lexical SI computation")
    
    # Identify low-entropy headers (need to do this on a sample of data)
    # Use a subset to identify low-entropy headers
    sample_for_entropy = sampled_rows[:min(50, len(sampled_rows))]
    # Remove metadata fields for entropy calculation
    sample_for_entropy_clean = []
    for r in sample_for_entropy:
        clean_r = {k: v for k, v in r.items() if not k.startswith('_')}
        if clean_r:
            sample_for_entropy_clean.append(clean_r)
    
    low_entropy_headers_dict = identify_low_entropy_headers(
        sample_for_entropy_clean,
        domain.split('/')[-1].lower(),
        random_seed
    )
    # Flatten to a single set
    all_low_entropy = set()
    for headers_set in low_entropy_headers_dict.values():
        all_low_entropy.update(headers_set)
    
    # Compute SI for each row
    si_values = []
    for row_json in tqdm(sampled_rows, desc="Computing lexical SI"):
        table_id, row_idx = row_json.pop('_table_id'), row_json.pop('_row_idx')
        if len(row_json.keys()) < 2:
            continue
        
        # Get original embedding
        if embedding_type == 'cls':
            z_orig = get_cls_row_embedding(row_json, model, model_name)
        else:
            z_orig = get_primary_row_embedding(row_json, model, model_name)
        
        # Create synonym variant
        row_synonym = apply_synonym_to_single_row(
            row_json.copy(), synonym_map, domain.split('/')[-1].lower(),
            all_low_entropy, random_seed=random_seed + hash(str(row_json)) % 1000
        )
        if embedding_type == 'cls':
            z_synonym = get_cls_row_embedding(row_synonym, model, model_name)
        else:
            z_synonym = get_primary_row_embedding(row_synonym, model, model_name)
        
        # Create typo variant
        row_typo = apply_typo_to_single_row(
            row_json.copy(), domain.split('/')[-1].lower(),
            all_low_entropy, random_seed=random_seed + hash(str(row_json)) % 1000 + 1000
        )
        if embedding_type == 'cls':
            z_typo = get_cls_row_embedding(row_typo, model, model_name)
        else:
            z_typo = get_primary_row_embedding(row_typo, model, model_name)
        
        # Compute SI for each perturbation
        si_synonym = 1.0 - cosine_similarity(
            z_orig.reshape(1, -1), z_synonym.reshape(1, -1)
        )[0, 0]
        si_typo = 1.0 - cosine_similarity(
            z_orig.reshape(1, -1), z_typo.reshape(1, -1)
        )[0, 0]
        
        # Average SI (k=2)
        si_lexical = (si_synonym + si_typo) / 2.0
        
        si_values.append({
            'table_id': table_id,
            'row_idx': row_idx,
            'si_synonym': si_synonym,
            'si_typo': si_typo,
            'si_lexical': si_lexical
        })
    
    # Aggregate statistics
    si_array = np.array([r['si_lexical'] for r in si_values])
    mean_si = np.mean(si_array)
    std_si = np.std(si_array)
    
    logging.info(f"Lexical SI: mean={mean_si:.4f}, std={std_si:.4f}")
    
    return {
        'mean': mean_si,
        'std': std_si,
        'individual': si_values
    }


def compute_permutation_si(domain: str, model_name: str, model, data_dir: Path,
                           n_permutations: int = 5, n_samples: int = 100,
                           embedding_type: str = 'meanpooled', random_seed: int = 42):
    """
    Compute Permutation Sensitivity Index (SI) for column reordering.
    
    SI = E_k[1 - cos(z, z_tilde^(k))] where k=5 permutations
    """
    logging.info(f"Computing Permutation SI for {model_name} on {domain} (embedding_type={embedding_type})")
    
    random.seed(random_seed)
    
    # Load validation data
    domain_path = data_dir / domain / 'validation'
    all_rows = []
    for file_path in tqdm(domain_path.glob('*.json'), desc="Loading validation data"):
        table_id = file_path.stem
        if file_path.stat().st_size == 0:
            continue
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                try:
                    row = json.loads(line)
                    row['_table_id'], row['_row_idx'] = table_id, i
                    all_rows.append(row)
                except (json.JSONDecodeError, TypeError):
                    continue
    
    sampled_rows = random.sample(all_rows, min(len(all_rows), n_samples))
    logging.info(f"Sampled {len(sampled_rows)} rows for permutation SI computation")
    
    # Compute SI for each row
    si_values = []
    for row_json in tqdm(sampled_rows, desc="Computing permutation SI"):
        table_id, row_idx = row_json.pop('_table_id'), row_json.pop('_row_idx')
        if len(row_json.keys()) < 2:
            continue
        
        original_order = list(row_json.keys())
        
        # Get original embedding
        if embedding_type == 'cls':
            z_orig = get_cls_row_embedding(row_json, model, model_name)
        else:
            z_orig = get_primary_row_embedding(row_json, model, model_name)
        
        # Compute SI for k permutations
        per_permutation_si = []
        for k in range(n_permutations):
            permuted_order = random.sample(original_order, len(original_order))
            permuted_row = {key: row_json[key] for key in permuted_order}
            
            if embedding_type == 'cls':
                z_perm = get_cls_row_embedding(permuted_row, model, model_name)
            else:
                z_perm = get_primary_row_embedding(permuted_row, model, model_name)
            
            si_k = 1.0 - cosine_similarity(
                z_orig.reshape(1, -1), z_perm.reshape(1, -1)
            )[0, 0]
            per_permutation_si.append(si_k)
        
        # Average SI over k permutations
        si_permutation = np.mean(per_permutation_si)
        
        si_values.append({
            'table_id': table_id,
            'row_idx': row_idx,
            'si_permutation': si_permutation,
            'per_permutation': per_permutation_si
        })
    
    # Aggregate statistics
    si_array = np.array([r['si_permutation'] for r in si_values])
    mean_si = np.mean(si_array)
    std_si = np.std(si_array)
    
    logging.info(f"Permutation SI: mean={mean_si:.4f}, std={std_si:.4f}")
    
    return {
        'mean': mean_si,
        'std': std_si,
        'individual': si_values
    }


def apply_synonym_to_dataset(dataset: list, synonym_map: dict, domain: str, random_seed: int = 42) -> list:
    """
    Apply synonym replacement to an entire dataset.
    """
    random.seed(random_seed)
    domain_lower = domain.split('/')[-1].lower()
    
    # Identify low-entropy headers from a sample
    sample_for_entropy = dataset[:min(50, len(dataset))]
    sample_for_entropy_clean = []
    for r in sample_for_entropy:
        clean_r = {k: v for k, v in r.items() if not k.startswith('_')}
        if clean_r:
            sample_for_entropy_clean.append(clean_r)
    
    low_entropy_headers_dict = identify_low_entropy_headers(
        sample_for_entropy_clean,
        domain_lower,
        random_seed
    )
    # Flatten to a single set
    all_low_entropy = set()
    for headers_set in low_entropy_headers_dict.values():
        all_low_entropy.update(headers_set)
    
    # Apply synonym replacement to each row
    perturbed_dataset = []
    for i, row in enumerate(dataset):
        perturbed_row = apply_synonym_to_single_row(
            row.copy(), synonym_map, domain_lower,
            all_low_entropy, random_seed=random_seed + i
        )
        perturbed_dataset.append(perturbed_row)
    
    return perturbed_dataset


def apply_typo_to_dataset(dataset: list, domain: str, random_seed: int = 42) -> list:
    """
    Apply header typos to an entire dataset.
    """
    random.seed(random_seed)
    domain_lower = domain.split('/')[-1].lower()
    
    # Identify low-entropy headers from a sample
    sample_for_entropy = dataset[:min(50, len(dataset))]
    sample_for_entropy_clean = []
    for r in sample_for_entropy:
        clean_r = {k: v for k, v in r.items() if not k.startswith('_')}
        if clean_r:
            sample_for_entropy_clean.append(clean_r)
    
    low_entropy_headers_dict = identify_low_entropy_headers(
        sample_for_entropy_clean,
        domain_lower,
        random_seed
    )
    # Flatten to a single set
    all_low_entropy = set()
    for headers_set in low_entropy_headers_dict.values():
        all_low_entropy.update(headers_set)
    
    # Apply typos to each row
    perturbed_dataset = []
    for i, row in enumerate(dataset):
        perturbed_row = apply_typo_to_single_row(
            row.copy(), domain_lower,
            all_low_entropy, random_seed=random_seed + i
        )
        perturbed_dataset.append(perturbed_row)
    
    return perturbed_dataset


def evaluate_downstream_performance(domain: str, model_name: str, model, data_dir: Path,
                                    synonym_map: dict = None,
                                    perturbation_types: list = ['permuted', 'synonym', 'typo']):
    """
    Evaluate downstream performance (classification and imputation) under perturbations.
    """
    logging.info(f"Evaluating downstream performance for {model_name} on {domain}")
    
    domain_lower = domain.split('/')[-1].lower()
    results = {}
    
    # Classification evaluation
    for pert_type in perturbation_types:
        try:
            # Load classification data
            cls_data_path = data_dir / domain / 'test' / f'WDC_{domain_lower}_for_cls.jsonl'
            if not cls_data_path.exists():
                logging.warning(f"Classification data not found: {cls_data_path}")
                results[f'cls_{pert_type}'] = None
                continue
            
            cls_data = load_data(str(cls_data_path))
            
            # Apply perturbation on-the-fly
            if pert_type == 'permuted':
                cls_data = apply_column_reordering(cls_data, random_seed=42)
            elif pert_type == 'synonym':
                if synonym_map is None:
                    logging.warning("Synonym map not provided, skipping synonym perturbation")
                    results[f'cls_{pert_type}'] = None
                    continue
                cls_data = apply_synonym_to_dataset(cls_data, synonym_map, domain, random_seed=42)
            elif pert_type == 'typo':
                cls_data = apply_typo_to_dataset(cls_data, domain, random_seed=42)
            
            # Preprocess based on domain
            if domain_lower == 'product':
                target_col = 'category'
                cls_data = preprocess_wdc_product(cls_data)
                cls_data = stratified_sample(cls_data, target_col)
                cls_data = verify_target_columns_present(cls_data, [target_col])
            else:  # movie
                target_col = 'genres'
                cls_data = preprocess_wdc_movie(cls_data)
                cls_data = stratified_sample(cls_data, target_col)
                cls_data = verify_target_columns_present(cls_data, [target_col])
            
            # Run classification
            cls_result = run_repeated_classification(
                dataset=cls_data,
                target_col=target_col,
                model=model,
                model_name=model_name,
                domain=domain_lower,
                ml_model="xgboost",
                n_runs=5,
                embedding_type="cls"
            )
            
            results[f'cls_{pert_type}'] = cls_result['mean']
            logging.info(f"Classification ({pert_type}): {results[f'cls_{pert_type}']:.4f}")
            
        except Exception as e:
            logging.error(f"Error in classification evaluation ({pert_type}): {e}")
            results[f'cls_{pert_type}'] = None
    
    # Imputation evaluation
    for pert_type in perturbation_types:
        try:
            # Load masked prediction data
            mp_data_path = data_dir / domain / 'test' / f'WDC_{domain_lower}_for_mp.jsonl'
            if not mp_data_path.exists():
                logging.warning(f"Masked prediction data not found: {mp_data_path}")
                results[f'imp_{pert_type}'] = None
                continue
            
            mp_data = load_data(str(mp_data_path))
            # Sample 1000 rows
            random.seed(42)
            if len(mp_data) > 1000:
                mp_data = random.sample(mp_data, 1000)
            
            # Apply perturbation on-the-fly
            if pert_type == 'permuted':
                mp_data = apply_column_reordering(mp_data, random_seed=42)
            elif pert_type == 'synonym':
                if synonym_map is None:
                    logging.warning("Synonym map not provided, skipping synonym perturbation")
                    results[f'imp_{pert_type}'] = None
                    continue
                mp_data = apply_synonym_to_dataset(mp_data, synonym_map, domain, random_seed=42)
            elif pert_type == 'typo':
                mp_data = apply_typo_to_dataset(mp_data, domain, random_seed=42)
            
            # Create NaviDataset
            json_data = [(i, row) for i, row in enumerate(mp_data)]
            navi_dataset = NaviDataset(json_data)
            
            # Evaluate masked prediction
            collator = CollatorForMaskedPrediction(tokenizer=tokenizer)
            accuracy = evaluate_masked_prediction(
                dataset=navi_dataset,
                model=model,
                tokenizer=tokenizer,
                collator=collator,
                epoch=5
            )
            
            results[f'imp_{pert_type}'] = accuracy
            logging.info(f"Imputation ({pert_type}): {results[f'imp_{pert_type}']:.4f}")
            
        except Exception as e:
            logging.error(f"Error in imputation evaluation ({pert_type}): {e}")
            results[f'imp_{pert_type}'] = None
    
    return results


def evaluate_masked_prediction_task(domain: str, model_name: str, model, data_dir: Path, 
                                     perturbation_type: str = 'clean', synonym_map: dict = None):
    """
    Evaluate masked prediction task: header prediction (epoch=1) and value imputation (epoch=5).
    
    Args:
        domain: Domain name (e.g., 'cleaned/Movie')
        model_name: Model name
        model: Loaded model
        data_dir: Data directory path
        perturbation_type: Type of perturbation ('clean', 'permuted', 'synonym', 'typo')
        synonym_map: Synonym map dictionary (required for synonym perturbation)
    
    Returns:
        dict: Results with header_accuracy and value_accuracy
    """
    logging.info(f"Evaluating masked prediction for {model_name} on {domain} ({perturbation_type})")
    
    domain_lower = domain.split('/')[-1].lower()
    results = {}
    
    # Load masked prediction data
    mp_data_path = data_dir / domain / 'test' / f'WDC_{domain_lower}_for_mp.jsonl'
    if not mp_data_path.exists():
        logging.warning(f"Masked prediction data not found: {mp_data_path}")
        return {'header_accuracy': None, 'value_accuracy': None}
    
    mp_data = load_data(str(mp_data_path))
    # Sample 1000 rows
    random.seed(42)
    if len(mp_data) > 1000:
        mp_data = random.sample(mp_data, 1000)
    
    # Apply perturbation if not clean
    if perturbation_type == 'permuted':
        mp_data = apply_column_reordering(mp_data, random_seed=42)
    elif perturbation_type == 'synonym':
        if synonym_map is None:
            logging.warning("Synonym map not provided, skipping synonym perturbation")
            return {'header_accuracy': None, 'value_accuracy': None}
        mp_data = apply_synonym_to_dataset(mp_data, synonym_map, domain, random_seed=42)
    elif perturbation_type == 'typo':
        mp_data = apply_typo_to_dataset(mp_data, domain, random_seed=42)
    # If perturbation_type == 'clean', use data as-is
    
    # Create appropriate dataset based on model type
    if model_name in ['navi', 'woSSI', 'woMSM', 'woESA', 'full']:
        json_data = [(i, row) for i, row in enumerate(mp_data)]
        dataset = NaviDataset(json_data)
        model_tokenizer = tokenizer
    elif model_name == 'bert':
        dataset = BertDataset(mp_data, mode="masked_prediction")
        model_tokenizer = tokenizer
    elif model_name == 'tapas':
        dataset = TapasDataset(mp_data, mode="masked_prediction")
        model_tokenizer = tokenizer
    elif model_name == 'haetae':
        dataset = HaetaeDataset(json_data=mp_data)
        model_tokenizer = tokenizer
    else:
        logging.error(f"Unknown model type: {model_name}")
        return {'header_accuracy': None, 'value_accuracy': None}
    
    # Create collator
    collator = CollatorForMaskedPrediction(tokenizer=model_tokenizer)
    
    # Evaluate header prediction (epoch=1)
    try:
        logging.info(f"Evaluating header prediction (epoch=1) for {model_name} ({perturbation_type})...")
        header_accuracy = evaluate_masked_prediction(
            dataset=dataset,
            model=model,
            tokenizer=model_tokenizer,
            collator=collator,
            epoch=1
        )
        results['header_accuracy'] = header_accuracy
        logging.info(f"Header prediction accuracy ({perturbation_type}): {header_accuracy:.4f}")
    except Exception as e:
        logging.error(f"Error in header prediction evaluation ({perturbation_type}): {e}")
        results['header_accuracy'] = None
    
    # Evaluate value imputation (epoch=5)
    try:
        logging.info(f"Evaluating value imputation (epoch=5) for {model_name} ({perturbation_type})...")
        value_accuracy = evaluate_masked_prediction(
            dataset=dataset,
            model=model,
            tokenizer=model_tokenizer,
            collator=collator,
            epoch=5
        )
        results['value_accuracy'] = value_accuracy
        logging.info(f"Value imputation accuracy ({perturbation_type}): {value_accuracy:.4f}")
    except Exception as e:
        logging.error(f"Error in value imputation evaluation ({perturbation_type}): {e}")
        results['value_accuracy'] = None
    
    return results


def evaluate_header_clustering(domain: str, model_name: str, artifacts_dir: Path,
                               ablation_mode: bool = False):
    """
    Evaluate header clustering quality (B³-F1, NMI).
    """
    logging.info(f"Evaluating header clustering for {model_name} on {domain}")
    
    try:
        # Use the clustering function from domain_consistency_exp
        # We need to extract results from the CSV file it generates
        domain_artifact = domain.split('/')[-1].lower()
        clean_domain = domain.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        if ablation_mode:
            clustering_csv = artifacts_dir / f'results_clustering_ablation_{model_name}_{clean_domain}.csv'
        else:
            clustering_csv = artifacts_dir / f'results_clustering_{model_name}_{clean_domain}.csv'
        
        # Run clustering experiment if results don't exist
        if not clustering_csv.exists():
            logging.info(f"Running clustering experiment for {model_name} on {domain}")
            run_clustering_experiment(domain, model_name, artifacts_dir, ablation_mode)
        
        # Read results
        if clustering_csv.exists():
            df = pd.read_csv(clustering_csv)
            if len(df) > 0:
                return {
                    'b3_f1': df.iloc[0]['B3_F1'],
                    'nmi': df.iloc[0]['NMI']
                }
        
        logging.warning(f"Clustering results not found for {model_name} on {domain}")
        return {'b3_f1': None, 'nmi': None}
        
    except Exception as e:
        logging.error(f"Error in header clustering evaluation: {e}")
        return {'b3_f1': None, 'nmi': None}


def generate_results_table(all_results: list, output_dir: Path):
    """
    Generate results table (CSV + LaTeX) with masked prediction, SI, and downstream performance.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for table
    table_data = []
    for result in all_results:
        def format_value(val, default='N/A'):
            if val is None:
                return default
            try:
                return f"{float(val):.4f}"
            except (ValueError, TypeError):
                return default
        
        row = {
            'Model': result['model'],
            'Domain': result['domain'],
            'Header (Def)': format_value(result.get('header_accuracy_def')),
            'Value (Def)': format_value(result.get('value_accuracy_def')),
            'Header (Perm)': format_value(result.get('header_accuracy_permuted')),
            'Value (Perm)': format_value(result.get('value_accuracy_permuted')),
            'Header (Syn)': format_value(result.get('header_accuracy_synonym')),
            'Value (Syn)': format_value(result.get('value_accuracy_synonym')),
            'Header (Typo)': format_value(result.get('header_accuracy_typo')),
            'Value (Typo)': format_value(result.get('value_accuracy_typo')),
            'Permutation SI': format_value(result['permutation_si']['mean'] if result.get('permutation_si') else None),
            'Lexical SI': format_value(result['lexical_si']['mean'] if result.get('lexical_si') else None),
            'Cls (Perm)': format_value(result.get('downstream', {}).get('cls_permuted')),
            'Imp (Perm)': format_value(result.get('downstream', {}).get('imp_permuted')),
            'Cls (Synonym)': format_value(result.get('downstream', {}).get('cls_synonym')),
            'Imp (Synonym)': format_value(result.get('downstream', {}).get('imp_synonym')),
            'Cls (Typo)': format_value(result.get('downstream', {}).get('cls_typo')),
            'Imp (Typo)': format_value(result.get('downstream', {}).get('imp_typo')),
        }
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    
    # Save CSV
    csv_path = output_dir / 'masked_prediction_results.csv'
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved results table to {csv_path}")
    
    # Generate LaTeX table
    latex_path = output_dir / 'masked_prediction_results.tex'
    with open(latex_path, 'w') as f:
        f.write("% Requires \\usepackage{booktabs} in LaTeX document\n")
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{l" + "c" * (len(df.columns) - 1) + "}\n")
        f.write("\\toprule\n")
        # Escape underscores in column names for LaTeX
        columns_escaped = [col.replace('_', '\\_') for col in df.columns]
        f.write(" & ".join(columns_escaped) + " \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in df.iterrows():
            f.write(" & ".join(str(val) for val in row.values) + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Masked Prediction Results: Header Prediction and Value Imputation (Def=Clean, Perm=Permuted, Syn=Synonym)}\n")
        f.write("\\label{tab:masked_prediction}\n")
        f.write("\\end{table}\n")
    
    logging.info(f"Saved LaTeX table to {latex_path}")
    
    # Also save as text file with LaTeX table
    txt_path = output_dir / 'masked_prediction_results.txt'
    with open(txt_path, 'w') as f:
        f.write("Masked Prediction Results: Header Prediction and Value Imputation\n")
        f.write("(Def=Clean/Default, Perm=Permuted, Syn=Synonym)\n")
        f.write("=" * 80 + "\n\n")
        f.write("LaTeX Table:\n\n")
        with open(latex_path, 'r') as latex_file:
            f.write(latex_file.read())
        f.write("\n\nCSV Data:\n\n")
        df.to_string(f, index=False)
    logging.info(f"Saved text file with LaTeX table to {txt_path}")


def generate_clustering_table(all_results: list, output_dir: Path):
    """
    Generate header clustering table (CSV + LaTeX, half-width format).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for table (only include unique model/domain combinations)
    table_data = []
    seen_combinations = set()
    for result in all_results:
        combination = (result['model'], result['domain'])
        if result.get('clustering') and combination not in seen_combinations:
            seen_combinations.add(combination)
            def format_value(val, default='N/A'):
                if val is None:
                    return default
                try:
                    return f"{float(val):.4f}"
                except (ValueError, TypeError):
                    return default
            
            row = {
                'Model': result['model'],
                'Domain': result['domain'],
                'B³-F1': format_value(result['clustering'].get('b3_f1')),
                'NMI': format_value(result['clustering'].get('nmi')),
            }
            table_data.append(row)
    
    if not table_data:
        logging.warning("No clustering results to generate table")
        return
    
    df = pd.DataFrame(table_data)
    
    # Save CSV
    csv_path = output_dir / 'header_clustering.csv'
    df.to_csv(csv_path, index=False)
    logging.info(f"Saved clustering table to {csv_path}")
    
    # Generate LaTeX table (half-width)
    latex_path = output_dir / 'header_clustering.tex'
    with open(latex_path, 'w') as f:
        f.write("% Requires \\usepackage{booktabs,graphicx} in LaTeX document\n")
        f.write("\\begin{table}\n")
        f.write("\\centering\n")
        f.write("\\resizebox{0.5\\textwidth}{!}{%\n")  # Half-width
        f.write("\\begin{tabular}{l" + "c" * (len(df.columns) - 1) + "}\n")
        f.write("\\toprule\n")
        # Escape underscores in column names for LaTeX
        columns_escaped = [col.replace('_', '\\_') for col in df.columns]
        f.write(" & ".join(columns_escaped) + " \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in df.iterrows():
            f.write(" & ".join(str(val) for val in row.values) + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}%\n")
        f.write("\\caption{Header Clustering Quality}\n")
        f.write("\\label{tab:header_clustering}\n")
        f.write("\\end{table}\n")
    
    logging.info(f"Saved LaTeX clustering table to {latex_path}")
    
    # Also save as text file with LaTeX table
    txt_path = output_dir / 'header_clustering.txt'
    with open(txt_path, 'w') as f:
        f.write("Header Clustering Quality\n")
        f.write("=" * 80 + "\n\n")
        f.write("LaTeX Table:\n\n")
        with open(latex_path, 'r') as latex_file:
            f.write(latex_file.read())
        f.write("\n\nCSV Data:\n\n")
        df.to_string(f, index=False)
    logging.info(f"Saved text file with LaTeX clustering table to {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Robustness to Lexical and Structural Perturbations Experiment")
    parser.add_argument('--data_dir', type=Path, default='data', help="Directory containing the domain data.")
    parser.add_argument('--artifacts_dir', type=Path, default='artifacts', help="Directory for artifacts.")
    parser.add_argument('--output_dir', type=Path, default='experiments/robustness_results', help="Output directory for results.")
    parser.add_argument('--domains', type=str, nargs='+', default=['cleaned/Movie', 'cleaned/Product'], help="List of domains to process.")
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['bert', 'tapas', 'haetae', 'navi', 'woSSI', 'woMSM', 'woESA'],
                        help="Models to run. Options: bert, tapas, haetae, navi, woSSI, woMSM, woESA")
    parser.add_argument('--n_samples', type=int, default=100, help="Number of rows to sample for SI computation.")
    parser.add_argument('--n_permutations', type=int, default=5, help="Number of permutations per row.")
    parser.add_argument('--synonym_map', type=str, default='artifacts/schema_noise/synonym_map.json', help="Path to synonym map JSON file.")
    parser.add_argument('--embedding_types', type=str, nargs='+', default=['meanpooled'], 
                        choices=['meanpooled', 'cls'], help="Embedding types for SI computation (can specify multiple).")
    parser.add_argument('--skip_clustering', action='store_true', default=True, 
                        help="Skip header clustering evaluation.")
    parser.add_argument('--skip_downstream', action='store_true', default=True, 
                        help="Skip downstream performance evaluation (classification).")
    parser.add_argument('--skip_si', action='store_true', default=True,
                        help="Skip Sensitivity Index (SI) computation.")
    parser.add_argument('--run_masked_prediction', action='store_true', default=True,
                        help="Run masked prediction evaluation (header + value imputation).")
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate embedding types
    valid_embedding_types = ['meanpooled', 'cls']
    embedding_types = [et for et in args.embedding_types if et in valid_embedding_types]
    if not embedding_types:
        logging.error("No valid embedding types specified. Using default: meanpooled")
        embedding_types = ['meanpooled']
    
    # Load synonym map
    with open(args.synonym_map, 'r') as f:
        synonym_map = json.load(f)
    
    all_results = []
    
    # Run experiments for each model and domain
    for domain in args.domains:
        domain_name = domain.split('/')[-1].lower()
        
        for model_name in args.models:
            logging.info(f"\n{'='*80}")
            logging.info(f"Processing: {model_name} on {domain}")
            logging.info(f"{'='*80}\n")
            
            # Load model once per model/domain combination
            model = load_model(model_name, domain_name)
            if model is None:
                logging.error(f"Failed to load model {model_name} for domain {domain_name}, skipping...")
                continue
            
            result = {
                'model': model_name,
                'domain': domain_name,
                'header_accuracy_def': None,
                'value_accuracy_def': None,
                'header_accuracy_permuted': None,
                'value_accuracy_permuted': None,
                'header_accuracy_synonym': None,
                'value_accuracy_synonym': None,
                'header_accuracy_typo': None,
                'value_accuracy_typo': None,
                'permutation_si': None,
                'lexical_si': None,
                'downstream': {},
                'clustering': None
            }
            
            # Evaluate masked prediction (header + value imputation) on clean data (Def)
            if args.run_masked_prediction:
                try:
                    # Clean data (Def)
                    mp_results_clean = evaluate_masked_prediction_task(
                        domain, model_name, model, args.data_dir, 
                        perturbation_type='clean', synonym_map=synonym_map
                    )
                    result['header_accuracy_def'] = mp_results_clean.get('header_accuracy')
                    result['value_accuracy_def'] = mp_results_clean.get('value_accuracy')
                    
                    # Permuted data
                    mp_results_permuted = evaluate_masked_prediction_task(
                        domain, model_name, model, args.data_dir, 
                        perturbation_type='permuted', synonym_map=synonym_map
                    )
                    result['header_accuracy_permuted'] = mp_results_permuted.get('header_accuracy')
                    result['value_accuracy_permuted'] = mp_results_permuted.get('value_accuracy')
                    
                    # Synonym data
                    mp_results_synonym = evaluate_masked_prediction_task(
                        domain, model_name, model, args.data_dir, 
                        perturbation_type='synonym', synonym_map=synonym_map
                    )
                    result['header_accuracy_synonym'] = mp_results_synonym.get('header_accuracy')
                    result['value_accuracy_synonym'] = mp_results_synonym.get('value_accuracy')
                    
                    # Typo data
                    mp_results_typo = evaluate_masked_prediction_task(
                        domain, model_name, model, args.data_dir, 
                        perturbation_type='typo', synonym_map=synonym_map
                    )
                    result['header_accuracy_typo'] = mp_results_typo.get('header_accuracy')
                    result['value_accuracy_typo'] = mp_results_typo.get('value_accuracy')
                except Exception as e:
                    logging.error(f"Error evaluating masked prediction: {e}")
            
            # Evaluate downstream performance (classification) - skipped by default
            if not args.skip_downstream:
                try:
                    downstream_results = evaluate_downstream_performance(
                        domain, model_name, model, args.data_dir,
                        synonym_map=synonym_map,
                        perturbation_types=['permuted', 'synonym', 'typo']
                    )
                    result['downstream'] = downstream_results
                except Exception as e:
                    logging.error(f"Error evaluating downstream performance: {e}")
            
            # Evaluate header clustering - skipped by default
            if not args.skip_clustering:
                try:
                    clustering_result = evaluate_header_clustering(
                        domain, model_name, args.artifacts_dir,
                        ablation_mode=(model_name in ['woSSI', 'woMSM', 'woESA', 'full'])
                    )
                    result['clustering'] = clustering_result
                except Exception as e:
                    logging.error(f"Error evaluating header clustering: {e}")
            
            # Compute SI - skipped by default
            if not args.skip_si:
                # Use first embedding type for SI computation
                embedding_type = embedding_types[0] if embedding_types else 'meanpooled'
                logging.info(f"\n{'='*80}")
                logging.info(f"Computing SI for {model_name} on {domain} with {embedding_type} embeddings")
                logging.info(f"{'='*80}\n")
                
                # Compute Permutation SI
                try:
                    result['permutation_si'] = compute_permutation_si(
                        domain, model_name, model, args.data_dir,
                        n_permutations=args.n_permutations,
                        n_samples=args.n_samples,
                        embedding_type=embedding_type
                    )
                except Exception as e:
                    logging.error(f"Error computing permutation SI: {e}")
                
                # Compute Lexical SI
                try:
                    result['lexical_si'] = compute_lexical_si(
                        domain, model_name, model, args.data_dir,
                        synonym_map, n_samples=args.n_samples,
                        embedding_type=embedding_type
                    )
                except Exception as e:
                    logging.error(f"Error computing lexical SI: {e}")
            
            all_results.append(result)
            
            # Clean up model from memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Generate tables
    logging.info("\nGenerating results tables...")
    generate_results_table(all_results, args.output_dir)
    generate_clustering_table(all_results, args.output_dir)
    
    # Save full results as JSON
    json_path = args.output_dir / 'full_results.json'
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(all_results)
    with open(json_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    logging.info(f"Saved full results to {json_path}")


if __name__ == '__main__':
    main()
