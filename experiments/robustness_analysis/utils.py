"""
Utility functions for robustness analysis experiments.
Includes functions from schema_perturbations and structural_consistency experiments.
"""

import json
import random
import string
import glob
import os
from typing import List, Dict, Set
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import kendalltau
from scipy.spatial import procrustes
from sklearn.metrics.pairwise import cosine_similarity

from transformers import BertForMaskedLM, BertTokenizer, TapasForMaskedLM, BertConfig
from baselines.haetae.model import HAETAE
from model.navi import NaviForMaskedLM
from dataset.dataset import NaviDataset, BertDataset, TapasDataset, HaetaeDataset, FieldEntropyAnalyzer


# ========== Functions from schema_perturbations.py ==========

def identify_low_entropy_headers(data: List[Dict], domain: str, random_seed: int = 42) -> Dict[int, Set[str]]:
    """Identify low-entropy headers for each table in the dataset."""
    random.seed(random_seed)
    
    dataset_data = []
    table_id_map = {}
    
    has_table_id = any('_table_id' in row for row in data[:10]) if data else False
    
    if has_table_id:
        for idx, row in enumerate(data):
            table_id = row.get('_table_id', 0)
            dataset_data.append((table_id, row))
            table_id_map[idx] = table_id
    else:
        for idx, row in enumerate(data):
            table_id = idx // 100
            dataset_data.append((table_id, row))
            table_id_map[idx] = table_id
    
    dataset = NaviDataset(dataset_data, compute_field_entropy=True)
    field_categories = dataset.get_field_categories()
    
    low_entropy_headers = {}
    all_low_entropy = set()
    
    if field_categories:
        for table_id, categories in field_categories.items():
            table_low = categories.get('low_entropy', set())
            low_entropy_headers[table_id] = table_low
            all_low_entropy.update(table_low)
        
        if 0 not in low_entropy_headers:
            low_entropy_headers[0] = all_low_entropy
    
    return low_entropy_headers


def apply_synonym_replacement(data: List[Dict], synonym_map: Dict[str, List[str]], 
                             domain: str, sample_ratio: float = 0.5, random_seed: int = 42) -> List[Dict]:
    """Apply synonym replacement to sampled low-entropy headers."""
    random.seed(random_seed)
    
    low_entropy_headers = identify_low_entropy_headers(data, domain, random_seed)
    domain_synonyms = synonym_map.get(domain.lower(), {})
    
    headers_to_replace = {}
    for table_id, low_headers in low_entropy_headers.items():
        headers_list = list(low_headers)
        num_to_replace = max(1, int(len(headers_list) * sample_ratio))
        sampled_headers = random.sample(headers_list, min(num_to_replace, len(headers_list)))
        headers_to_replace[table_id] = set(sampled_headers)
    
    perturbed_data = []
    for idx, row in enumerate(data):
        if '_table_id' in row:
            table_id = row['_table_id']
        else:
            table_id = idx // 100
        
        new_row = {}
        
        for header, value in row.items():
            if header.startswith('_'):
                new_row[header] = value
                continue
            
            should_replace = False
            matched_low_entropy_header = None
            
            headers_to_check = set()
            if table_id in headers_to_replace:
                headers_to_check.update(headers_to_replace[table_id])
            if 0 in headers_to_replace:
                headers_to_check.update(headers_to_replace[0])
            
            for low_entropy_header in headers_to_check:
                if header == low_entropy_header:
                    should_replace = True
                    matched_low_entropy_header = low_entropy_header
                    break
                elif header.startswith(low_entropy_header + "."):
                    should_replace = True
                    matched_low_entropy_header = low_entropy_header
                    break
                elif header.endswith("." + low_entropy_header):
                    should_replace = True
                    matched_low_entropy_header = low_entropy_header
                    break
            
            if should_replace:
                matched_key = None
                for key in domain_synonyms.keys():
                    if header.startswith(key + ".") or header == key:
                        matched_key = key
                        break
                    elif header.endswith("." + key) or "." + key + "." in header:
                        matched_key = key
                        break
                
                if matched_key and domain_synonyms[matched_key]:
                    if header == matched_key:
                        synonym = random.choice(domain_synonyms[matched_key])
                        new_row[synonym] = value
                    elif header.startswith(matched_key + "."):
                        suffix = header[len(matched_key):]
                        synonym = random.choice(domain_synonyms[matched_key])
                        new_row[synonym + suffix] = value
                    elif header.endswith("." + matched_key):
                        prefix = header[:-len(matched_key) - 1]
                        synonym = random.choice(domain_synonyms[matched_key])
                        new_row[prefix + "." + synonym] = value
                    else:
                        synonym = random.choice(domain_synonyms[matched_key])
                        new_row[header.replace(matched_key, synonym)] = value
                else:
                    new_row[header] = value
            else:
                new_row[header] = value
        
        perturbed_data.append(new_row)
    
    return perturbed_data


def corrupt_header(header: str, num_chars: int = None) -> str:
    """Corrupt a header by randomly modifying 1-2 characters."""
    if len(header) == 0:
        return header
    
    if num_chars is None:
        num_chars = random.randint(1, min(2, len(header)))
    
    header_list = list(header)
    
    if len(header_list) == 0:
        return header
    
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


def apply_header_typos(data: List[Dict], domain: str, sample_ratio: float = 0.5, random_seed: int = 42) -> List[Dict]:
    """Apply typos to sampled low-entropy headers."""
    random.seed(random_seed)
    
    low_entropy_headers = identify_low_entropy_headers(data, domain, random_seed)
    
    headers_to_corrupt = {}
    for table_id, low_headers in low_entropy_headers.items():
        headers_list = list(low_headers)
        num_to_corrupt = max(1, int(len(headers_list) * sample_ratio))
        sampled_headers = random.sample(headers_list, min(num_to_corrupt, len(headers_list)))
        headers_to_corrupt[table_id] = set(sampled_headers)
    
    header_replacements = {}
    
    perturbed_data = []
    for idx, row in enumerate(data):
        if '_table_id' in row:
            table_id = row['_table_id']
        else:
            table_id = idx // 100
        
        new_row = {}
        
        for header, value in row.items():
            if header.startswith('_'):
                new_row[header] = value
                continue
            
            should_corrupt = False
            matched_low_entropy_header = None
            
            headers_to_check = set()
            if table_id in headers_to_corrupt:
                headers_to_check.update(headers_to_corrupt[table_id])
            if 0 in headers_to_corrupt:
                headers_to_check.update(headers_to_corrupt[0])
            
            for low_entropy_header in headers_to_check:
                if header == low_entropy_header:
                    should_corrupt = True
                    matched_low_entropy_header = low_entropy_header
                    break
                elif header.startswith(low_entropy_header + "."):
                    should_corrupt = True
                    matched_low_entropy_header = low_entropy_header
                    break
                elif header.endswith("." + low_entropy_header):
                    should_corrupt = True
                    matched_low_entropy_header = low_entropy_header
                    break
            
            if should_corrupt:
                key = (table_id, header)
                if key not in header_replacements:
                    header_replacements[key] = corrupt_header(header)
                new_row[header_replacements[key]] = value
            else:
                new_row[header] = value
        
        perturbed_data.append(new_row)
    
    return perturbed_data


def apply_column_reordering(data: List[Dict], random_seed: int = 42) -> List[Dict]:
    """Apply column reordering by randomly shuffling column order for each row."""
    random.seed(random_seed)
    
    perturbed_data = []
    for row in data:
        metadata = {k: v for k, v in row.items() if k.startswith('_')}
        data_fields = {k: v for k, v in row.items() if not k.startswith('_')}
        
        items = list(data_fields.items())
        random.shuffle(items)
        
        new_row = {**metadata, **dict(items)}
        perturbed_data.append(new_row)
    
    return perturbed_data


# ========== Functions from structural_consistency_exp.py ==========

def find_epoch_path(base_path, epoch):
    """Find epoch directory using glob pattern."""
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
    try:
        from config import config as app_config
        _bert_name = app_config.get_bert_name()
        _bert_kw = {"local_files_only": app_config.use_local_files_only()}
        _checkpoint_epoch = getattr(app_config, "CHECKPOINT_EPOCH", 4)
    except ImportError:
        _bert_name = "bert-base-uncased"
        _bert_kw = {}
        _checkpoint_epoch = 4
    bconfig = BertConfig.from_pretrained(_bert_name, **_bert_kw)
    
    model_path_map = {
        'bert': f'./models/bert_{domain}/epoch_{_checkpoint_epoch}',
        'tapas': f'./models/tapas_{domain}/epoch_{_checkpoint_epoch}',
        'haetae': f'./models/haetae_{domain}/epoch_{_checkpoint_epoch}',
        'navi': f'./models/navi_{domain}'
    }
    
    if config_suffix or model_name in ['woSSI', 'woMSM', 'woESA', 'woGHA', 'woGHC', 'full']:
        domain_mapping = {
            'movie': 'movie',
            'product': 'product'
        }
        domain_lower = domain_mapping.get(domain, domain.lower())
        
        ablation_variants = {
            'woSSI': f'./models/navi_{domain_lower}_woSSI',
            'woMSM': f'./models/navi_{domain_lower}_woMSM',
            'woESA': f'./models/navi_{domain_lower}_woESA',
            'woGHA': f'./models/navi_{domain_lower}_woGHA',
            'woGHC': f'./models/navi_{domain_lower}_woGHC',
            'full': f'./models/navi_{domain_lower}'
        }
        model_path_map.update(ablation_variants)
    
    model_path = model_path_map.get(model_name)

    if not model_path:
        return None

    if model_name in ['navi', 'woSSI', 'woMSM', 'woESA', 'woGHA', 'woGHC', 'full']:
        epoch_path = find_epoch_path(model_path, epoch=_checkpoint_epoch)
        if not epoch_path:
            return None
        model_path = epoch_path
    elif not Path(model_path).exists():
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == 'bert':
        model = BertForMaskedLM.from_pretrained(model_path, local_files_only=True)
    elif model_name == 'tapas':
        model = TapasForMaskedLM.from_pretrained(model_path, local_files_only=True)
    elif model_name == 'haetae':
        tokenizer = BertTokenizer.from_pretrained(_bert_name, **_bert_kw)
        model = HAETAE(bconfig, tokenizer, model_path)
    elif model_name in ['navi', 'woSSI', 'woMSM', 'woESA', 'woGHA', 'woGHC', 'full']:
        # Model ablation_mode only for woGHA/woGHC; woSSI/woMSM/woESA are dataset/training-only
        model_ablation = model_name if model_name in ('woGHA', 'woGHC') else 'full'
        model = NaviForMaskedLM(model_path, ablation_mode=model_ablation)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.to(device)
    model.eval()
    return model


# ========== Functions from experiment_utils.py needed by robustness ==========

def load_data(path, path_is="jsonl"):
    """Load data from JSONL or CSV file."""
    data = []

    if path_is == "jsonl":
        if path is not None:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))

    return data


def get_cls_embedding(dataset, idx, model):
    """Get CLS token embedding from model."""
    if hasattr(dataset, '__getitem__'):
        if hasattr(dataset[idx], 'copy'):
            data_item = dataset[idx].copy()
        else:
            data_item = dict(dataset[idx])
    else:
        data_item = dataset[idx].copy()
    
    if isinstance(model, NaviForMaskedLM):
        allowed_keys = ['input_ids', 'attention_mask', 'position_ids', 'segment_ids', 'header_strings']
        if 'header_positions' in data_item and isinstance(data_item['header_positions'], dict):
            data_item['header_positions'] = [data_item['header_positions']]
        if 'header_strings' in data_item and isinstance(data_item['header_strings'], list):
            if all(isinstance(k, str) for k in data_item['header_strings']):
                data_item['header_strings'] = [data_item['header_strings']]
    elif isinstance(model, HAETAE):
        allowed_keys = ['input_ids', 'attention_mask', 'key_positions']
        if 'key_positions' in data_item and isinstance(data_item['key_positions'], dict):
            data_item['key_positions'] = [data_item['key_positions']]
    elif isinstance(model, TapasForMaskedLM):
        allowed_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        if 'token_type_ids' in data_item and isinstance(data_item['token_type_ids'], dict):
            data_item['token_type_ids'] = [data_item['token_type_ids']]
    else:
        allowed_keys = ['input_ids', 'attention_mask']
        
    device = next(model.parameters()).device

    inputs = {
        k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in data_item.items()
        if k in allowed_keys
    }

    with torch.no_grad():
        if isinstance(model, NaviForMaskedLM):
            outputs = model(**inputs)
            cls_embedding = outputs[0][:, 0, :].squeeze().to("cpu").numpy()
        elif isinstance(model, HAETAE):
            outputs = model(**inputs)
            cls_embedding = outputs["hidden_states"][:, 0, :].squeeze().to("cpu").numpy()
        elif isinstance(model, BertForMaskedLM):
            outputs = model(**inputs, output_hidden_states=True)
            cls_embedding = outputs.hidden_states[-1][:, 0, :].squeeze().to("cpu").numpy()
        elif isinstance(model, TapasForMaskedLM):
            outputs = model(**inputs, output_hidden_states=True)
            cls_embedding = outputs.hidden_states[-1][:, 0, :].squeeze().to("cpu").numpy()
    return cls_embedding


def get_meanpooled_embedding(dataset, idx, model):
    """Get mean-pooled embedding from model."""
    if hasattr(dataset, '__getitem__'):
        if hasattr(dataset[idx], 'copy'):
            data_item = dataset[idx].copy()
        else:
            data_item = dict(dataset[idx])
    else:
        data_item = dataset[idx].copy()
    
    if isinstance(model, NaviForMaskedLM):
        allowed_keys = ['input_ids', 'attention_mask', 'position_ids', 'segment_ids', 'header_strings']
        if 'header_positions' in data_item and isinstance(data_item['header_positions'], dict):
            data_item['header_positions'] = [data_item['header_positions']]
        if 'header_strings' in data_item and isinstance(data_item['header_strings'], list):
            if all(isinstance(k, str) for k in data_item['header_strings']):
                data_item['header_strings'] = [data_item['header_strings']]
    elif isinstance(model, HAETAE):
        allowed_keys = ['input_ids', 'attention_mask', 'key_positions']
        if 'key_positions' in data_item and isinstance(data_item['key_positions'], dict):
            data_item['key_positions'] = [data_item['key_positions']]
    elif isinstance(model, TapasForMaskedLM):
        allowed_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        if 'token_type_ids' in data_item and isinstance(data_item['token_type_ids'], dict):
            data_item['token_type_ids'] = [data_item['token_type_ids']]
    else:
        allowed_keys = ['input_ids', 'attention_mask']
        
    device = next(model.parameters()).device

    inputs = {
        k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in data_item.items()
        if k in allowed_keys
    }

    with torch.no_grad():
        if isinstance(model, NaviForMaskedLM):
            outputs = model(**inputs)
            last_hidden = outputs[0]
        elif isinstance(model, HAETAE):
            outputs = model(**inputs)
            last_hidden = outputs["hidden_states"]
        elif isinstance(model, BertForMaskedLM):
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
        elif isinstance(model, TapasForMaskedLM):
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
    attention_mask = inputs['attention_mask'].unsqueeze(-1)
    masked_hidden = last_hidden * attention_mask
    sum_hidden = masked_hidden.sum(dim=1)
    valid_token_counts = attention_mask.sum(dim=1)
    mean_pooled = (sum_hidden / valid_token_counts).squeeze().to("cpu").numpy()

    return mean_pooled


def get_meanpooled_segment_embedding(dataset, idx, model):
    """Extracts mean-pooled segment embeddings for Navi models."""
    if not isinstance(model, NaviForMaskedLM):
        raise TypeError("This function is only for NaviForMaskedLM models.")

    if hasattr(dataset, '__getitem__'):
        data_item = dict(dataset[idx])
    else:
        data_item = dataset[idx].copy()

    allowed_keys = ['input_ids', 'attention_mask', 'position_ids', 'segment_ids', 'header_strings']
    device = next(model.parameters()).device
    inputs = {
        k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in data_item.items()
        if k in allowed_keys
    }
    if 'header_strings' in inputs and isinstance(inputs['header_strings'], list):
         if all(isinstance(k, str) for k in inputs['header_strings']):
                inputs['header_strings'] = [inputs['header_strings']]

    with torch.no_grad():
        outputs = model(**inputs)
        contextualized_embeddings = outputs[0]

    header_positions = data_item.get('header_positions')
    value_positions = data_item.get('value_positions')
    header_strings = data_item.get('header_strings')

    if not all([header_positions, value_positions, header_strings]):
        raise ValueError("Missing header_positions, value_positions, or header_strings for segment embedding.")
    
    if isinstance(header_strings, list) and all(isinstance(k, str) for k in header_strings):
        header_strings_for_encoder = [header_strings]
    else:
        header_strings_for_encoder = header_strings
    
    with torch.no_grad():
        E_univ, _ = model.bert.embeddings.header_encoder(header_strings_for_encoder)
        H_ctx, V_ctx = extract_header_value_embeddings_from_positions(
            contextualized_embeddings,
            header_positions,
            value_positions
        )
        segment_embeddings = model.create_segment_embeddings(E_univ, H_ctx, V_ctx)
        
    mean_pooled = segment_embeddings.mean(dim=1).squeeze().cpu().numpy()
    return mean_pooled


def extract_header_value_embeddings_from_positions(embeddings, header_positions, value_positions):
    """Extracts header and value embeddings from contextualized embeddings."""
    hidden_size = embeddings.size(-1)
    embeddings = embeddings.squeeze(0)
    
    header_pos_dict = header_positions[0] if isinstance(header_positions, list) else header_positions
    value_pos_dict = value_positions[0] if isinstance(value_positions, list) else value_positions

    header_keys = list(header_pos_dict.keys())
    num_headers = len(header_keys)
    
    header_embeds = torch.zeros((1, num_headers, hidden_size), device=embeddings.device)
    val_embeds = torch.zeros((1, num_headers, hidden_size), device=embeddings.device)
    
    for i, header_name in enumerate(header_keys):
        h_token_indices = header_pos_dict.get(header_name, [])
        if h_token_indices:
            header_tokens = embeddings[h_token_indices]
            header_embeds[0, i] = header_tokens.mean(dim=0)
            
        v_token_indices = value_pos_dict.get(header_name, [])
        if v_token_indices:
            value_tokens = embeddings[v_token_indices]
            val_embeds[0, i] = value_tokens.mean(dim=0)
            
    return header_embeds, val_embeds


def evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch):
    """Evaluate masked prediction accuracy."""
    from experiments.masked_prediction.mp_utils import evaluate_masked_prediction as _evaluate_masked_prediction
    return _evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch)


def b_cubed_score(y_true, y_pred):
    """Calculate B-cubed precision, recall, and F1 score."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    
    precision_sum = 0
    recall_sum = 0
    total_items = len(y_true)
    
    for i in range(total_items):
        true_cluster = y_true[i]
        pred_cluster = y_pred[i]
        
        true_cluster_size = np.sum(y_true == true_cluster)
        pred_cluster_size = np.sum(y_pred == pred_cluster)
        intersection = np.sum((y_true == true_cluster) & (y_pred == pred_cluster))
        
        if pred_cluster_size > 0:
            precision_sum += intersection / pred_cluster_size
        
        if true_cluster_size > 0:
            recall_sum += intersection / true_cluster_size
    
    precision = precision_sum / total_items
    recall = recall_sum / total_items
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }
