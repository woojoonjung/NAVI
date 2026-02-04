import torch
import numpy as np
import json
import random
import pandas as pd
import os
import argparse
import gc
import glob
from datetime import datetime
from experiments.experiment_utils import (
    load_data, 
    run_row_classification,
    run_row_classification_tabpfn,
    extract_raw_features,
    get_cls_embedding,
    get_meanpooled_embedding
)
from skrub import TableVectorizer, TextEncoder

# Model imports
from transformers import BertForMaskedLM, BertTokenizer, TapasForMaskedLM
from baselines.haetae.model import HAETAE
from model.navi import NaviForMaskedLM

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from dataset.dataset import NaviDataset, BertDataset, TapasDataset
from baselines.haetae.dataset import JSONDataset
from collections import Counter, defaultdict

from transformers import BertTokenizer, BertConfig

config = BertConfig.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def _failed_result(err: Exception):
    """Standard shape for a failed evaluation so JSON/summary don't crash."""
    return {"mean": None, "std": None, "scores": []}

def _warn_failed(phase: str, err: Exception):
    print(f"⚠️  Skipping {phase} due to error: {err}")


def verify_target_columns_present(data, target_columns):
    """
    Verify that target columns are present in the data.
    Data is already cleaned in preprocess.py with token budget considered.
    CLS data is filtered to only include rows with valid target columns.
    """
    processed_data = []
    missing_targets = 0
    
    for row_dict in data:
        # Verify all target columns are present
        all_present = all(target_col in row_dict for target_col in target_columns)
        
        if not all_present:
            missing_targets += 1
            # Skip rows missing target columns (shouldn't happen if CLS data is properly filtered)
            continue
        
        processed_data.append(row_dict)
        
        if len(processed_data) % 1000 == 0:
            print(f"   Verified {len(processed_data)} instances...")
    
    if missing_targets > 0:
        print(f"⚠️  Warning: {missing_targets} instances missing target columns (skipped)")
    
    print(f"✅ Verification complete. {len(processed_data)} instances with target columns.")
    return processed_data


def remove_target_column(row, target_col):
    """
    Remove the target column from the row.
    """
    return {k: v for k, v in row.items() if k != target_col}  

def run_cls_classification(dataset, target_col, model, model_name, domain, ml_model="xgboost", embedding_type="cls"):
    """
    Run CLS-based row classification using the specified ML model.
    Evaluates the model's ability to encode discriminative row-level semantics.
    """
    # Extract CLS embeddings and labels
    embeddings = []
    labels = []

    preprocessed_dataset = [remove_target_column(row, target_col) for row in dataset]

    if isinstance(model, NaviForMaskedLM):
        dataset_X = NaviDataset(preprocessed_dataset)
    elif isinstance(model, HAETAE):
        dataset_X = JSONDataset(path=None, tokenizer=tokenizer)
        dataset_X.data = preprocessed_dataset
    elif isinstance(model, BertForMaskedLM):
        dataset_X = BertDataset(preprocessed_dataset)
    elif isinstance(model, TapasForMaskedLM):
        dataset_X = TapasDataset(preprocessed_dataset)
    else:
        print("Unknown Model")
        return None

    for i, row in enumerate(dataset):

        if embedding_type == 'cls':
            emb = get_cls_embedding(dataset_X, i, model)
        elif embedding_type == 'mean':
            emb = get_meanpooled_embedding(dataset_X, i, model)
        
        embeddings.append(emb)

        labels.append(row[target_col])

    X = np.stack(embeddings)
    y = np.array(labels)

    # Run classification
    if ml_model == "tabpfn":
        f1 = run_row_classification_tabpfn(X, y, test_size=0.2)
    else:
        f1 = run_row_classification(X, y, model_type=ml_model)
    return f1

def run_repeated_classification(dataset, target_col, model, model_name, domain, ml_model="xgboost", n_runs=8, embedding_type="cls"):
    """
    Run classification multiple times and return mean ± std.
    """
    f1_scores = []
    
    print(f"Running {n_runs} iterations for {model_name} on {domain} domain using {ml_model.upper()}...")
    
    for run in range(n_runs):
        print(f"  Run {run + 1}/{n_runs}...", end=" ")
        
        # Use different random seeds for each run
        random.seed(42 + run)
        np.random.seed(42 + run)
        
        f1 = run_cls_classification(dataset, target_col, model, model_name, domain, ml_model, embedding_type)
        f1_scores.append(f1)
        print(f"F1: {f1:.4f}")
    
    # Calculate statistics
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    print(f"\n  Results: {mean_f1:.4f} ± {std_f1:.4f}\n\n")
    
    return {
        'mean': mean_f1,
        'std': std_f1,
        'scores': f1_scores
    }

def run_repeated_classification_raw(X, y, model_type="xgboost", n_runs=8):
    """
    Run classification on raw features multiple times and return mean ± std.
    
    Args:
        X: Feature matrix (numpy array)
        y: Labels (array-like)
        model_type: Type of model ("xgboost", "catboost", "tabpfn", or "lr")
        n_runs: Number of runs
        
    Returns:
        dict: Results with mean, std, and scores
    """
    f1_scores = []
    
    print(f"Running {n_runs} iterations for {model_type.upper()} on raw features...")
    
    for run in range(n_runs):
        print(f"  Run {run + 1}/{n_runs}...", end=" ")
        
        # Use different random seeds for each run
        random.seed(42 + run)
        np.random.seed(42 + run)
        
        if model_type == "tabpfn":
            f1 = run_row_classification_tabpfn(X, y, test_size=0.2)
        elif model_type in ["xgboost", "catboost", "lr"]:
            f1 = run_row_classification(X, y, model_type=model_type, test_size=0.2)
        else:
            raise ValueError(f"Unsupported model_type for raw features: {model_type}")
        
        f1_scores.append(f1)
        print(f"F1: {f1:.4f}")
    
    # Calculate statistics
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    print(f"\n  Results: {mean_f1:.4f} ± {std_f1:.4f}\n\n")
    
    return {
        'mean': mean_f1,
        'std': std_f1,
        'scores': f1_scores
    }

def evaluate_end_to_end_baselines(data, target_col, domain, n_runs=8):
    """
    Evaluate end-to-end baselines (XGBoost and TabPFN) on raw features.
    
    Args:
        data: List of JSON dictionaries (rows)
        target_col: Name of target column
        domain: Domain name (for logging)
        n_runs: Number of evaluation runs
        
    Returns:
        dict: Results dictionary with same format as other baselines
    """
    print(f"\n{domain} - End-to-End Baselines (Raw Features)")
    
    results = {}
    
    # Extract raw features
    print("\nExtracting raw features...")
    X, y = extract_raw_features(data, target_col)
    print(f"  Extracted {X.shape[1]} features from {X.shape[0]} samples")
    
    # XGBoost on raw features
    print("\nXGBoost (Raw Features)")
    results['xgboost_raw'] = {}
    result = run_repeated_classification_raw(X, y, model_type="xgboost", n_runs=n_runs)
    results['xgboost_raw'][f"xgboost_{domain}"] = result
    
    # TabPFN on raw features
    print("\nTabPFN (Raw Features)")
    results['tabpfn_raw'] = {}
    result = run_repeated_classification_raw(X, y, model_type="tabpfn", n_runs=n_runs)
    results['tabpfn_raw'][f"tabpfn_{domain}"] = result
    
    return results

def group_data_by_table(data):
    """
    Group data by table based on row_id patterns.
    Each table contains 10 consecutive rows with row_id 0-449.
    
    Args:
        data: List of dictionaries with row_id field
        
    Returns:
        List of tuples (table_idx, table_rows) where table_rows is a list of 10 rows
    """
    grouped_tables = []
    current_table = []
    table_idx = 0
    
    for row in data:
        row_id = int(row['row_id'])
        
        # Check if this is the start of a new table (row_id 0)
        if row_id == 0 and current_table:
            # Save the previous table
            if len(current_table) == 450:  # Ensure we have a complete table
                grouped_tables.append((table_idx, current_table))
                table_idx += 1
            current_table = []
        
        current_table.append(row)
    
    # Don't forget the last table
    if current_table and len(current_table) == 450:
        grouped_tables.append((table_idx, current_table))
    
    print(f"Grouped {len(data)} rows into {len(grouped_tables)} tables")
    return grouped_tables

def preprocess_wdc_movie(wdc_movie_data):
    """
    Preprocess WDC Movie data.
    """
    def is_valid_bert_word(word):
        tokens = tokenizer.tokenize(word.strip().lower())
        return len(tokens) == 1 and not tokens[0].startswith("##")

    def is_valid_genre_combo(genre_str):
        if not isinstance(genre_str, str): return False
        genres = [g.strip().lower() for g in genre_str.split(',')]
        return all(is_valid_bert_word(g) for g in genres)

    ## Count combinations that pass the BERT vocab test
    normalized_genres = []
    for row_dict in wdc_movie_data:
        genre_str = row_dict.get("genres", "")
        normalized = ", ".join(g.strip().lower() for g in genre_str.split(',')) if isinstance(genre_str, str) else ""
        if is_valid_genre_combo(normalized):
            normalized_genres.append(normalized)

    ## Get top 20 valid combinations
    genre_combo_counts = Counter(normalized_genres)
    top_10_combos = set(combo for combo, _ in genre_combo_counts.most_common(10))

    ## Filter the original dataset
    filtered_data = []
    for row_dict in wdc_movie_data:
        genre_str = row_dict.get("genres", "")
        normalized = ", ".join(g.strip().lower() for g in genre_str.split(',')) if isinstance(genre_str, str) else ""
        if normalized in top_10_combos:
            filtered_data.append(row_dict)

    wdc_movie_data = filtered_data

    ## Step 1: Extract all genre labels
    labels = [row_dict["genres"] for row_dict in wdc_movie_data]

    ## Step 2: Count frequencies
    label_counts = Counter(labels)

    ## Step 3: Identify valid labels (appear more than once)
    valid_labels = {label for label, count in label_counts.items() if count > 1}

    ## Step 4: Filter the data
    final_data = []
    for row_dict in wdc_movie_data:
        if row_dict["genres"] in valid_labels:
            final_data.append(row_dict)

    return final_data
    

def preprocess_wdc_product(wdc_product_data):
    """
    Preprocess WDC Product data.
    """
    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Helper function to check if a token is an English word in BERT vocab
    def is_english_category(category):
        tokens = tokenizer.tokenize(category.strip().lower())
        return len(tokens) == 1 and not tokens[0].startswith("##")

    # Convert to DataFrame
    df = pd.DataFrame(wdc_product_data)

    # Check if 'category' column exists
    if 'category' not in df.columns:
        print("Available columns:", df.columns.tolist())
        raise KeyError("'category' column not found in the data")

    # Split multi-label categories and normalize
    all_categories = []
    for category_str in df['category'].dropna():
        if category_str and category_str != 'nan':
            individual = [cat.strip().lower() for cat in category_str.split(',')]
            all_categories.extend(individual)

    # Count individual categories
    category_counts = Counter(all_categories)

    # Keep only those categories that are valid BERT words
    english_categories = {
        cat: count for cat, count in category_counts.items() if is_english_category(cat)
    }

    # Select top 20 English categories
    top_categories = sorted(english_categories.items(), key=lambda x: x[1], reverse=True)[:10]
    top_categories = [cat for cat, _ in top_categories]

    # Filter wdc_product_data to include only rows that contain at least one of the top categories
    filtered_wdc_product_data = []
    for row_dict in wdc_product_data:
        cats = row_dict.get("category", "")
        if isinstance(cats, str):
            split_cats = [cat.strip().lower() for cat in cats.split(',')]
            if any(cat in top_categories for cat in split_cats):
                filtered_wdc_product_data.append(row_dict)

    return filtered_wdc_product_data

def stratified_sample(data, label_key, sample_size=1000):
    # Group items by class
    label_to_items = defaultdict(list)
    for row_dict in data:
        label = row_dict.get(label_key)
        if label:
            label_to_items[label].append(row_dict)

    num_classes = len(label_to_items)
    per_class = max(1, sample_size // num_classes)

    # Sample per class
    sampled = []
    for items in label_to_items.values():
        sampled.extend(random.sample(items, min(len(items), per_class)))

    # Final adjustment to exactly sample_size
    if len(sampled) > sample_size:
        sampled = random.sample(sampled, sample_size)

    return sampled

def find_navi_checkpoint(base_dir, domain, seed=None, epoch=2):
    """
    Find NAVI model checkpoint path.
    
    Args:
        base_dir: Base models directory (e.g., './models')
        domain: Domain name (e.g., 'movie', 'product')
        seed: Training seed (None to find any, or specific seed)
        epoch: Epoch number (default: 2)
    
    Returns:
        str: Checkpoint path or None if not found
    """
    import glob
    
    if seed is not None:
        # Look for specific seed
        pattern = os.path.join(base_dir, f"navi_{domain}*", f"*seed{seed}*_epoch_{epoch}")
    else:
        # Look for any seed (prefer seed42 for backward compatibility)
        pattern = os.path.join(base_dir, f"navi_{domain}*", f"*seed42*_epoch_{epoch}")
        matches = glob.glob(pattern)
        if not matches:
            # Fallback to any seed
            pattern = os.path.join(base_dir, f"navi_{domain}*", f"*_epoch_{epoch}")
    
    matches = glob.glob(pattern)
    if matches:
        # Prefer the most specific match (longest path)
        matches.sort(key=len, reverse=True)
        return matches[0]
    
    # Fallback to old pattern for backward compatibility
    old_path = os.path.join(base_dir, f"navi_{domain}", f"epoch_{epoch}")
    if os.path.exists(old_path):
        return old_path
    
    return None

def load_baseline_models(tokenizer, navi_seed=None):
    """
    Load baseline models (BERT, HAETAE, TAPAS, NAVI).
    
    Args:
        tokenizer: Tokenizer instance
        navi_seed: Training seed for NAVI models (None = use default/seed42)
    """
    models = {}
    
    # Config
    config = BertConfig.from_pretrained('bert-base-uncased')

    # Baseline models - using trained versions
    models['bert_movie'] = BertForMaskedLM.from_pretrained('./models/bert_movie/epoch_2', local_files_only=True)
    models['bert_movie'] = models['bert_movie'].to(device)
    models['bert_movie'].eval()

    models['bert_product'] = BertForMaskedLM.from_pretrained('./models/bert_product/epoch_2', local_files_only=True)
    models['bert_product'] = models['bert_product'].to(device)
    models['bert_product'].eval()

    models['haetae_product'] = HAETAE(config, tokenizer, "./models/haetae_product/epoch_2")
    models['haetae_product'] = models['haetae_product'].to(device)
    models['haetae_product'].eval()

    models['haetae_movie'] = HAETAE(config, tokenizer, "./models/haetae_movie/epoch_2")
    models['haetae_movie'] = models['haetae_movie'].to(device)
    models['haetae_movie'].eval()

    models['tapas_movie'] = TapasForMaskedLM.from_pretrained('./models/tapas_movie/epoch_2', local_files_only=True)
    models['tapas_movie'] = models['tapas_movie'].to(device)
    models['tapas_movie'].eval()

    models['tapas_product'] = TapasForMaskedLM.from_pretrained('./models/tapas_product/epoch_2', local_files_only=True)
    models['tapas_product'] = models['tapas_product'].to(device)
    models['tapas_product'].eval()

    # NAVI models - use navi_movie and navi_product_default_3epoch
    def find_epoch2_path(base_path):
        """Find epoch_2 directory using glob pattern"""
        pattern = os.path.join(base_path, "*epoch_2")
        matches = glob.glob(pattern)
        if matches:
            # Prefer the most specific match (longest path)
            matches.sort(key=len, reverse=True)
            return matches[0]
        return None
    
    # NAVI Movie - use navi_movie
    navi_movie_base = './models/navi_movie'
    navi_movie_path = find_epoch2_path(navi_movie_base)
    if navi_movie_path:
        models['navi_movie'] = NaviForMaskedLM(navi_movie_path)
        models['navi_movie'] = models['navi_movie'].to(device)
        models['navi_movie'].eval()
        print(f"✓ Loaded NAVI Movie from: {navi_movie_path}")
    else:
        print(f"⚠️  NAVI Movie model not found: {navi_movie_base}/*epoch_2")
    
    # NAVI Product - use navi_product_default_3epoch
    navi_product_base = './models/navi_product'
    navi_product_path = find_epoch2_path(navi_product_base)
    if navi_product_path:
        models['navi_product'] = NaviForMaskedLM(navi_product_path)
        models['navi_product'] = models['navi_product'].to(device)
        models['navi_product'].eval()
        print(f"✓ Loaded NAVI Product from: {navi_product_path}")
    else:
        print(f"⚠️  NAVI Product model not found: {navi_product_base}/*epoch_2")
    
    return models

def load_ablation_models(tokenizer):
    """Load ablation models"""
    models = {}
    config = BertConfig.from_pretrained('bert-base-uncased')
    
    def find_epoch2_path(base_path):
        """Find epoch_2 directory using glob pattern"""
        pattern = os.path.join(base_path, "*epoch_2")
        matches = glob.glob(pattern)
        if matches:
            # Prefer the most specific match (longest path)
            matches.sort(key=len, reverse=True)
            return matches[0]
        return None
    
    domains = ['movie', 'product']
    ablation_values = ['woSSI', 'woMSM', 'woESA']

    for domain in domains:
        for ablation in ablation_values:
            base_path = f'./models/navi_{domain}_{ablation}'
            epoch_path = find_epoch2_path(base_path)
            if epoch_path:
                model_name = f'navi_{domain}_{ablation}'
                models[model_name] = NaviForMaskedLM(epoch_path, ablation_mode=ablation)
                models[model_name] = models[model_name].to(device)
                models[model_name].eval()
                print(f"✓ Loaded {model_name} from: {epoch_path}")
            else:
                print(f"⚠️  Model not found: {base_path}/*epoch_2")
    
    return models

def load_hyperparam_models_batch(tokenizer, hv_value, domain):
    """Load hyperparameter models for a specific hv value and domain"""
    models = {}
    
    # Convert domain to lowercase to match training script naming
    domain_lower = domain.lower()
    
    # Alignment weight variants
    align_values = ['0p25', '0p5', '1p0', '2p0', '4p0']
    for align in align_values:
        model_path = f'./models/navi_{domain_lower}_align{align}/epoch_2'
        if os.path.exists(model_path):
            model_name = f'navi_{domain_lower}_align{align}'
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
        else:
            print(f"⚠️  Model not found: {model_path}")
    
    # HV weight & Value ratio variants
    hv_vr_combinations = [
        ('0p8', '0p25'), ('0p8', '0p75'), 
        ('0p4', '0p25'), ('0p4', '0p5'), ('0p4', '0p75')
    ]
    for hv, vr in hv_vr_combinations:
        model_path = f'./models/navi_{domain_lower}_hv{hv}_vr{vr}/epoch_2'
        if os.path.exists(model_path):
            model_name = f'navi_{domain_lower}_hv{hv}_vr{vr}'
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
        else:
            print(f"⚠️  Model not found: {model_path}")
    
    # Tau variants
    tau_values = ['0p07', '0p1']
    for tau in tau_values:
        model_path = f'./models/navi_{domain_lower}_tau{tau}/epoch_2'
        if os.path.exists(model_path):
            model_name = f'navi_{domain_lower}_tau{tau}'
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
        else:
            print(f"⚠️  Model not found: {model_path}")
    
    return models

def load_rebuttal_models(tokenizer, domain):
    """Load rebuttal models: entropy threshold, tau variants, and gradient accumulation variants"""
    models = {}
    domain_lower = domain.lower()
    
    def find_epoch2_path(base_path):
        """Find epoch directory using glob pattern, filtered to only models with *_tau0.02_0.05_*"""
        pattern = os.path.join(base_path, "*epoch*")
        matches = glob.glob(pattern)
        # Filter to only include paths with _tau0.02_0.05_ in them
        matches = [m for m in matches if "_tau0.02_0.05_" in m]
        if matches:
            # Prefer the most specific match (longest path)
            matches.sort(key=len, reverse=True)
            return matches[0]
        return None
    
    # Entropy threshold variants
    ethresh_values = ['10_90', '40_60', '50_50']
    for ethresh in ethresh_values:
        base_path = f'./models/navi_{domain_lower}_ethresh{ethresh}'
        model_path = find_epoch2_path(base_path)
        if model_path:
            model_name = f'navi_{domain_lower}_ethresh{ethresh}'
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {model_path}")
        else:
            print(f"⚠️  Model not found: {base_path}/*epoch_2")
    
    # Tau variants
    tau_values = ['0p07_0p07', '0p13_0p13']
    for tau in tau_values:
        base_path = f'./models/navi_{domain_lower}_tau{tau}'
        model_path = find_epoch2_path(base_path)
        if model_path:
            model_name = f'navi_{domain_lower}_tau{tau}'
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {model_path}")
        else:
            print(f"⚠️  Model not found: {base_path}/*epoch_2")
    
    # Gradient accumulation variants
    ga_values = ['1', '4']
    for ga in ga_values:
        base_path = f'./models/navi_{domain_lower}_ga{ga}'
        model_path = find_epoch2_path(base_path)
        if model_path:
            model_name = f'navi_{domain_lower}_ga{ga}'
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {model_path}")
        else:
            print(f"⚠️  Model not found: {base_path}/*epoch_2")
    
    return models

def load_navi_models(tokenizer):
    """Load standard NAVI models (navi_movie, navi_product)"""
    models = {}
    
    def find_epoch2_path(base_path):
        """Find epoch_2 directory using glob pattern"""
        pattern = os.path.join(base_path, "*epoch_2")
        matches = glob.glob(pattern)
        if matches:
            # Prefer the most specific match (longest path)
            matches.sort(key=len, reverse=True)
            return matches[0]
        return None
    
    # NAVI Movie
    base_path = './models/navi_movie'
    navi_movie_path = find_epoch2_path(base_path)
    if navi_movie_path:
        models['navi_movie'] = NaviForMaskedLM(navi_movie_path)
        models['navi_movie'] = models['navi_movie'].to(device)
        models['navi_movie'].eval()
        print(f"✓ Loaded NAVI Movie from: {navi_movie_path}")
    else:
        print(f"⚠️  NAVI Movie model not found: {base_path}/*epoch_2")
    
    # NAVI Product
    base_path = './models/navi_product'
    navi_product_path = find_epoch2_path(base_path)
    if navi_product_path:
        models['navi_product'] = NaviForMaskedLM(navi_product_path)
        models['navi_product'] = models['navi_product'].to(device)
        models['navi_product'].eval()
        print(f"✓ Loaded NAVI Product from: {navi_product_path}")
    else:
        print(f"⚠️  NAVI Product model not found: {base_path}/*epoch_2")
    
    return models

def find_epoch_path(base_path, epoch):
    """Find epoch directory using glob pattern"""
    pattern = os.path.join(base_path, f"*epoch_{epoch}")
    matches = glob.glob(pattern)
    if matches:
        # Prefer the most specific match (longest path)
        matches.sort(key=len, reverse=True)
        return matches[0]
    return None

def load_training_variant_model(domain, epoch=None, header_encoder_mode='full'):
    """
    Load a training variant model for classification evaluation.
    
    Args:
        domain: Domain name ('movie' or 'product')
        epoch: Epoch number (1, 2, or 3). If None, uses epoch 2.
        header_encoder_mode: 'full', 'frozen', or 'partial'
    
    Returns:
        (Model instance, model_name) or (None, None) if not found
    """
    domain_lower = domain.lower()
    
    # Determine model directory based on variant
    if header_encoder_mode == 'frozen':
        base_path = f'./models/navi_{domain_lower}_hefrozen'
        model_name = f'navi_{domain_lower}_hefrozen'
    elif header_encoder_mode == 'partial':
        base_path = f'./models/navi_{domain_lower}_hepartial'
        model_name = f'navi_{domain_lower}_hepartial'
    elif epoch is not None and epoch in [1, 3]:
        # E1 or E3 from default_3epoch
        base_path = f'./models/navi_{domain_lower}_default_3epoch'
        model_name = f'navi_{domain_lower}_epoch{epoch}'
    else:
        # Default E2
        base_path = f'./models/navi_{domain_lower}_default_3epoch'
        model_name = f'navi_{domain_lower}_epoch2'
    
    # Find epoch directory
    target_epoch = epoch if epoch is not None else 2
    model_path = find_epoch_path(base_path, target_epoch)
    
    if model_path:
        print(f"✓ Loading {model_name} from: {model_path}")
        model = NaviForMaskedLM(model_path)
        model = model.to(device)
        model.eval()
        return model, model_name
    else:
        print(f"⚠️  Model not found: {base_path}/*epoch_{target_epoch}")
        return None, None

def load_ablation_variants_models(tokenizer, domain):
    """Load ablation variant models for a specific domain (includes default + all variants)"""
    models = {}
    domain_lower = domain.lower()
    
    def find_epoch2_path(base_path):
        """Find epoch directory using glob pattern, filtered to only models with *_tau0.02_0.05_*"""
        pattern = os.path.join(base_path, "*epoch*")
        matches = glob.glob(pattern)
        # Filter to only include paths with _tau0.02_0.05_ in them
        matches = [m for m in matches if "_tau0.02_0.05_" in m]
        if matches:
            # Prefer the most specific match (longest path)
            matches.sort(key=len, reverse=True)
            return matches[0]
        return None
    
    # 1. Default model
    base_path = f'./models/navi_{domain_lower}'
    model_path = find_epoch2_path(base_path)
    if model_path:
        model_name = f'navi_{domain_lower}'
        models[model_name] = NaviForMaskedLM(model_path)
        models[model_name] = models[model_name].to(device)
        models[model_name].eval()
        print(f"✓ Loaded {model_name} from: {model_path}")
    else:
        print(f"⚠️  Model not found: {base_path}/*epoch_2")
    
    # 2-4. Entropy threshold variants
    ethresh_values = ['10_90', '40_60', '50_50']
    for ethresh in ethresh_values:
        base_path = f'./models/navi_{domain_lower}_ethresh{ethresh}'
        model_path = find_epoch2_path(base_path)
        if model_path:
            model_name = f'navi_{domain_lower}_ethresh{ethresh}'
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {model_path}")
        else:
            print(f"⚠️  Model not found: {base_path}/*epoch_2")
    
    # 5-6. Temperature variants
    tau_values = ['0p07_0p07', '0p13_0p13']
    for tau in tau_values:
        base_path = f'./models/navi_{domain_lower}_tau{tau}'
        model_path = find_epoch2_path(base_path)
        if model_path:
            model_name = f'navi_{domain_lower}_tau{tau}'
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {model_path}")
        else:
            print(f"⚠️  Model not found: {base_path}/*epoch_2")
    
    # 7-8. Gradient accumulation variants (Neg batch size)
    ga_values = ['1', '4']  # ga1 = batch size 32, ga4 = batch size 128
    for ga in ga_values:
        base_path = f'./models/navi_{domain_lower}_ga{ga}'
        model_path = find_epoch2_path(base_path)
        if model_path:
            model_name = f'navi_{domain_lower}_ga{ga}'
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {model_path}")
        else:
            print(f"⚠️  Model not found: {base_path}/*epoch_2")
    
    return models

def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def print_class_distribution(data, label_key, dataset_name):
    label_list = [item[label_key] for item in data if item.get(label_key)]
    label_counts = Counter(label_list)
    print(f"\n=== {dataset_name} ===")
    for label, count in label_counts.most_common():
        print(f"{label:<30} {count:>5}")
    print(f"Total samples: {sum(label_counts.values())}")

def evaluate_baselines(data, target_col, models, domain, n_runs=8, embedding_type="cls"):
    """Evaluate baseline models (BERT, HAETAE, TAPAS)"""
    print(f"\n{domain} - Baselines")
    
    results = {}
    ml_models = ["xgboost", "catboost", "lr", "tabpfn"]
    
    # BERT
    print("\nBERT")
    bert_key = f'bert_{domain.lower()}'
    results['bert'] = {}
    for ml_model in ml_models:
        try:
            result = run_repeated_classification(
                data, target_col, models[bert_key], 'bert', domain,
                ml_model=ml_model, n_runs=n_runs, embedding_type=embedding_type
            )
        except Exception as e:
            _warn_failed(f"BERT/{ml_model} ({domain})", e)
            result = _failed_result(e)
        results['bert'][f"{ml_model}_{domain}"] = result
    
    # TAPAS
    print("\nTAPAS")
    tapas_key = f'tapas_{domain.lower()}'
    results['tapas'] = {}
    for ml_model in ml_models:
        try:
            result = run_repeated_classification(
                data, target_col, models[tapas_key], 'tapas', domain,
                ml_model=ml_model, n_runs=n_runs, embedding_type=embedding_type
            )
        except Exception as e:
            _warn_failed(f"TAPAS/{ml_model} ({domain})", e)
            result = _failed_result(e)
        results['tapas'][f"{ml_model}_{domain}"] = result
    
    # HAETAE
    print("\nHAETAE")
    haetae_key = f'haetae_{domain.lower()}'
    results['haetae'] = {}
    for ml_model in ml_models:
        try:
            result = run_repeated_classification(
                data, target_col, models[haetae_key], 'haetae', domain,
                ml_model=ml_model, n_runs=n_runs, embedding_type=embedding_type
            )
        except Exception as e:
            _warn_failed(f"HAETAE/{ml_model} ({domain})", e)
            result = _failed_result(e)
        results['haetae'][f"{ml_model}_{domain}"] = result

    # NAVI
    print("\nNAVI")
    navi_key = f'navi_{domain.lower()}'
    results['navi'] = {}
    for ml_model in ml_models:
        try:
            result = run_repeated_classification(
                data, target_col, models[navi_key], 'navi', domain,
                ml_model=ml_model, n_runs=n_runs, embedding_type=embedding_type
            )
        except Exception as e:
            _warn_failed(f"NAVI/{ml_model} ({domain})", e)
            result = _failed_result(e)
        results['navi'][f"{ml_model}_{domain}"] = result
    
    return results

def evaluate_ablations(data, target_col, models, domain, n_runs=8, embedding_type="cls"):
    """Evaluate ablation models"""
    print(f"\n{domain} - Ablations")
    
    results = {}
    ml_models = ["xgboost", "catboost", "lr", "tabpfn"]
    
    # Filter models for this domain
    domain_models = {k: v for k, v in models.items() if f"_{domain.lower()}_" in k}
    
    for model_name, model in domain_models.items():
        print(f"\n{model_name}")
        results[model_name] = {}
        for ml_model in ml_models:
            result = run_repeated_classification(
                data, target_col, model, model_name, domain, 
                ml_model=ml_model, n_runs=n_runs, embedding_type=embedding_type
            )
            results[model_name][f"{ml_model}_{domain}"] = result
    
    return results

def evaluate_hyperparams(data, target_col, tokenizer, domain, n_runs=8, embedding_type="cls"):
    """Evaluate hyperparameter variations with memory management"""
    print(f"\n{domain} - Hyperparameter Variations")
    
    results = {}
    ml_models = ["xgboost", "catboost", "lr", "tabpfn"]
    
    # First batch: hv0p4 (15 variants)
    print("\n=== Evaluating hv0p4 variants ===")
    hv0p4_models = load_hyperparam_models_batch(tokenizer, '0p4', domain)
    
    for model_name, model in hv0p4_models.items():
        print(f"\n{model_name}")
        results[model_name] = {}
        for ml_model in ml_models:
            result = run_repeated_classification(
                data, target_col, model, model_name, domain, 
                ml_model=ml_model, n_runs=n_runs, embedding_type=embedding_type
            )
            results[model_name][f"{ml_model}_{domain}"] = result
    
    # Clear memory after first batch
    del hv0p4_models
    clear_memory()
    
    # Second batch: hv0p8 (15 variants)
    print("\n=== Evaluating hv0p8 variants ===")
    hv0p8_models = load_hyperparam_models_batch(tokenizer, '0p8', domain)
    
    for model_name, model in hv0p8_models.items():
        print(f"\n{model_name}")
        results[model_name] = {}
        for ml_model in ml_models:
            result = run_repeated_classification(
                data, target_col, model, model_name, domain, 
                ml_model=ml_model, n_runs=n_runs, embedding_type=embedding_type
            )
            results[model_name][f"{ml_model}_{domain}"] = result
    
    # Clear memory after second batch
    del hv0p8_models
    clear_memory()
    
    return results

def evaluate_navi_embeddings(data, target_col, tokenizer, domain, n_runs=8, embedding_type="cls"):
    """Evaluate NAVI embeddings approach"""
    print(f"\n{domain} - NAVI Embeddings")
    
    results = {}
    ml_models = ["xgboost", "catboost", "lr", "tabpfn"]
    
    # Load NAVI model
    models = load_navi_models(tokenizer)
    navi_key = f'navi_{domain.lower()}'
    
    if navi_key not in models:
        print(f"⚠️  {navi_key} model not found")
        return results
    
    results['navi_emb'] = {}
    for ml_model in ml_models:
        try:
            result = run_repeated_classification(
                data, target_col, models[navi_key], navi_key, domain,
                ml_model=ml_model, n_runs=n_runs, embedding_type=embedding_type
            )
        except Exception as e:
            _warn_failed(f"NAVI-emb/{ml_model} ({domain})", e)
            result = _failed_result(e)
        results['navi_emb'][f"{ml_model}_{domain.lower()}"] = result
    
    del models
    clear_memory()
    
    return results

def evaluate_tablevectorizer(data, target_col, domain, n_runs=8):
    """Evaluate TableVectorizer features approach"""
    print(f"\n{domain} - TableVectorizer Features")
    
    from experiments.experiment_utils import extract_features_tablevectorizer, run_classification_with_features
    
    results = {}
    ml_models = ["xgboost", "catboost", "lr", "tabpfn"]
    
    # Extract features using TableVectorizer
    print("Extracting features with TableVectorizer...")
    try:
        X, y = extract_features_tablevectorizer(data, target_col)
        print(f"  Extracted {X.shape[1]} features from {X.shape[0]} samples")
    except Exception as e:
        _warn_failed(f"TableVectorizer feature extraction ({domain})", e)
        return results
    
    results['tablevectorizer'] = {}
    for ml_model in ml_models:
        print(f"\n{ml_model.upper()}")
        f1_scores = []
        
        for run in range(n_runs):
            print(f"  Run {run + 1}/{n_runs}...", end=" ")
            random.seed(42 + run)
            np.random.seed(42 + run)
            try:
                f1 = run_classification_with_features(X, y, model_type=ml_model, test_size=0.2)
                f1_scores.append(f1)
                print(f"F1: {f1:.4f}")
            except Exception as e:
                _warn_failed(f"TableVectorizer/{ml_model} ({domain})", e)
                f1_scores = []
                break
        
        if not f1_scores:
            results['tablevectorizer'][f"{ml_model}_{domain.lower()}"] = _failed_result(RuntimeError("No successful runs"))
        else:
            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)
            print(f"\n  Results: {mean_f1:.4f} ± {std_f1:.4f}\n")
            results['tablevectorizer'][f"{ml_model}_{domain.lower()}"] = {
                'mean': mean_f1,
                'std': std_f1,
                'scores': f1_scores
            }
    
    return results

def evaluate_textencoder(data, target_col, domain, n_runs=8):
    """Evaluate TextEncoder features approach"""
    print(f"\n{domain} - TextEncoder Features")
    
    from experiments.experiment_utils import extract_features_textencoder, run_classification_with_features
    
    results = {}
    ml_models = ["xgboost", "tabpfn"]
    
    # Extract features using TextEncoder
    print("Extracting features with TextEncoder...")
    X, y = extract_features_textencoder(data, target_col)
    print(f"  Extracted {X.shape[1]} features from {X.shape[0]} samples")
    
    results['textencoder'] = {}
    for ml_model in ml_models:
        print(f"\n{ml_model.upper()}")
        f1_scores = []
        
        for run in range(n_runs):
            print(f"  Run {run + 1}/{n_runs}...", end=" ")
            random.seed(42 + run)
            np.random.seed(42 + run)
            try:
                f1 = run_classification_with_features(X, y, model_type=ml_model, test_size=0.2)
                f1_scores.append(f1)
                print(f"F1: {f1:.4f}")
            except Exception as e:
                _warn_failed(f"TextEncoder/{ml_model} ({domain})", e)
                f1_scores = []
                break
        
        if not f1_scores:
            results['textencoder'][f"{ml_model}_{domain.lower()}"] = _failed_result(RuntimeError("No successful runs"))
        else:
            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)
            print(f"\n  Results: {mean_f1:.4f} ± {std_f1:.4f}\n")
            results['textencoder'][f"{ml_model}_{domain.lower()}"] = {
                'mean': mean_f1,
                'std': std_f1,
                'scores': f1_scores
            }
    
    return results

def evaluate_raw_features(data, target_col, domain, n_runs=8):
    """Evaluate raw features approach"""
    print(f"\n{domain} - Raw Features")
    
    results = {}
    ml_models = ["xgboost", "catboost", "lr", "tabpfn"]
    
    # Extract raw features
    print("Extracting raw features...")
    X, y = extract_raw_features(data, target_col)
    print(f"  Extracted {X.shape[1]} features from {X.shape[0]} samples")
    
    results['raw'] = {}
    for ml_model in ml_models:
        print(f"\n{ml_model.upper()}")
        try:
            result = run_repeated_classification_raw(X, y, model_type=ml_model, n_runs=n_runs)
        except Exception as e:
            _warn_failed(f"Raw/{ml_model} ({domain})", e)
            result = _failed_result(e)
        results['raw'][f"{ml_model}_{domain.lower()}"] = result
    
    return results

def evaluate_concatenated_navi(data, target_col, tokenizer, domain, n_runs=8, embedding_type="cls"):
    """Evaluate concatenated NAVI + numerical features approach"""
    print(f"\n{domain} - Concatenated NAVI + Numerical Features")
    
    from experiments.experiment_utils import extract_concatenated_navi_features, run_classification_with_features
    
    results = {}
    ml_models = ["xgboost", "catboost", "lr", "tabpfn"]
    
    # Load NAVI model
    models = load_navi_models(tokenizer)
    navi_key = f'navi_{domain.lower()}'
    
    if navi_key not in models:
        print(f"⚠️  {navi_key} model not found")
        return results
    
    # Prepare dataset without target column
    preprocessed_dataset = [remove_target_column(row, target_col) for row in data]
    dataset_X = NaviDataset(preprocessed_dataset)
    
    # Extract concatenated features
    print("Extracting concatenated NAVI + numerical features...")
    X, y = extract_concatenated_navi_features(data, target_col, models[navi_key], dataset_X)
    print(f"  Extracted {X.shape[1]} features from {X.shape[0]} samples (768 NAVI + {X.shape[1] - 768} numerical)")
    
    results['navi_concat'] = {}
    for ml_model in ml_models:
        print(f"\n{ml_model.upper()}")
        f1_scores = []
        
        for run in range(n_runs):
            print(f"  Run {run + 1}/{n_runs}...", end=" ")
            random.seed(42 + run)
            np.random.seed(42 + run)
            try:
                f1 = run_classification_with_features(X, y, model_type=ml_model, test_size=0.2)
                f1_scores.append(f1)
                print(f"F1: {f1:.4f}")
            except Exception as e:
                _warn_failed(f"NAVI+num/{ml_model} ({domain})", e)
                f1_scores = []
                break
        
        if not f1_scores:
            results['navi_concat'][f"{ml_model}_{domain.lower()}"] = _failed_result(RuntimeError("No successful runs"))
        else:
            mean_f1 = np.mean(f1_scores)
            std_f1 = np.std(f1_scores)
            print(f"\n  Results: {mean_f1:.4f} ± {std_f1:.4f}\n")
            results['navi_concat'][f"{ml_model}_{domain.lower()}"] = {
                'mean': mean_f1,
                'std': std_f1,
                'scores': f1_scores
            }
    
    del models
    clear_memory()
    
    return results

def evaluate_rebuttal(data, target_col, tokenizer, domain, n_runs=8, embedding_type="cls"):
    """Evaluate rebuttal models (entropy threshold, tau variants, gradient accumulation)"""
    print(f"\n{domain} - Rebuttal Models")
    
    results = {}
    ml_models = ["xgboost", "catboost", "lr", "tabpfn"]
    
    # # Load rebuttal models
    # models = load_navi_models(tokenizer)
    
    # # Filter models for this domain
    # domain_models = {k: v for k, v in models.items() if f"_{domain.lower()}_" in k}

    # for model_name, model in domain_models.items():
    #     print(f"\n{model_name}")
    #     results[model_name] = {}
    #     for ml_model in ml_models:
    #         result = run_repeated_classification(
    #             data, target_col, model, model_name, domain, 
    #             ml_model=ml_model, n_runs=n_runs, embedding_type=embedding_type
    #         )
    #         results[model_name][f"{ml_model}_{domain.lower()}"] = result
    
    # # Clear memory
    # del models
    # clear_memory()
    
    # Now evaluate all 5 baseline approaches
    print(f"\n{'='*80}")
    print(f"Evaluating 5 Classification Approaches")
    print(f"{'='*80}\n")
    
    # # 1. NAVI embeddings
    # navi_results = evaluate_navi_embeddings(data, target_col, tokenizer, domain, n_runs=n_runs, embedding_type=embedding_type)
    # results.update(navi_results)
    
    # # 3. TextEncoder features
    # textencoder_results = evaluate_textencoder(data, target_col, domain, n_runs=n_runs)
    # results.update(textencoder_results)
    
    # 4. Raw features
    raw_results = evaluate_raw_features(data, target_col, domain, n_runs=n_runs)
    results.update(raw_results)
    
    # 5. Concatenated NAVI + numerical features
    concat_results = evaluate_concatenated_navi(data, target_col, tokenizer, domain, n_runs=n_runs, embedding_type=embedding_type)
    results.update(concat_results)

    # 2. TableVectorizer features
    tablevectorizer_results = evaluate_tablevectorizer(data, target_col, domain, n_runs=n_runs)
    results.update(tablevectorizer_results)
    
    return results

def evaluate_navi(data, target_col, tokenizer, domain, n_runs=8, embedding_type="cls"):
    """Evaluate standard NAVI models (navi_movie, navi_product)"""
    print(f"\n{domain} - NAVI Models")
    
    results = {}
    ml_models = ["xgboost", "catboost", "lr", "tabpfn"]
    
    # Load NAVI models
    models = load_navi_models(tokenizer)
    
    # Get the model for this domain
    navi_key = f'navi_{domain.lower()}'
    if navi_key in models:
        print(f"\n{navi_key}")
        results['navi'] = {}
        for ml_model in ml_models:
            result = run_repeated_classification(
                data, target_col, models[navi_key], navi_key, domain, 
                ml_model=ml_model, n_runs=n_runs, embedding_type=embedding_type
            )
            results['navi'][f"{ml_model}_{domain.lower()}"] = result
    else:
        print(f"⚠️  {navi_key} model not found")
    
    return results

def load_tau_align_ethresh_models(tokenizer, domain):
    """Load tau/align/ethresh variant models for a specific domain"""
    models = {}
    domain_lower = domain.lower()
    
    def find_epoch2_path(base_path):
        """Find epoch directory using glob pattern, filtered to only models with *_tau0.02_0.05_*"""
        pattern = os.path.join(base_path, "*epoch*")
        matches = glob.glob(pattern)
        # Filter to only include paths with _tau0.02_0.05_ in them
        matches = [m for m in matches if "_tau0.02_0.05_" in m]
        if matches:
            # Prefer the most specific match (longest path)
            matches.sort(key=len, reverse=True)
            return matches[0]
        return None
    
    # Define all model variants based on log files
    # Format: (tau, align, ga) where ga can be None or a string like '1'
    variants = [
        ('0p01_0p1', '0p1', None),
        ('0p01_0p14', '0p1', None),
        ('0p01_0p14', '0p1', '1'),
        ('0p01_0p14', '0p05', None),
        ('0p01_0p14', '0p05', '1'),
        ('0p02_0p14', '0p1', None),
        ('0p02_0p14', '0p05', None),
        ('0p02_0p14', '0p05', '1'),
        ('0p02_0p14', '1p0', None),
        ('0p03_0p25', '0p1', None),
        ('0p05_0p1', '0p1', None),
    ]
    
    ethresh = '10_90'
    
    for tau, align, ga in variants:
        # Build base path
        if ga is not None:
            base_path = f'./models/navi_{domain_lower}_tau{tau}_align{align}_ethresh{ethresh}_ga{ga}'
            model_name = f'navi_{domain_lower}_tau{tau}_align{align}_ethresh{ethresh}_ga{ga}'
        else:
            base_path = f'./models/navi_{domain_lower}_tau{tau}_align{align}_ethresh{ethresh}'
            model_name = f'navi_{domain_lower}_tau{tau}_align{align}_ethresh{ethresh}'
        
        model_path = find_epoch2_path(base_path)
        if model_path:
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {model_path}")
        else:
            print(f"⚠️  Model not found: {base_path}/*epoch_2")
    
    return models

def evaluate_ablation_variants(data, target_col, tokenizer, domain, n_runs=8, embedding_type="cls", ml_model="xgboost"):
    """Evaluate ablation variants (entropy threshold, temperature, gradient accumulation)"""
    print(f"\n{domain} - Ablation Variants")
    
    results = {}
    
    # Load all ablation variant models
    models = load_ablation_variants_models(tokenizer, domain)
    
    # Evaluate each model
    for model_name, model in models.items():
        print(f"\n{model_name}")
        result = run_repeated_classification(
            data, target_col, model, model_name, domain, 
            ml_model=ml_model, n_runs=n_runs, embedding_type=embedding_type
        )
        results[model_name] = {
            f"{ml_model}_{domain.lower()}": result
        }
    
    # Clear memory
    del models
    clear_memory()
    
    return results

def evaluate_tau_align_ethresh(data, target_col, tokenizer, domain, n_runs=8, embedding_type="cls", ml_model="xgboost"):
    """Evaluate tau/align/ethresh variant models"""
    print(f"\n{domain} - Tau/Align/Ethresh Variants")
    
    results = {}
    ml_models = ["xgboost", "catboost", "lr"]
    
    # Load tau/align/ethresh models
    models = load_tau_align_ethresh_models(tokenizer, domain)
    
    # Evaluate each model
    for model_name, model in models.items():
        for ml_model in ml_models:
            print(f"\n{model_name}")
            result = run_repeated_classification(
                data, target_col, model, model_name, domain, 
                ml_model=ml_model, n_runs=n_runs, embedding_type=embedding_type
            )
            results[model_name] = {
                f"{ml_model}_{domain.lower()}": result
            }
    
    # Clear memory
    del models
    clear_memory()
    
    return results

def load_hyperparam_sensitivity_models(tokenizer, domain):
    """Load hyperparameter sensitivity models: only align*, ethresh*, and tau* variants"""
    models = {}
    domain_lower = domain.lower()
    
    def find_epoch2_path(base_path):
        """Find epoch directory using glob pattern, filtered to only models with *_tau0.02_0.05_*"""
        pattern = os.path.join(base_path, "*epoch*")
        matches = glob.glob(pattern)
        # Filter to only include paths with _tau0.02_0.05_ in them
        matches = [m for m in matches if "_tau0.02_0.05_" in m]
        if matches:
            matches.sort(key=len, reverse=True)
            return matches[0]
        return None
    
    # Find all align* variants using glob
    align_pattern = f'./models/navi_{domain_lower}_align*'
    align_dirs = glob.glob(align_pattern)
    for align_dir in sorted(align_dirs):
        if os.path.isdir(align_dir):
            # Extract model name from path
            model_name = os.path.basename(align_dir)
            # Skip if it contains hv, vr, or ga (combined variants)
            if '_hv' in model_name or '_vr' in model_name or '_ga' in model_name:
                continue
            model_path = find_epoch2_path(align_dir)
            if model_path:
                models[model_name] = NaviForMaskedLM(model_path)
                models[model_name] = models[model_name].to(device)
                models[model_name].eval()
                print(f"✓ Loaded {model_name} from: {model_path}")
            else:
                print(f"⚠️  Model not found: {align_dir}/*epoch_2")
    
    # Find all ethresh* variants using glob
    ethresh_pattern = f'./models/navi_{domain_lower}_ethresh*'
    ethresh_dirs = glob.glob(ethresh_pattern)
    for ethresh_dir in sorted(ethresh_dirs):
        if os.path.isdir(ethresh_dir):
            # Extract model name from path
            model_name = os.path.basename(ethresh_dir)
            # Skip if it contains tau, align, hv, vr, or ga (combined variants)
            if '_tau' in model_name or '_align' in model_name or '_hv' in model_name or '_vr' in model_name or '_ga' in model_name:
                continue
            model_path = find_epoch2_path(ethresh_dir)
            if model_path:
                models[model_name] = NaviForMaskedLM(model_path)
                models[model_name] = models[model_name].to(device)
                models[model_name].eval()
                print(f"✓ Loaded {model_name} from: {model_path}")
            else:
                print(f"⚠️  Model not found: {ethresh_dir}/*epoch_2")
    
    # Find all tau* variants using glob
    tau_pattern = f'./models/navi_{domain_lower}_tau*'
    tau_dirs = glob.glob(tau_pattern)
    for tau_dir in sorted(tau_dirs):
        if os.path.isdir(tau_dir):
            # Extract model name from path
            model_name = os.path.basename(tau_dir)
            # Skip if it contains align, ethresh, hv, vr, or ga (combined variants)
            if '_align' in model_name or '_ethresh' in model_name or '_hv' in model_name or '_vr' in model_name or '_ga' in model_name:
                continue
            model_path = find_epoch2_path(tau_dir)
            if model_path:
                models[model_name] = NaviForMaskedLM(model_path)
                models[model_name] = models[model_name].to(device)
                models[model_name].eval()
                print(f"✓ Loaded {model_name} from: {model_path}")
            else:
                print(f"⚠️  Model not found: {tau_dir}/*epoch_2")
    
    return models

def evaluate_hyperparam_sensitivity(data, target_col, tokenizer, domain, n_runs=8, embedding_type="cls"):
    """Evaluate hyperparameter sensitivity analysis - XGBoost, CatBoost, LogisticRegression, and TabPFN"""
    print(f"\n{domain} - Hyperparameter Sensitivity Analysis")
    
    results = {}
    ml_models = ["xgboost", "catboost", "lr", "tabpfn"]
    
    # Load all hyperparameter sensitivity models
    models = load_hyperparam_sensitivity_models(tokenizer, domain)
    
    # Evaluate each model
    for model_name, model in models.items():
        print(f"\n{model_name}")
        results[model_name] = {}
        for ml_model in ml_models:
            result = run_repeated_classification(
                data, target_col, model, model_name, domain,
                ml_model=ml_model, n_runs=n_runs, embedding_type=embedding_type
            )
            results[model_name][f"{ml_model}_{domain.lower()}"] = result
    
    # Clear memory
    del models
    clear_memory()
    
    return results

def evaluate_training_variants(data, target_col, tokenizer, domain, n_runs=8, embedding_type="cls", ml_model="xgboost"):
    """Evaluate training variants (E1, E2, E3, Header Encoder variants) for classification"""
    print(f"\n{domain} - Training Variants")
    
    results = {}
    domain_lower = domain.lower()
    
    # Variants to evaluate
    variants = [
        ('Default (E2)', None, 'full'),
        ('E1', 1, 'full'),
        ('E3', 3, 'full'),
        ('Header Enc Frozen', None, 'frozen'),
        ('Header Enc Partial', None, 'partial'),
    ]
    
    for variant_name, epoch, header_mode in variants:
        print(f"\n{'='*60}")
        print(f"Evaluating: {variant_name}")
        print(f"{'='*60}")
        
        # Load model
        model, model_name = load_training_variant_model(domain_lower, epoch=epoch, header_encoder_mode=header_mode)
        if model is None:
            print(f"⚠️  Skipping {variant_name} - model not found")
            continue
        
        # Evaluate
        result = run_repeated_classification(
            data, target_col, model, variant_name, domain, 
            ml_model=ml_model, n_runs=n_runs, embedding_type=embedding_type
        )
        results[variant_name] = {
            f"{ml_model}_{domain_lower}": result
        }
        
        # Clear memory
        del model
        clear_memory()
    
    return results

def evaluate_with_seed_variance(data, target_col, tokenizer, domain, seeds=[0, 1, 2], 
                                n_runs=8, embedding_type="cls"):
    """
    Evaluate models across multiple training seeds and aggregate results.
    
    Args:
        data: Dataset
        target_col: Target column name
        tokenizer: Tokenizer
        domain: Domain name
        seeds: List of training seeds to evaluate
        n_runs: Number of evaluation runs per seed (for train/test split variance)
        embedding_type: Embedding type
    
    Returns:
        dict: Aggregated results across seeds
    """
    print(f"\n{'='*80}")
    print(f"Evaluating seed variance for {domain} domain")
    print(f"Training seeds: {seeds}")
    print(f"Evaluation runs per seed: {n_runs}")
    print(f"{'='*80}\n")
    
    all_seed_results = {}
    ml_models = ["xgboost", "catboost", "lr", "tabpfn"]
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Evaluating seed {seed}")
        print(f"{'='*60}\n")
        
        # Load models with this seed
        models = load_baseline_models(tokenizer, navi_seed=seed)
        
        if 'navi_movie' not in models and 'navi_product' not in models:
            print(f"⚠️  Skipping seed {seed} - NAVI model not found")
            continue
        
        # Run experiments for this seed
        seed_results = {}
        domain_key = domain.lower()
        navi_key = f'navi_{domain_key}'
        
        if navi_key in models:
            seed_results['navi'] = {}
            for ml_model in ml_models:
                result = run_repeated_classification(
                    data, target_col, models[navi_key], navi_key, domain, 
                    ml_model=ml_model, n_runs=n_runs, embedding_type=embedding_type
                )
                seed_results['navi'][f"{ml_model}_{domain_key}"] = result
        
        all_seed_results[seed] = seed_results
        
        # Clear memory
        del models
        clear_memory()
    
    # Aggregate across seeds
    print(f"\n{'='*80}")
    print("Aggregating results across seeds...")
    print(f"{'='*80}\n")
    
    aggregated = {}
    for model_type in ['navi']:
        aggregated[model_type] = {}
        for ml_model in ml_models:
            metric_key = f"{ml_model}_{domain_key}"
            seed_values = []
            
            for seed, seed_results in all_seed_results.items():
                if model_type in seed_results and metric_key in seed_results[model_type]:
                    mean_f1 = seed_results[model_type][metric_key].get('mean')
                    if mean_f1 is not None:
                        seed_values.append(mean_f1)
            
            if seed_values:
                aggregated[model_type][metric_key] = {
                    'mean': np.mean(seed_values),
                    'std': np.std(seed_values),
                    'min': np.min(seed_values),
                    'max': np.max(seed_values),
                    'values': seed_values,
                    'num_seeds': len(seed_values)
                }
                print(f"{model_type} {ml_model}: {aggregated[model_type][metric_key]['mean']:.4f} ± {aggregated[model_type][metric_key]['std']:.4f} (across {len(seed_values)} seeds)")
    
    return aggregated, all_seed_results

def main():
    parser = argparse.ArgumentParser(description='Run row classification experiments')
    parser.add_argument('--model', choices=['baselines', 'ablations', 'hyperparams', 'seed_variance', 'rebuttal', 'navi', 'ablation_variants', 'training_variants', 'tau_align_ethresh', 'hyperparam_sensitivity'], 
                       required=True, help='Type of models to evaluate')
    parser.add_argument('--domain', choices=['Movie', 'Product'], 
                       required=True, help='Domain to evaluate on')
    parser.add_argument("--embedding_type", type=str, default="cls", choices=["cls", "mean"], 
                       help="Type of embedding to use")
    parser.add_argument("--seeds", type=int, nargs='+', default=[0, 1, 2],
                       help="Training seeds for seed variance analysis (only used with --model seed_variance)")
    
    args = parser.parse_args()
    
    print(f"Evaluating {args.model} models on {args.domain} domain")
    
    # Load and process data for the specified domain only
    if args.domain == 'Product':
        # Load Product classification dataset
        wdc_product_data = load_data("data/cleaned/Product/test/WDC_product_for_cls.jsonl")
        
        # Verify target columns are present (data already cleaned in preprocess.py)
        print("Verifying target columns in product data...")
        wdc_product_data = verify_target_columns_present(wdc_product_data, target_columns=['category'])
        
        # Preprocess WDC data
        wdc_product_data = preprocess_wdc_product(wdc_product_data)
        wdc_product_data = stratified_sample(wdc_product_data, "category")
        
        print(f"WDC Product data: {len(wdc_product_data)}")
        print_class_distribution(wdc_product_data, "category", "WDC Product")
        
        data = wdc_product_data
        target_col = 'category'
    else:  # Movie
        # Load Movie classification dataset
        wdc_movie_data = load_data("data/cleaned/Movie/test/WDC_movie_for_cls.jsonl")
        
        # Verify target columns are present (data already cleaned in preprocess.py)
        print("Verifying target columns in movie data...")
        wdc_movie_data = verify_target_columns_present(wdc_movie_data, target_columns=['genres'])
        
        # Preprocess WDC data
        wdc_movie_data = preprocess_wdc_movie(wdc_movie_data)
        wdc_movie_data = stratified_sample(wdc_movie_data, "genres")
        
        print(f"WDC Movie data: {len(wdc_movie_data)}")
        print_class_distribution(wdc_movie_data, "genres", "WDC Movie")
        
        data = wdc_movie_data
        target_col = 'genres'

    # Load models based on type
    if args.model == 'baselines':
        models = load_baseline_models(tokenizer)
        results = evaluate_baselines(data, target_col, models, args.domain.lower(), embedding_type=args.embedding_type)
        
        # # Add end-to-end baselines (XGBoost and TabPFN on raw features)
        # e2e_results = evaluate_end_to_end_baselines(data, target_col, args.domain.lower(), n_runs=8)
        # results.update(e2e_results)
        
    elif args.model == 'ablations':
        models = load_ablation_models(tokenizer)
        results = evaluate_ablations(data, target_col, models, args.domain.lower(), embedding_type=args.embedding_type)
        
    elif args.model == 'hyperparams':
        results = evaluate_hyperparams(data, target_col, tokenizer, args.domain, embedding_type=args.embedding_type)
    
    elif args.model == 'seed_variance':
        aggregated, all_seed_results = evaluate_with_seed_variance(
            data, target_col, tokenizer, args.domain, seeds=args.seeds, 
            embedding_type=args.embedding_type
        )
        results = {
            'aggregated': aggregated,
            'per_seed': all_seed_results
        }
    
    elif args.model == 'rebuttal':
        results = evaluate_rebuttal(data, target_col, tokenizer, args.domain, embedding_type=args.embedding_type)
    
    elif args.model == 'navi':
        results = evaluate_navi(data, target_col, tokenizer, args.domain, embedding_type=args.embedding_type)
    
    elif args.model == 'ablation_variants':
        results = evaluate_ablation_variants(data, target_col, tokenizer, args.domain, embedding_type=args.embedding_type)
    
    elif args.model == 'training_variants':
        results = evaluate_training_variants(data, target_col, tokenizer, args.domain, embedding_type=args.embedding_type)
    
    elif args.model == 'tau_align_ethresh':
        results = evaluate_tau_align_ethresh(data, target_col, tokenizer, args.domain, embedding_type=args.embedding_type)
    
    elif args.model == 'hyperparam_sensitivity':
        results = evaluate_hyperparam_sensitivity(data, target_col, tokenizer, args.domain, embedding_type=args.embedding_type)

    # Save results to JSON file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.model == 'seed_variance':
        results_file = f"experiments/logs/row_classification_{args.model}_{args.domain.lower()}_seeds{'_'.join(map(str, args.seeds))}_{timestamp}.json"
    else:
        results_file = f"experiments/logs/row_classification_{args.model}_{args.domain.lower()}_{timestamp}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    # Prepare results for JSON serialization
    json_results = {}
    for model_name, model_results in results.items():
        json_results[model_name] = {}
        for task_name, task_result in model_results.items():
            json_results[model_name][task_name] = {
                'mean': convert_numpy(task_result['mean']),
                'std': convert_numpy(task_result['std']),
                'scores': convert_numpy(task_result['scores'])
            }
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ Results saved to {results_file}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY RESULTS (Mean ± Std)")
    print("="*80)
    
    # Print all results (xgboost, catboost, lr, tabpfn)
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        for task_name, task_result in model_results.items():
            mean_val = task_result.get('mean')
            std_val = task_result.get('std')
            if mean_val is None or std_val is None:
                print(f"  {task_name}: FAILED/SKIPPED")
            else:
                print(f"  {task_name}: {mean_val:.4f} ± {std_val:.4f}")


if __name__ == "__main__":
    main()