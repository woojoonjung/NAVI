import torch
import numpy as np
import json
import random
import pandas as pd
import os
import sys
import argparse
import gc
import glob
from datetime import datetime
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
from experiments.downstream_tasks.cls_utils import (
    load_data, 
    run_row_classification,
    run_row_classification_tabpfn,
    extract_raw_features,
    get_cls_embedding,
    get_meanpooled_embedding,
    extract_features_tablevectorizer,
    extract_features_textencoder,
    extract_concatenated_navi_features,
    run_classification_with_features,
    apply_product_column_ablation,
)

try:
    import transformers  # noqa: F401 — fail fast with a clear fix if HF stack is broken
except ImportError as e:
    if "HfFolder" in str(e):
        raise ImportError(
            "transformers/huggingface_hub mismatch: older transformers expect HfFolder, "
            "which was removed in newer huggingface_hub. If you use the CM2 conda env, run:\n"
            "  pip install 'huggingface-hub==0.16.4'\n"
            "or align with baselines/CM2/requirements.txt (transformers==4.24.0, huggingface-hub==0.16.4)."
        ) from e
    raise

# Model imports (HAETAE is lazy — see _get_haetae_cls — so CM2-only runs skip safetensors/haetae.model)
from transformers import BertForMaskedLM, BertTokenizer, TapasForMaskedLM
from model.navi import NaviForMaskedLM

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

DATA_ROOT = "data/"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from dataset.dataset import NaviDataset, BertDataset, TapasDataset
from baselines.haetae.dataset import JSONDataset
from collections import Counter, defaultdict

from transformers import BertTokenizer, BertConfig
from config import config as app_config

from utils.navi_checkpoints import (
    discover_navi_domain_model_dirs,
    resolve_navi_checkpoint_path_with_fallback,
    navi_ablation_mode_for_model_name,
    iter_chunks,
)
from experiments.downstream_tasks.cm2_utils import (
    configure_cm2_classifier_schema,
    get_cm2_row_embedding,
    is_cm2_mask_model,
    load_cm2_classifier_model,
    load_cm2_mask_model,
    load_cm2_models_for_domain,
)

_haetae_cls = None


def _get_haetae_cls():
    """Lazy import: HAETAE pulls safetensors via baselines.haetae.model — not needed for --mode cm2."""
    global _haetae_cls
    if _haetae_cls is None:
        from baselines.haetae.model import HAETAE

        _haetae_cls = HAETAE
    return _haetae_cls


_bert_name = app_config.get_bert_name()
_checkpoint_epoch = getattr(app_config, "CHECKPOINT_EPOCH", 2)
_bert_kw = {"local_files_only": app_config.use_local_files_only()}
bert_config = BertConfig.from_pretrained(_bert_name, **_bert_kw)
tokenizer = BertTokenizer.from_pretrained(_bert_name, **_bert_kw)

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


def _default_cm2_schema_csv(heldout_path: Path, domain: str) -> Path:
    """Strict WDC split: finetune-train CSV in the same directory as heldout."""
    return heldout_path.parent / f"WDC_{domain.lower()}_for_cls_finetune_train.csv"


def _load_heldout_finetune_strict(heldout_path: str):
    """Load strict heldout from CSV or JSONL (one dict per row)."""
    p = Path(heldout_path)
    if not p.is_file():
        raise FileNotFoundError(f"Heldout not found: {p}")
    suf = p.suffix.lower()
    if suf == ".jsonl":
        return load_data(str(p), path_is="jsonl")
    if suf == ".csv":
        return load_data(str(p), path_is="csv")
    raise ValueError(f"Heldout must be .csv or .jsonl, got {p}")


def _filter_rows_top_k_labels(data: list, target_col: str, k: int) -> list:
    """Keep rows whose label is among the k most frequent labels in this dataset."""
    if k <= 0:
        return data

    def _label_key(row):
        lab = row[target_col]
        return tuple(lab) if isinstance(lab, (list, tuple)) else lab

    counts = Counter(_label_key(row) for row in data)
    top_keys = {key for key, _ in counts.most_common(k)}
    return [row for row in data if _label_key(row) in top_keys]


def run_cls_classification(
    dataset,
    target_col,
    model,
    model_name,
    domain,
    ml_model="xgboost",
    embedding_type="cls",
    cm2_table_flag=0,
):
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
    elif isinstance(model, _get_haetae_cls()):
        dataset_X = JSONDataset(path=None, tokenizer=tokenizer)
        dataset_X.data = preprocessed_dataset
    elif isinstance(model, BertForMaskedLM):
        dataset_X = BertDataset(preprocessed_dataset)
    elif isinstance(model, TapasForMaskedLM):
        dataset_X = TapasDataset(preprocessed_dataset)
    elif is_cm2_mask_model(model):
        for i, row in enumerate(dataset):
            row_x = remove_target_column(row, target_col)
            try:
                emb = get_cm2_row_embedding(
                    model,
                    row_x,
                    table_flag=cm2_table_flag,
                    embedding_type=embedding_type,
                )
            except Exception as e:
                print(f"⚠️  CM2 skip row {i}: {e}")
                continue
            embeddings.append(emb)
            labels.append(row[target_col])
        if not embeddings:
            print("CM2: no rows produced embeddings")
            return None
        X = np.stack(embeddings)
        y = np.array(labels)
        if ml_model == "tabpfn":
            f1 = run_row_classification_tabpfn(X, y, test_size=0.2)
        else:
            f1 = run_row_classification(X, y, model_type=ml_model)
        return f1
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

def run_repeated_classification(
    dataset,
    target_col,
    model,
    model_name,
    domain,
    ml_model="xgboost",
    n_runs=8,
    embedding_type="cls",
    cm2_table_flag=0,
):
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
        
        f1 = run_cls_classification(
            dataset,
            target_col,
            model,
            model_name,
            domain,
            ml_model,
            embedding_type,
            cm2_table_flag=cm2_table_flag,
        )
        if f1 is None:
            print("F1: None (skipped)")
            continue
        f1_scores.append(f1)
        print(f"F1: {f1:.4f}")
    
    # Calculate statistics
    if not f1_scores:
        print("\n  Results: no successful runs\n\n")
        return {'mean': None, 'std': None, 'scores': []}

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
    # Load BERT tokenizer (local or online per RUN_ONLINE)
    tokenizer = BertTokenizer.from_pretrained(_bert_name, **_bert_kw)

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

def find_navi_checkpoint(base_dir, domain, seed=None, epoch=None):
    """
    Find NAVI model checkpoint path.
    
    Args:
        base_dir: Base models directory (e.g., './models')
        domain: Domain name (e.g., 'movie', 'product')
        seed: Training seed (None to find any, or specific seed)
        epoch: Epoch number (None = use config CHECKPOINT_EPOCH)
    
    Returns:
        str: Checkpoint path or None if not found
    """
    import glob
    if epoch is None:
        epoch = _checkpoint_epoch
    
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


def find_epoch_path_for_navi(base_path):
    """Find checkpoint epoch directory using glob pattern (module-level for reuse)."""
    model_name = Path(base_path).name
    p = resolve_navi_checkpoint_path_with_fallback(model_name, base_path, _checkpoint_epoch)
    return str(p) if p else None


def load_baseline_models(tokenizer, navi_seed=None):
    """
    Load baseline models (BERT, HAETAE, TAPAS, NAVI).
    
    Args:
        tokenizer: Tokenizer instance
        navi_seed: Training seed for NAVI models (None = use default/seed42)
    """
    models = {}
    
    # Config for HAETAE (BERT base)
    bconfig = BertConfig.from_pretrained(_bert_name, **_bert_kw)

    ep = _checkpoint_epoch
    # Baseline models - using trained versions
    models['bert_movie'] = BertForMaskedLM.from_pretrained(f'./models/bert_movie/epoch_{ep}', local_files_only=True)
    models['bert_movie'] = models['bert_movie'].to(device)
    models['bert_movie'].eval()

    models['bert_product'] = BertForMaskedLM.from_pretrained(f'./models/bert_product/epoch_{ep}', local_files_only=True)
    models['bert_product'] = models['bert_product'].to(device)
    models['bert_product'].eval()

    models['haetae_product'] = _get_haetae_cls()(bconfig, tokenizer, f"./models/haetae_product/epoch_{ep}")
    models['haetae_product'] = models['haetae_product'].to(device)
    models['haetae_product'].eval()

    models['haetae_movie'] = _get_haetae_cls()(bconfig, tokenizer, f"./models/haetae_movie/epoch_{ep}")
    models['haetae_movie'] = models['haetae_movie'].to(device)
    models['haetae_movie'].eval()

    models['tapas_movie'] = TapasForMaskedLM.from_pretrained(f'./models/tapas_movie/epoch_{ep}', local_files_only=True)
    models['tapas_movie'] = models['tapas_movie'].to(device)
    models['tapas_movie'].eval()

    models['tapas_product'] = TapasForMaskedLM.from_pretrained(f'./models/tapas_product/epoch_{ep}', local_files_only=True)
    models['tapas_product'] = models['tapas_product'].to(device)
    models['tapas_product'].eval()

    # NAVI models - use navi_movie and navi_product
    # NAVI Movie - use navi_movie
    navi_movie_base = './models/navi_movie'
    navi_movie_path = find_epoch_path_for_navi(navi_movie_base)
    if navi_movie_path:
        models['navi_movie'] = NaviForMaskedLM(navi_movie_path)
        models['navi_movie'] = models['navi_movie'].to(device)
        models['navi_movie'].eval()
        print(f"✓ Loaded NAVI Movie from: {navi_movie_path}")
    else:
        print(f"⚠️  NAVI Movie model not found: {navi_movie_base}/*epoch_{_checkpoint_epoch}")
    
    # NAVI Product
    navi_product_base = './models/navi_product'
    navi_product_path = find_epoch_path_for_navi(navi_product_base)
    if navi_product_path:
        models['navi_product'] = NaviForMaskedLM(navi_product_path)
        models['navi_product'] = models['navi_product'].to(device)
        models['navi_product'].eval()
        print(f"✓ Loaded NAVI Product from: {navi_product_path}")
    else:
        print(f"⚠️  NAVI Product model not found: {navi_product_base}/*epoch_{_checkpoint_epoch}")
    
    return models

def load_b200_models(tokenizer, navi_seed=None):
    """
    Load baseline models from models_b200/ directory (BERT, HAETAE, TAPAS, NAVI).
    
    Args:
        tokenizer: Tokenizer instance
        navi_seed: Training seed for NAVI models (None = use default/seed42)
    """
    models = {}
    
    # Config for HAETAE (BERT base)
    bconfig = BertConfig.from_pretrained(_bert_name, **_bert_kw)

    ep = _checkpoint_epoch
    # Baseline models - using trained versions from models_b200/
    models['bert_movie'] = BertForMaskedLM.from_pretrained(f'./models_b200/bert_movie/epoch_{ep}', local_files_only=True)
    models['bert_product'] = BertForMaskedLM.from_pretrained(f'./models_b200/bert_product/epoch_{ep}', local_files_only=True)
    models['haetae_product'] = _get_haetae_cls()(bconfig, tokenizer, f"./models_b200/haetae_product/epoch_{ep}")
    models['haetae_movie'] = _get_haetae_cls()(bconfig, tokenizer, f"./models_b200/haetae_movie/epoch_{ep}")
    models['tapas_movie'] = TapasForMaskedLM.from_pretrained(f'./models_b200/tapas_movie/epoch_{ep}', local_files_only=True)
    models['tapas_product'] = TapasForMaskedLM.from_pretrained(f'./models_b200/tapas_product/epoch_{ep}', local_files_only=True)
    models['bert_movie'] = models['bert_movie'].to(device)
    models['bert_movie'].eval()
    models['bert_product'] = models['bert_product'].to(device)
    models['bert_product'].eval()
    models['haetae_product'] = models['haetae_product'].to(device)
    models['haetae_product'].eval()
    models['haetae_movie'] = models['haetae_movie'].to(device)
    models['haetae_movie'].eval()
    models['tapas_movie'] = models['tapas_movie'].to(device)
    models['tapas_movie'].eval()
    models['tapas_product'] = models['tapas_product'].to(device)
    models['tapas_product'].eval()

    # NAVI models
    navi_movie_base = './models_b200/navi_movie'
    navi_movie_path = find_epoch_path_for_navi(navi_movie_base)
    if navi_movie_path:
        models['navi_movie'] = NaviForMaskedLM(navi_movie_path)
        models['navi_movie'] = models['navi_movie'].to(device)
        models['navi_movie'].eval()
        print(f"✓ Loaded NAVI Movie from: {navi_movie_path}")
    else:
        print(f"⚠️  NAVI Movie model not found: {navi_movie_base}/*epoch_{_checkpoint_epoch}")
    navi_product_base = './models_b200/navi_product'
    navi_product_path = find_epoch_path_for_navi(navi_product_base)
    if navi_product_path:
        models['navi_product'] = NaviForMaskedLM(navi_product_path)
        models['navi_product'] = models['navi_product'].to(device)
        models['navi_product'].eval()
        print(f"✓ Loaded NAVI Product from: {navi_product_path}")
    else:
        print(f"⚠️  NAVI Product model not found: {navi_product_base}/*epoch_{_checkpoint_epoch}")
    
    return models

def _navi_ablation_mode_for_load(ablation):
    """Model only has woGHA/woGHC; woSSI, woMSM, woESA are dataset/training-only -> load as full."""
    return ablation if ablation in ('woGHA', 'woGHC') else 'full'

def load_ablation_models(tokenizer):
    """Load ablation models (woSSI: dataset-only, woGHA/woGHC: model ablations, woMSM/woESA: training-only)."""
    models = {}
    bconfig = BertConfig.from_pretrained(_bert_name, **_bert_kw)
    domains = ['movie', 'product']
    ablation_values = ['woGHA', 'woMSM', 'woESA']

    for domain in domains:
        for ablation in ablation_values:
            base_path = f'./models/navi_{domain}_{ablation}'
            epoch_path = find_epoch_path_for_navi(base_path)
            if epoch_path:
                model_name = f'navi_{domain}_{ablation}'
                model_ablation = _navi_ablation_mode_for_load(ablation)
                models[model_name] = NaviForMaskedLM(epoch_path, ablation_mode=model_ablation)
                models[model_name] = models[model_name].to(device)
                models[model_name].eval()
                print(f"✓ Loaded {model_name} from: {epoch_path}")
            else:
                print(f"⚠️  Model not found: {base_path}/*epoch_{_checkpoint_epoch}")
    
    return models

def load_prototyping_models(tokenizer, domain):
    """Load prototyping models: default navi_{domain} plus woGHA and woGHC ablations."""
    models = {}
    bconfig = BertConfig.from_pretrained(_bert_name, **_bert_kw)
    domain_lower = domain.lower()
    # 1. Default navi_{domain}
    base_path = f'./models/navi_{domain_lower}'
    epoch_path = find_epoch_path_for_navi(base_path)
    if epoch_path:
        model_name = f'navi_{domain_lower}'
        models[model_name] = NaviForMaskedLM(epoch_path)
        models[model_name] = models[model_name].to(device)
        models[model_name].eval()
        print(f"✓ Loaded {model_name} from: {epoch_path}")
    else:
        print(f"⚠️  Model not found: {base_path}/*epoch_{_checkpoint_epoch}")

    # 2. woGHA and woGHC
    for ablation in ['woGHA', 'woGHC']:
        base_path = f'./models/navi_{domain_lower}_{ablation}'
        epoch_path = find_epoch_path_for_navi(base_path)
        if epoch_path:
            model_name = f'navi_{domain_lower}_{ablation}'
            models[model_name] = NaviForMaskedLM(epoch_path, ablation_mode=ablation)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {epoch_path}")
        else:
            print(f"⚠️  Model not found: {base_path}/*epoch_{_checkpoint_epoch}")

    return models

def load_rebuttal_models(tokenizer, domain):
    """Load rebuttal models: entropy threshold, tau variants, and gradient accumulation variants"""
    models = {}
    domain_lower = domain.lower()
    # Entropy threshold variants
    ethresh_values = ['10_90', '40_60', '50_50']
    for ethresh in ethresh_values:
        base_path = f'./models/navi_{domain_lower}_ethresh{ethresh}'
        model_path = find_epoch_path_for_navi(base_path)
        if model_path:
            model_name = f'navi_{domain_lower}_ethresh{ethresh}'
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {model_path}")
        else:
            print(f"⚠️  Model not found: {base_path}/*epoch_{_checkpoint_epoch}")
    
    # Tau variants
    tau_values = ['0p07_0p07', '0p13_0p13']
    for tau in tau_values:
        base_path = f'./models/navi_{domain_lower}_tau{tau}'
        model_path = find_epoch_path_for_navi(base_path)
        if model_path:
            model_name = f'navi_{domain_lower}_tau{tau}'
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {model_path}")
        else:
            print(f"⚠️  Model not found: {base_path}/*epoch_{_checkpoint_epoch}")
    
    # Gradient accumulation variants
    ga_values = ['1', '4']
    for ga in ga_values:
        base_path = f'./models/navi_{domain_lower}_ga{ga}'
        model_path = find_epoch_path_for_navi(base_path)
        if model_path:
            model_name = f'navi_{domain_lower}_ga{ga}'
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {model_path}")
        else:
            print(f"⚠️  Model not found: {base_path}/*epoch_{_checkpoint_epoch}")
    
    return models

def load_navi_models(tokenizer):
    """Load standard NAVI models (navi_movie, navi_product)"""
    models = {}
    # NAVI Movie
    base_path = './models/navi_movie'
    navi_movie_path = find_epoch_path_for_navi(base_path)
    if navi_movie_path:
        models['navi_movie'] = NaviForMaskedLM(navi_movie_path)
        models['navi_movie'] = models['navi_movie'].to(device)
        models['navi_movie'].eval()
        print(f"✓ Loaded NAVI Movie from: {navi_movie_path}")
    else:
        print(f"⚠️  NAVI Movie model not found: {base_path}/*epoch_{_checkpoint_epoch}")
    
    # NAVI Product
    base_path = './models/navi_product'
    navi_product_path = find_epoch_path_for_navi(base_path)
    if navi_product_path:
        models['navi_product'] = NaviForMaskedLM(navi_product_path)
        models['navi_product'] = models['navi_product'].to(device)
        models['navi_product'].eval()
        print(f"✓ Loaded NAVI Product from: {navi_product_path}")
    else:
        print(f"⚠️  NAVI Product model not found: {base_path}/*epoch_{_checkpoint_epoch}")
    
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
        epoch: Epoch number (1, 2, 3, or 4). If None, uses config CHECKPOINT_EPOCH.
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
        model_name = f'navi_{domain_lower}_epoch{_checkpoint_epoch}'
    
    # Find epoch directory
    target_epoch = epoch if epoch is not None else _checkpoint_epoch
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
    # 1. Default model
    base_path = f'./models/navi_{domain_lower}'
    model_path = find_epoch_path_for_navi(base_path)
    if model_path:
        model_name = f'navi_{domain_lower}'
        models[model_name] = NaviForMaskedLM(model_path)
        models[model_name] = models[model_name].to(device)
        models[model_name].eval()
        print(f"✓ Loaded {model_name} from: {model_path}")
    else:
        print(f"⚠️  Model not found: {base_path}/*epoch_{_checkpoint_epoch}")
    
    # 2-4. Entropy threshold variants
    ethresh_values = ['10_90', '40_60', '50_50']
    for ethresh in ethresh_values:
        base_path = f'./models/navi_{domain_lower}_ethresh{ethresh}'
        model_path = find_epoch_path_for_navi(base_path)
        if model_path:
            model_name = f'navi_{domain_lower}_ethresh{ethresh}'
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {model_path}")
        else:
            print(f"⚠️  Model not found: {base_path}/*epoch_{_checkpoint_epoch}")
    
    # 5-6. Temperature variants
    tau_values = ['0p07_0p07', '0p13_0p13']
    for tau in tau_values:
        base_path = f'./models/navi_{domain_lower}_tau{tau}'
        model_path = find_epoch_path_for_navi(base_path)
        if model_path:
            model_name = f'navi_{domain_lower}_tau{tau}'
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {model_path}")
        else:
            print(f"⚠️  Model not found: {base_path}/*epoch_{_checkpoint_epoch}")
    
    # 7-8. Gradient accumulation variants (Neg batch size)
    ga_values = ['1', '4']  # ga1 = batch size 32, ga4 = batch size 128
    for ga in ga_values:
        base_path = f'./models/navi_{domain_lower}_ga{ga}'
        model_path = find_epoch_path_for_navi(base_path)
        if model_path:
            model_name = f'navi_{domain_lower}_ga{ga}'
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {model_path}")
        else:
            print(f"⚠️  Model not found: {base_path}/*epoch_{_checkpoint_epoch}")
    
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
    ml_models = ["xgboost", "lr", "tabpfn"]
    
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

def evaluate_cm2(
    data,
    target_col,
    domain,
    n_runs=8,
    embedding_type="cls",
    cm2_checkpoint=None,
    cm2_table_flag=0,
    cm2_model_type="mask",
    cm2_pretrain_checkpoint=None,
    cm2_num_layer=3,
    cm2_schema_csv=None,
    cm2_schema_target=None,
):
    """
    Frozen CM2 (CM2ForMask pretrain) row embeddings + sklearn classifier,
    same protocol as BERT/NAVI CLS in evaluate_baselines.
    """
    domain_key = domain.lower()
    cm2_key = f"cm2_{domain_key}"
    print(f"\n{domain} - CM2 frozen row embeddings")

    results: dict = {}
    ml_models = ["tabpfn"]

    try:
        if cm2_checkpoint:
            if cm2_model_type == "classifier":
                model = load_cm2_classifier_model(
                    cm2_checkpoint,
                    device=device,
                    pretrain_dir=cm2_pretrain_checkpoint,
                    num_layer=cm2_num_layer,
                )
                if cm2_schema_csv:
                    configure_cm2_classifier_schema(
                        model,
                        schema_csv=cm2_schema_csv,
                        target=cm2_schema_target or target_col,
                    )
            else:
                model = load_cm2_mask_model(cm2_checkpoint, device)
        else:
            loaded = load_cm2_models_for_domain(domain_key, device=device)
            if cm2_key not in loaded:
                print(f"⚠️  {cm2_key} not loaded (missing checkpoint or incomplete)")
                return results
            model = loaded[cm2_key]
    except Exception as e:
        _warn_failed(f"CM2 load ({domain})", e)
        return results

    results["cm2"] = {}
    for ml_model in ml_models:
        try:
            result = run_repeated_classification(
                data,
                target_col,
                model,
                "cm2",
                domain,
                ml_model=ml_model,
                n_runs=n_runs,
                embedding_type=embedding_type,
                cm2_table_flag=cm2_table_flag,
            )
        except Exception as e:
            _warn_failed(f"CM2/{ml_model} ({domain})", e)
            result = _failed_result(e)
        results["cm2"][f"{ml_model}_{domain_key}"] = result

    del model
    clear_memory()
    return results

def evaluate_ablations(data, target_col, models, domain, n_runs=8, embedding_type="cls"):
    """Evaluate selected ablation models."""
    print(f"\n{domain} - Ablations")
    
    results = {}
    ml_models = ["xgboost", "lr", "tabpfn"]
    domain_lower = domain.lower()
    target_model_names = [
        f"navi_{domain_lower}_woGHA",
        f"navi_{domain_lower}_woMSM",
        f"navi_{domain_lower}_woESA",
    ]
    missing = [name for name in target_model_names if name not in models]
    for name in missing:
        print(f"⚠️  Skipping missing model: {name}")
    domain_models = {
        name: models[name]
        for name in target_model_names
        if name in models
    }
    
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

def evaluate_navi_hyperparam_sweep(
    data, target_col, tokenizer, domain, n_runs=8, embedding_type="cls", max_loaded_variants=1,
):
    """Evaluate all `./models/navi_{domain}` and `./models/navi_{domain}_*` checkpoints for this domain."""
    print(f"\n{domain} - NAVI hyperparameter sweep (max_loaded_variants={max_loaded_variants})")
    results = {}
    ml_models = ["xgboost", "lr", "tabpfn"]
    domain_key = domain.lower()
    discovered = discover_navi_domain_model_dirs(domain)
    pending = []
    for model_name, base_dir in discovered:
        epoch_path = resolve_navi_checkpoint_path_with_fallback(model_name, base_dir, _checkpoint_epoch)
        if epoch_path is None:
            print(f"⚠️  Skipping {model_name}: no checkpoint at epoch_{_checkpoint_epoch}")
            continue
        pending.append((model_name, epoch_path))

    for wave in iter_chunks(pending, max_loaded_variants):
        wave_models = {}
        for model_name, epoch_path in wave:
            mode = navi_ablation_mode_for_model_name(model_name)
            print(f"✓ Loading {model_name} from {epoch_path}")
            wave_models[model_name] = NaviForMaskedLM(str(epoch_path), ablation_mode=mode).to(device).eval()
        for model_name, model in wave_models.items():
            print(f"\n{model_name}")
            results[model_name] = {}
            for ml_model in ml_models:
                result = run_repeated_classification(
                    data, target_col, model, model_name, domain,
                    ml_model=ml_model, n_runs=n_runs, embedding_type=embedding_type
                )
                results[model_name][f"{ml_model}_{domain_key}"] = result
        del wave_models
        clear_memory()
    return results


def evaluate_hyperparams(data, target_col, tokenizer, domain, n_runs=8, embedding_type="cls", max_loaded_variants=1):
    """Same sweep as hyperparam_sensitivity; kept for backward compatibility."""
    return evaluate_navi_hyperparam_sweep(
        data, target_col, tokenizer, domain, n_runs, embedding_type, max_loaded_variants,
    )


def evaluate_hyperparam_sensitivity(data, target_col, tokenizer, domain, n_runs=8, embedding_type="cls", max_loaded_variants=1):
    """Same sweep as hyperparams; kept for backward compatibility."""
    return evaluate_navi_hyperparam_sweep(
        data, target_col, tokenizer, domain, n_runs, embedding_type, max_loaded_variants,
    )

def evaluate_navi_embeddings(data, target_col, tokenizer, domain, n_runs=8, embedding_type="cls"):
    """Evaluate NAVI embeddings approach"""
    print(f"\n{domain} - NAVI Embeddings")
    
    results = {}
    ml_models = ["xgboost", "lr", "tabpfn"]
    
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

def evaluate_tablevectorizer(
    data, target_col, domain, n_runs=8, high_cardinality_encoder="text"
):
    """Evaluate TableVectorizer features approach (high-cardinality: TextEncoder or StringEncoder)."""
    enc_label = "TextEncoder" if high_cardinality_encoder == "text" else "StringEncoder"
    print(f"\n{domain} - TableVectorizer + {enc_label}")

    results = {}
    ml_models = ["xgboost", "lr", "tabpfn"]

    # Extract features using TableVectorizer
    print(f"Extracting features with TableVectorizer (high_cardinality={high_cardinality_encoder})...")
    try:
        X, y = extract_features_tablevectorizer(
            data, target_col, high_cardinality_encoder=high_cardinality_encoder
        )
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
    ml_models = ["xgboost", "lr", "tabpfn"]
    
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
    
    results = {}
    ml_models = ["xgboost", "lr", "tabpfn"]
    
    # Load NAVI model from baseline models
    models = load_baseline_models(tokenizer)
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

def evaluate_rebuttal(
    data,
    target_col,
    tokenizer,
    domain,
    n_runs=8,
    embedding_type="cls",
    high_cardinality_encoder="text",
):
    """Evaluate rebuttal models (entropy threshold, tau variants, gradient accumulation)"""
    print(f"\n{domain} - Rebuttal Models")
    
    results = {}
    ml_models = ["xgboost", "lr", "tabpfn"]
    
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
    tablevectorizer_results = evaluate_tablevectorizer(
        data,
        target_col,
        domain,
        n_runs=n_runs,
        high_cardinality_encoder=high_cardinality_encoder,
    )
    results.update(tablevectorizer_results)
    
    return results

def evaluate_navi(data, target_col, tokenizer, domain, n_runs=8, embedding_type="cls"):
    """Evaluate standard NAVI models (navi_movie, navi_product)"""
    print(f"\n{domain} - NAVI Models")
    
    results = {}
    ml_models = ["xgboost", "lr", "tabpfn"]
    
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
        
        model_path = find_epoch_path_for_navi(base_path)
        if model_path:
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {model_path}")
        else:
            print(f"⚠️  Model not found: {base_path}/*epoch_{_checkpoint_epoch}")
    
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
    ml_models = ["xgboost", "lr", "tabpfn"]
    
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

def _icml_entropy_ablation_variant_map():
    """
    Map table row labels -> model folder suffix.

    Model folders follow: navi_{domain}_esaRouting{suffix}
    """
    return [
        ("Default", "entropy"),
        ("AllLow", "all_low"),
        ("AllHigh", "all_high"),
        ("Random", "random"),
    ]


def _fmt_cell_or_na(v) -> str:
    if v is None:
        return "---"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def _write_icml_entropy_ablation_table(out_path, table_rows):
    """
    table_rows: List[Tuple[row_label, imp_movie, cls_movie, imp_product, cls_product]]
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header_1 = "            |    Movie   | Product"
    header_2 = "            | Imp | Cls | Imp | Cls"
    bar = "-" * 34

    lines = [header_1, header_2, bar]
    for (row_label, imp_m, cls_m, imp_p, cls_p) in table_rows:
        lines.append(
            f"{row_label}|{_fmt_cell_or_na(imp_m)}|{_fmt_cell_or_na(cls_m)}|"
            f"{_fmt_cell_or_na(imp_p)}|{_fmt_cell_or_na(cls_p)}"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✅ ICML entropy ablation table written to: {out_path}")


def evaluate_icml_entropy_ablation(tokenizer, max_samples=1000, run_id=None):
    """
    ICML entropy routing ablation:
    - Imp: masked value prediction accuracy (epoch=5)
    - Cls: row-classification F1 (xgboost, cls embedding, single split)
    """
    import numpy as np
    from datetime import datetime as _dt

    from dataset.collator import CollatorForMaskedPrediction
    from experiments.masked_prediction.mp_utils import evaluate_masked_prediction, load_data as mp_load_data

    rid = run_id or _dt.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path("experiments/icml_entropy_ablation") / f"icml_entropy_ablation_{rid}.txt"

    domains = ["Movie", "Product"]
    variant_map = _icml_entropy_ablation_variant_map()

    collator = CollatorForMaskedPrediction(tokenizer=tokenizer)

    domain_mp_datasets = {}
    domain_cls_datasets = {}

    for domain in domains:
        domain_lower = domain.lower()

        # --- Imp dataset (masked prediction) ---
        mp_path = Path(DATA_ROOT) / "cleaned" / domain / "test" / f"WDC_{domain_lower}_for_mp.jsonl"
        raw_mp = mp_load_data(str(mp_path))
        print(f"\n[{domain}] Loaded {len(raw_mp)} masked-prediction rows from: {mp_path}")

        random.seed(42)
        if len(raw_mp) > max_samples:
            raw_mp = random.sample(raw_mp, max_samples)
            print(f"[{domain}] Sampled {max_samples} rows (seed=42) for Imp")
        else:
            print(f"[{domain}] Using all {len(raw_mp)} rows for Imp")

        json_data = [(i, row) for i, row in enumerate(raw_mp)]
        navi_dataset = NaviDataset(json_data)
        domain_mp_datasets[domain] = navi_dataset

        # --- Cls dataset (row classification) ---
        if domain == "Product":
            cls_path = Path(DATA_ROOT) / "cleaned" / "Product" / "test" / "WDC_product_for_cls.jsonl"
            raw_cls = load_data(str(cls_path))
            raw_cls = verify_target_columns_present(raw_cls, target_columns=["category"])
            raw_cls = preprocess_wdc_product(raw_cls)
            raw_cls = stratified_sample(raw_cls, "category", sample_size=max_samples)
            target_col = "category"
        else:
            cls_path = Path(DATA_ROOT) / "cleaned" / "Movie" / "test" / "WDC_movie_for_cls.jsonl"
            raw_cls = load_data(str(cls_path))
            raw_cls = verify_target_columns_present(raw_cls, target_columns=["genres"])
            raw_cls = preprocess_wdc_movie(raw_cls)
            raw_cls = stratified_sample(raw_cls, "genres", sample_size=max_samples)
            target_col = "genres"

        print(f"[{domain}] Classification dataset size after sampling: {len(raw_cls)}")
        domain_cls_datasets[domain] = (raw_cls, target_col)

    table_rows = []
    for row_label, suffix in variant_map:
        imp_movie = cls_movie = imp_product = cls_product = None

        for domain in domains:
            domain_lower = domain.lower()
            base_dir = Path("./models") / f"navi_{domain_lower}_esaRouting{suffix}"
            epoch_path = find_epoch_path_for_navi(str(base_dir))

            if not epoch_path:
                print(f"⚠️  Missing model for {domain} {row_label}: {base_dir}/*epoch_{_checkpoint_epoch}")
                continue

            model_name = f"navi_{domain_lower}_esaRouting{suffix}"
            print(f"\n[{domain}] Loading {model_name} from: {epoch_path}")
            model = NaviForMaskedLM(epoch_path).to(device).eval()

            try:
                imp_acc = evaluate_masked_prediction(
                    domain_mp_datasets[domain],
                    model,
                    tokenizer,
                    collator,
                    epoch=5,
                )

                cls_data, target_col = domain_cls_datasets[domain]

                random.seed(42)
                np.random.seed(42)
                f1 = run_cls_classification(
                    dataset=cls_data,
                    target_col=target_col,
                    model=model,
                    model_name=model_name,
                    domain=domain_lower,
                    ml_model="xgboost",
                    embedding_type="cls",
                )
            except Exception as e:
                print(f"⚠️  Evaluation failed for {model_name}: {e}")
                imp_acc, f1 = None, None
            finally:
                del model
                clear_memory()

            if domain == "Movie":
                imp_movie, cls_movie = imp_acc, f1
            else:
                imp_product, cls_product = imp_acc, f1

        table_rows.append((row_label, imp_movie, cls_movie, imp_product, cls_product))

    _write_icml_entropy_ablation_table(out_path, table_rows)
    return out_path

def main():
    parser = argparse.ArgumentParser(description='Run row classification experiments')
    parser.add_argument('--mode', choices=[
        'lm_encoders', 'fe_pipelines', 'ablations', 'prototyping', 'b200',
        'hyperparams', 'hyperparam_sensitivity', 'icml_entropy_ablation', 'cm2',
        'cm2_finetune_strict_csv',
    ],
                       required=True, help='Mode to run: lm_encoders, fe_pipelines, ablations, prototyping, b200, cm2 (frozen CM2 row emb + XGB), cm2_finetune_strict_csv (CM2 finetuned classifier: encoder CLS embedding + sklearn on heldout CSV/JSONL), or hyperparams/hyperparam_sensitivity (sweep all ./models/navi_{domain}*)')
    parser.add_argument('--domain', choices=['Movie', 'Product'], 
                       required=True, help='Domain to evaluate on')
    parser.add_argument("--embedding_type", type=str, default="cls", choices=["cls", "mean"], 
                       help="Type of embedding to use")
    parser.add_argument('--max_loaded_variants', type=int, default=4, metavar='K',
                       help='NAVI checkpoints to keep on GPU at once during hyperparam sweeps (default 1)')
    parser.add_argument('--run_id', type=str, default=None,
                       help='Optional id for output JSON filename (e.g. shared with masked_prediction + table export)')
    parser.add_argument(
        '--tablevectorizer_encoder',
        type=str,
        default='text',
        choices=['text', 'string'],
        help='High-cardinality encoder for TableVectorizer in fe_pipelines: text (TextEncoder) or string (StringEncoder/TF-IDF+SVD)',
    )
    parser.add_argument(
        '--product_column_ablation',
        type=str,
        default='allcol',
        choices=['allcol', 'text_only', 'half_text'],
        help='Product only: allcol | text_only (drop predefined numeric cols) | half_text (numeric + half of text cols)',
    )
    parser.add_argument(
        '--cm2_checkpoint',
        type=str,
        default=None,
        help='Optional directory with pytorch_model.bin for CM2 mode; default uses CM2_MOVIE_CHECKPOINT / CM2_PRODUCT_CHECKPOINT or baselines/CM2/mask_v1_*',
    )
    parser.add_argument(
        '--cm2_table_flag',
        type=int,
        default=0,
        help='table_flag for CM2 feature extractor when multiple pretrain tables exist (default 0)',
    )
    parser.add_argument(
        '--heldout_csv',
        type=str,
        default=None,
        help='For cm2_finetune_strict_csv: heldout CSV or JSONL (e.g., .../strict/WDC_*_heldout_test.csv or .jsonl). Target column is kept for labels only; it is stripped before the encoder.',
    )
    parser.add_argument(
        '--cm2_pretrain_checkpoint',
        type=str,
        default=None,
        help='Optional pretrain checkpoint for loading weights-only CM2 finetuned classifier checkpoints.',
    )
    parser.add_argument(
        '--cm2_num_layer',
        type=int,
        default=3,
        help='CM2 classifier num_layer used when loading cm2_finetune_strict_csv checkpoints (default 3).',
    )
    parser.add_argument(
        '--cm2_schema_csv',
        type=str,
        default=None,
        help='Finetune-train CSV for CM2 classifier schema (cat/num/bin). Default: WDC_{domain}_for_cls_finetune_train.csv in the same directory as --heldout_csv.',
    )
    parser.add_argument(
        '--top_k_labels',
        type=int,
        default=None,
        metavar='K',
        help='If set, keep only rows whose label is in the K most frequent classes in the loaded data (e.g. 20 for top-20 categories).',
    )

    args = parser.parse_args()

    if args.domain != 'Product' and args.product_column_ablation != 'allcol':
        parser.error('--product_column_ablation may only differ from allcol when --domain Product')
    
    print(f"Evaluating {args.mode} on {args.domain} domain")
    
    if args.mode == 'icml_entropy_ablation':
        # Note: this mode always evaluates both Movie + Product.
        from experiments.icml_entropy_ablation import run_icml_entropy_ablation

        run_icml_entropy_ablation(
            max_samples=1000,
            imp_runs=5,
            xgb_runs=5,
            run_id=args.run_id,
        )
        return

    # Load and process data for the specified domain only
    if args.mode == 'cm2_finetune_strict_csv':
        if not args.heldout_csv:
            parser.error('--heldout_csv is required when --mode cm2_finetune_strict_csv')
        if not args.cm2_checkpoint:
            parser.error('--cm2_checkpoint is required when --mode cm2_finetune_strict_csv')
        heldout_path = Path(args.heldout_csv).resolve()
        schema_csv = args.cm2_schema_csv
        if not schema_csv:
            schema_csv = str(_default_cm2_schema_csv(heldout_path, args.domain))
        if not Path(schema_csv).is_file():
            parser.error(
                f"CM2 schema CSV not found: {schema_csv}. "
                "Pass --cm2_schema_csv to the finetune-train CSV used for CM2 typing (cat/num/bin)."
            )
        args.cm2_schema_csv = schema_csv
        kind = "JSONL" if heldout_path.suffix.lower() == ".jsonl" else "CSV"
        print(f"Loading heldout {kind}: {args.heldout_csv}")
        data = _load_heldout_finetune_strict(args.heldout_csv)
        if args.domain == 'Product':
            target_col = 'category'
            print("Verifying target columns in product heldout...")
            data = verify_target_columns_present(data, target_columns=['category'])
        else:
            target_col = 'genres'
            print("Verifying target columns in movie heldout...")
            data = verify_target_columns_present(data, target_columns=['genres'])
        print(f"Heldout rows: {len(data)}")
        print(f"CM2 schema CSV: {args.cm2_schema_csv}")
        print_class_distribution(data, target_col, f"WDC {args.domain} Heldout ({kind})")
    elif args.domain == 'Product':
        # Load Product classification dataset
        wdc_product_data = load_data(str(Path(DATA_ROOT) / "cleaned" / "Product" / "test" / "WDC_product_for_cls.jsonl"))
        
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

        data, ablation_info = apply_product_column_ablation(
            data, target_col, args.product_column_ablation
        )
        print("\nProduct column ablation (predefined numeric / text split):")
        print(f"  numeric_keys: {ablation_info['numeric_keys']}")
        print(f"  text_keys ({len(ablation_info['text_keys'])}): {ablation_info['text_keys']}")
        print(f"  half_text_kept ({len(ablation_info['half_text_kept'])}): {ablation_info['half_text_kept']}")
        print(f"  active mode: {args.product_column_ablation}\n")
    else:  # Movie
        # Load Movie classification dataset
        wdc_movie_data = load_data(str(Path(DATA_ROOT) / "cleaned" / "Movie" / "test" / "WDC_movie_for_cls.jsonl"))
        
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

    if args.top_k_labels is not None:
        n_before = len(data)
        data = _filter_rows_top_k_labels(data, target_col, args.top_k_labels)
        if not data:
            parser.error(
                f'--top_k_labels={args.top_k_labels} removed all rows ({n_before} before filter).'
            )
        print(
            f"top_k_labels={args.top_k_labels}: kept {len(data)} / {n_before} rows "
            f"(top-{args.top_k_labels} classes by frequency in loaded data)"
        )
        print_class_distribution(
            data, target_col, f"After top-{args.top_k_labels} label filter"
        )

    # Run experiments based on mode
    results = {}
    
    if args.mode == 'lm_encoders':
        # Language model encoders: bert, tapas, haetae, navi
        models = load_baseline_models(tokenizer)
        # results = evaluate_baselines(data, target_col, models, args.domain.lower(), embedding_type=args.embedding_type)
        results = evaluate_navi(
            data,
            target_col,
            tokenizer,
            args.domain.lower(),
            embedding_type=args.embedding_type,
        )
        
    elif args.mode == 'fe_pipelines':
        # Feature engineering pipelines: tablevectorizer, navi_concat
        # tablevectorizer_results = evaluate_tablevectorizer(
        #     data,
        #     target_col,
        #     args.domain.lower(),
        #     high_cardinality_encoder=args.tablevectorizer_encoder,
        # )
        # results.update(tablevectorizer_results)
        
        # concat_results = evaluate_concatenated_navi(
        #     data, target_col, tokenizer, args.domain.lower(), embedding_type=args.embedding_type
        # )
        # results.update(concat_results)
        results = evaluate_raw_features(
            data,
            target_col,
            args.domain.lower(),
        )
        
    elif args.mode == 'ablations':
        # Ablations: wo_PER, wo_MSM, wo_ESA, wo_GHA, wo_GHC
        models = load_ablation_models(tokenizer)
        results = evaluate_ablations(data, target_col, models, args.domain.lower(), embedding_type=args.embedding_type)

    elif args.mode == 'prototyping':
        # Prototyping: default navi_{domain} plus woGHA, woGHC
        models = load_prototyping_models(tokenizer, args.domain)
        results = evaluate_ablations(data, target_col, models, args.domain.lower(), embedding_type=args.embedding_type)

    elif args.mode == 'b200':
        # b200: models from models_b200/ directory
        models = load_b200_models(tokenizer)
        results = evaluate_baselines(data, target_col, models, args.domain.lower(), embedding_type=args.embedding_type)

    elif args.mode == 'hyperparams':
        results = evaluate_hyperparams(
            data, target_col, tokenizer, args.domain,
            embedding_type=args.embedding_type,
            max_loaded_variants=args.max_loaded_variants,
        )

    elif args.mode == 'hyperparam_sensitivity':
        results = evaluate_hyperparam_sensitivity(
            data, target_col, tokenizer, args.domain,
            embedding_type=args.embedding_type,
            max_loaded_variants=args.max_loaded_variants,
        )

    elif args.mode == 'cm2':
        results = evaluate_cm2(
            data,
            target_col,
            args.domain,
            n_runs=8,
            embedding_type=args.embedding_type,
            cm2_checkpoint=args.cm2_checkpoint,
            cm2_table_flag=args.cm2_table_flag,
        )
    elif args.mode == 'cm2_finetune_strict_csv':
        results = evaluate_cm2(
            data,
            target_col,
            args.domain,
            n_runs=8,
            embedding_type=args.embedding_type,
            cm2_checkpoint=args.cm2_checkpoint,
            cm2_table_flag=args.cm2_table_flag,
            cm2_model_type="classifier",
            cm2_pretrain_checkpoint=args.cm2_pretrain_checkpoint,
            cm2_num_layer=args.cm2_num_layer,
            cm2_schema_csv=args.cm2_schema_csv,
            cm2_schema_target=target_col,
        )

    # Save results to JSON file with timestamp or shared run_id
    run_stamp = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    tv_suffix = ""
    if args.mode == "fe_pipelines":
        tv_suffix = f"_tv_{args.tablevectorizer_encoder}"
    ablation_suffix = ""
    if (
        args.mode != "cm2_finetune_strict_csv"
        and args.domain == "Product"
        and args.product_column_ablation != "allcol"
    ):
        ablation_suffix = f"_ablation_{args.product_column_ablation}"
    results_file = (
        f"experiments/logs/row_classification_{args.mode}_{args.domain.lower()}"
        f"{ablation_suffix}{tv_suffix}_{run_stamp}.json"
    )
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