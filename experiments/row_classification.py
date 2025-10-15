import torch
import numpy as np
import json
import random
import pandas as pd
import os
import argparse
import gc
from datetime import datetime
from experiments.experiment_utils import (
    load_data, 
    run_row_classification,
    get_cls_embedding,
    get_meanpooled_embedding
)

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
from dataset.preprocess import clean_table_data
from baselines.haetae.dataset import JSONDataset
from collections import Counter, defaultdict

from transformers import BertTokenizer, BertConfig

config = BertConfig.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def clean_table_data_preserve_targets(json_data, target_columns, tokenizer_name="bert-base-uncased", 
                                     max_tokens=500, max_indexed_fields=3, max_tokens_per_field=20):
    """
    Clean table data while preserving target columns.
    """
    from dataset.preprocess import handle_indexed_fields, truncate_long_fields, select_fields_for_tokenization, estimate_token_count
    
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    processed_data = []
    
    print("🔄 Preprocessing table data while preserving target columns...")
    
    for table_id, table_dict in json_data:
        # Step 1: Handle indexed fields
        processed_table = handle_indexed_fields(table_dict, max_indexed_fields)
        
        # Step 2: Truncate long text fields
        processed_table = truncate_long_fields(processed_table, max_tokens_per_field, tokenizer)
        
        # Step 3: Select fields that fit in token limit, but preserve target columns
        processed_table = select_fields_for_tokenization_preserve_targets(
            processed_table, tokenizer, max_tokens, target_columns
        )
        
        processed_data.append((table_id, processed_table))
        
        if len(processed_data) % 1000 == 0:
            print(f"   Processed {len(processed_data)} instances...")
    
    print(f"✅ Preprocessing complete. Processed {len(processed_data)} instances.")
    return processed_data


def select_fields_for_tokenization_preserve_targets(table_dict, tokenizer, max_tokens=480, target_columns=None):
    """Select fields that will fit within token limit while preserving target columns."""
    if target_columns is None:
        target_columns = []
    
    selected_fields = {}
    current_tokens = 0
    
    # First, always include target columns
    for target_col in target_columns:
        if target_col in table_dict:
            selected_fields[target_col] = table_dict[target_col]
            current_tokens += estimate_token_count(target_col, table_dict[target_col], tokenizer)
    
    # Then select other fields by priority (shorter fields first)
    remaining_fields = {k: v for k, v in table_dict.items() if k not in target_columns}
    field_items = list(remaining_fields.items())
    field_items.sort(key=lambda x: len(str(x[1])))
    
    for field_name, field_value in field_items:
        estimated_tokens = estimate_token_count(field_name, field_value, tokenizer)
        
        if current_tokens + estimated_tokens <= max_tokens:
            selected_fields[field_name] = field_value
            current_tokens += estimated_tokens
        else:
            break  # Stop when we would exceed limit
    
    return selected_fields


def estimate_token_count(field_name, field_value, tokenizer):
    """Estimate token count for a field."""
    if field_value is None:
        return 0
    
    # Include field name in token count
    full_text = f"{field_name} : {field_value}"
    tokens = tokenizer.tokenize(str(full_text))
    return len(tokens)


def remove_target_column(row, target_col):
    """
    Remove the target column from the row.
    """
    return {k: v for k, v in row.items() if k != target_col}  

def run_cls_classification(dataset, target_col, model, model_name, domain, ml_model="rf", embedding_type="cls"):
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
    f1 = run_row_classification(X, y, model_type=ml_model)
    return f1

def run_repeated_classification(dataset, target_col, model, model_name, domain, ml_model="rf", n_runs=8, embedding_type="cls"):
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
    for table_id, row_dict in wdc_movie_data:
        genre_str = row_dict.get("genres", "")
        normalized = ", ".join(g.strip().lower() for g in genre_str.split(',')) if isinstance(genre_str, str) else ""
        if is_valid_genre_combo(normalized):
            normalized_genres.append(normalized)

    ## Get top 20 valid combinations
    genre_combo_counts = Counter(normalized_genres)
    top_20_combos = set(combo for combo, _ in genre_combo_counts.most_common(20))

    ## Filter the original dataset
    filtered_data = []
    for table_id, row_dict in wdc_movie_data:
        genre_str = row_dict.get("genres", "")
        normalized = ", ".join(g.strip().lower() for g in genre_str.split(',')) if isinstance(genre_str, str) else ""
        if normalized in top_20_combos:
            filtered_data.append((table_id, row_dict))

    wdc_movie_data = filtered_data

    ## Step 1: Extract all genre labels
    labels = [row_dict["genres"] for table_id, row_dict in wdc_movie_data]

    ## Step 2: Count frequencies
    label_counts = Counter(labels)

    ## Step 3: Identify valid labels (appear more than once)
    valid_labels = {label for label, count in label_counts.items() if count > 1}

    ## Step 4: Filter the data
    final_data = []
    for table_id, row_dict in wdc_movie_data:
        if row_dict["genres"] in valid_labels:
            final_data.append((table_id, row_dict))

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

    # Extract the actual data from tuples (table_id, table_dict)
    actual_data = [row_dict for table_id, row_dict in wdc_product_data]
    
    # Convert to DataFrame
    df = pd.DataFrame(actual_data)

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
    top_categories = sorted(english_categories.items(), key=lambda x: x[1], reverse=True)[:20]
    top_categories = [cat for cat, _ in top_categories]

    # Filter wdc_product_data to include only rows that contain at least one of the top categories
    filtered_wdc_product_data = []
    for table_id, row_dict in wdc_product_data:
        cats = row_dict.get("category", "")
        if isinstance(cats, str):
            split_cats = [cat.strip().lower() for cat in cats.split(',')]
            if any(cat in top_categories for cat in split_cats):
                filtered_wdc_product_data.append((table_id, row_dict))

    return filtered_wdc_product_data

def stratified_sample(data, label_key, sample_size=1000):
    # Group items by class
    label_to_items = defaultdict(list)
    for table_id, row_dict in data:
        label = row_dict.get(label_key)
        if label:
            label_to_items[label].append((table_id, row_dict))

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

def load_baseline_models(tokenizer):
    """Load baseline models (BERT, HAETAE, TAPAS)"""
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

    models['navi_movie'] = NaviForMaskedLM('./models/navi_movie/epoch_2')
    models['navi_movie'] = models['navi_movie'].to(device)
    models['navi_movie'].eval()
    
    models['navi_product'] = NaviForMaskedLM('./models/navi_product/epoch_2')
    models['navi_product'] = models['navi_product'].to(device)
    models['navi_product'].eval()
    
    return models

def load_ablation_models(tokenizer):
    """Load ablation models"""
    models = {}
    config = BertConfig.from_pretrained('bert-base-uncased')
    
    domains = ['movie', 'product']
    ablation_values = ['woSSI', 'woMSM', 'woESA']

    for domain in domains:
        for ablation in ablation_values:
            model_path = f'./models/navi_{domain}_{ablation}/epoch_2'
            if os.path.exists(model_path):
                model_name = f'navi_{domain}_{ablation}'
                models[model_name] = NaviForMaskedLM(model_path, ablation_mode=ablation)
                models[model_name] = models[model_name].to(device)
                models[model_name].eval()
            else:
                print(f"⚠️  Model not found: {model_path}")
    
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

def evaluate_baselines(data, target_col, models, domain, n_runs=5, embedding_type="cls"):
    """Evaluate baseline models (BERT, HAETAE, TAPAS)"""
    print(f"\n{domain} - Baselines")
    
    results = {}
    ml_models = ["rf", "xgboost", "lr", "svm"]
    
    # BERT
    print("\nBERT")
    bert_key = f'bert_{domain.lower()}'
    results['bert'] = {}
    for ml_model in ml_models:
        result = run_repeated_classification(
            data, target_col, models[bert_key], 'bert', domain, 
            ml_model=ml_model, n_runs=n_runs, embedding_type=embedding_type
        )
        results['bert'][f"{ml_model}_{domain}"] = result
    
    # TAPAS
    print("\nTAPAS")
    tapas_key = f'tapas_{domain.lower()}'
    results['tapas'] = {}
    for ml_model in ml_models:
        result = run_repeated_classification(
            data, target_col, models[tapas_key], 'tapas', domain, 
            ml_model=ml_model, n_runs=n_runs, embedding_type=embedding_type
        )
        results['tapas'][f"{ml_model}_{domain}"] = result
    
    # HAETAE
    print("\nHAETAE")
    haetae_key = f'haetae_{domain.lower()}'
    results['haetae'] = {}
    for ml_model in ml_models:
        result = run_repeated_classification(
            data, target_col, models[haetae_key], 'haetae', domain, 
            ml_model=ml_model, n_runs=n_runs, embedding_type=embedding_type
        )
        results['haetae'][f"{ml_model}_{domain}"] = result

    # NAVI
    print("\nNAVI")
    navi_key = f'navi_{domain.lower()}'
    results['navi'] = {}
    for ml_model in ml_models:
        result = run_repeated_classification(
            data, target_col, models[navi_key], 'navi', domain, 
            ml_model=ml_model, n_runs=n_runs, embedding_type=embedding_type
        )
        results['navi'][f"{ml_model}_{domain}"] = result
    
    return results

def evaluate_ablations(data, target_col, models, domain, n_runs=8, embedding_type="cls"):
    """Evaluate ablation models"""
    print(f"\n{domain} - Ablations")
    
    results = {}
    ml_models = ["rf", "xgboost", "lr", "svm"]
    
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
    ml_models = ["rf", "xgboost", "lr", "svm"]
    
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

def main():
    parser = argparse.ArgumentParser(description='Run row classification experiments')
    parser.add_argument('--model', choices=['baselines', 'ablations', 'hyperparams'], 
                       required=True, help='Type of models to evaluate')
    parser.add_argument('--domain', choices=['Movie', 'Product'], 
                       required=True, help='Domain to evaluate on')
    parser.add_argument("--embedding_type", type=str, default="cls", choices=["cls", "mean"], 
                       help="Type of embedding to use")
    
    args = parser.parse_args()
    
    print(f"Evaluating {args.model} models on {args.domain} domain")
    
    # Load classification datasets
    wdc_product_data = load_data("data/WDC_product_for_cls.jsonl")
    wdc_movie_data = load_data("data/wd_WDC_movie_for_cls.jsonl")
    
    grouped_wdc_product_data = group_data_by_table(wdc_product_data)
    grouped_wdc_movie_data = group_data_by_table(wdc_movie_data)

    wdc_product_data = []
    wdc_movie_data = []

    for idx, table in grouped_wdc_product_data:
        for row in table:
            wdc_product_data.append((idx, row))
    for idx, table in grouped_wdc_movie_data:
        for row in table:
            wdc_movie_data.append((idx, row))

    # Clean data while preserving target columns
    print("Cleaning product data...")
    wdc_product_data = clean_table_data_preserve_targets(wdc_product_data, target_columns=['category'])
    print("Cleaning movie data...")
    wdc_movie_data = clean_table_data_preserve_targets(wdc_movie_data, target_columns=['genres'])

    # Preprocess WDC data
    wdc_movie_data = preprocess_wdc_movie(wdc_movie_data)
    wdc_product_data = preprocess_wdc_product(wdc_product_data)

    wdc_movie_data = stratified_sample(wdc_movie_data, "genres")
    wdc_product_data = stratified_sample(wdc_product_data, "category")

    wdc_movie_data = [row for idx, row in wdc_movie_data]
    wdc_product_data = [row for idx, row in wdc_product_data]

    print(f"WDC Movie data: {len(wdc_movie_data)}")
    print(f"WDC Product data: {len(wdc_product_data)}")

    print_class_distribution(wdc_movie_data, "genres", "WDC Movie")
    print_class_distribution(wdc_product_data, "category", "WDC Product")

    # Prepare data for the specified domain
    if args.domain == 'Product':
        data = wdc_product_data
        target_col = 'category'
    else:  # Movie
        data = wdc_movie_data
        target_col = 'genres'

    # Load models based on type
    if args.model == 'baselines':
        models = load_baseline_models(tokenizer)
        results = evaluate_baselines(data, target_col, models, args.domain.lower(), embedding_type=args.embedding_type)
        
    elif args.model == 'ablations':
        models = load_ablation_models(tokenizer)
        results = evaluate_ablations(data, target_col, models, args.domain.lower(), embedding_type=args.embedding_type)
        
    elif args.model == 'hyperparams':
        results = evaluate_hyperparams(data, target_col, tokenizer, args.domain, embedding_type=args.embedding_type)

    # Save results to JSON file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    
    for model_name, model_results in results.items():
        print(f"\n{model_name}:")
        for task_name, task_result in model_results.items():
            mean_val = task_result['mean']
            std_val = task_result['std']
            print(f"  {task_name}: {mean_val:.4f} ± {std_val:.4f}")


if __name__ == "__main__":
    main()
