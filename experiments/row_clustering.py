import torch
import numpy as np
import json
import random
import os
import argparse
import gc
from datetime import datetime
from collections import Counter
from pathlib import Path

# Quantitative analysis imports
from sklearn.metrics.pairwise import cosine_similarity

# Utility imports from your existing experiment scripts and utils
from experiments.experiment_utils import (
    load_data, 
    run_row_clustering,
    get_cls_embedding,
    get_meanpooled_embedding,
    get_meanpooled_segment_embedding
)
from row_classification import (
    clean_table_data_preserve_targets,
    preprocess_wdc_movie,
    preprocess_wdc_product,
    stratified_sample,
    remove_target_column,
    print_class_distribution,
    group_data_by_table
)

# Model imports
from transformers import BertForMaskedLM, BertTokenizer, TapasForMaskedLM, BertConfig
from baselines.haetae.model import HAETAE
from model.navi import NaviForMaskedLM

# Dataset imports
from dataset.dataset import NaviDataset, BertDataset, TapasDataset
from baselines.haetae.dataset import JSONDataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Re-initialize tokenizer locally
config = BertConfig.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def load_baseline_models(tokenizer):
    """Load baseline models (BERT, HAETAE, TAPAS, NAVI)"""
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

def run_model_clustering(dataset, target_col, model, model_name, domain, embedding_type="cls"):
    """
    Run clustering for a given model and return the results.
    """
    embeddings = []
    labels = []

    # Create the appropriate dataset for embedding extraction
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
        print(f"Unknown Model: {model_name}")
        return None

    # Extract embeddings
    for i, row in enumerate(dataset):
        if embedding_type == 'cls':
            emb = get_cls_embedding(dataset_X, i, model)
        elif embedding_type == 'mean':
            emb = get_meanpooled_embedding(dataset_X, i, model)
        embeddings.append(emb)
        labels.append(row[target_col])

    X = np.stack(embeddings)
    y = np.array(labels)

    # Run clustering
    results = run_row_clustering(X, y)
    
    return results

def evaluate_baselines(data, target_col, models, domain, embedding_type="cls"):
    """Evaluate baseline models (BERT, HAETAE, TAPAS, NAVI)"""
    print(f"\n{domain} - Baselines")
    results = {}
    
    # BERT
    print("\nBERT")
    bert_key = f'bert_{domain.lower()}'
    results['bert'] = run_model_clustering(data, target_col, models[bert_key], 'bert', domain, embedding_type)
    
    # TAPAS
    print("\nTAPAS")
    tapas_key = f'tapas_{domain.lower()}'
    results['tapas'] = run_model_clustering(data, target_col, models[tapas_key], 'tapas', domain, embedding_type)
    
    # HAETAE
    print("\nHAETAE")
    haetae_key = f'haetae_{domain.lower()}'
    results['haetae'] = run_model_clustering(data, target_col, models[haetae_key], 'haetae', domain, embedding_type)
    
    # NAVI
    print("\nNAVI")
    navi_key = f'navi_{domain.lower()}'
    results['navi'] = run_model_clustering(data, target_col, models[navi_key], 'navi', domain, embedding_type)
    
    return results

def evaluate_ablations(data, target_col, models, domain, embedding_type="cls"):
    """Evaluate ablation models"""
    print(f"\n{domain} - Ablations")
    results = {}
    
    domain_models = {k: v for k, v in models.items() if f"_{domain.lower()}_" in k}
    
    for model_name, model in domain_models.items():
        print(f"\n{model_name}")
        results[model_name] = run_model_clustering(data, target_col, model, model_name, domain, embedding_type)
        
    return results

def evaluate_hyperparams(data, target_col, tokenizer, domain, embedding_type="cls"):
    """Evaluate hyperparameter variations with memory management"""
    print(f"\n{domain} - Hyperparameter Variations")
    results = {}
    
    # Batch 1: hv0p4
    print("\n=== Evaluating hv0p4 variants ===")
    hv0p4_models = load_hyperparam_models_batch(tokenizer, '0p4', domain)
    for model_name, model in hv0p4_models.items():
        print(f"\n{model_name}")
        results[model_name] = run_model_clustering(data, target_col, model, model_name, domain, embedding_type)
    del hv0p4_models
    clear_memory()
    
    # Batch 2: hv0p8
    print("\n=== Evaluating hv0p8 variants ===")
    hv0p8_models = load_hyperparam_models_batch(tokenizer, '0p8', domain)
    for model_name, model in hv0p8_models.items():
        print(f"\n{model_name}")
        results[model_name] = run_model_clustering(data, target_col, model, model_name, domain, embedding_type)
    del hv0p8_models
    clear_memory()
    
    return results

def create_json_safe_results(results):
    """Remove numpy arrays from results to make them JSON serializable."""
    json_safe_results = {}
    
    for model_name, model_results in results.items():
        if isinstance(model_results, dict):
            # This is a model result dictionary
            json_safe_results[model_name] = {}
            for key, value in model_results.items():
                if key in ['embeddings', 'labels']:
                    # Skip numpy arrays
                    continue
                else:
                    json_safe_results[model_name][key] = value
        else:
            # Handle any other data types
            json_safe_results[model_name] = model_results
    
    return json_safe_results



def main():
    parser = argparse.ArgumentParser(description='Run row clustering experiments')
    parser.add_argument('--model', choices=['baselines', 'ablations', 'hyperparams'], 
                       required=True, help='Type of models to evaluate')
    parser.add_argument('--domain', choices=['Movie', 'Product'], 
                       required=True, help='Domain to evaluate on')
    parser.add_argument("--embedding_type", type=str, default="cls", choices=["cls", "mean"], 
                       help="Type of embedding to use")

    args = parser.parse_args()
    
    print(f"Evaluating {args.model} models on {args.domain} domain for clustering")
    
    # Load datasets
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

    # Clean data while preserving target columns for label extraction
    print("Cleaning product data...")
    wdc_product_data = clean_table_data_preserve_targets(wdc_product_data, target_columns=['category'])
    print("Cleaning movie data...")
    wdc_movie_data = clean_table_data_preserve_targets(wdc_movie_data, target_columns=['genres'])

    # Preprocess and filter data to create a manageable set for evaluation
    wdc_movie_data = preprocess_wdc_movie(wdc_movie_data)
    wdc_product_data = preprocess_wdc_product(wdc_product_data)

    # Stratified sampling to ensure class representation while keeping dataset size manageable
    wdc_movie_data = stratified_sample(wdc_movie_data, "genres", sample_size=500)
    wdc_product_data = stratified_sample(wdc_product_data, "category", sample_size=500)

    # Finalize dataset format
    wdc_movie_data = [row for idx, row in wdc_movie_data]
    wdc_product_data = [row for idx, row in wdc_product_data]

    print(f"WDC Movie data sample size: {len(wdc_movie_data)}")
    print(f"WDC Product data sample size: {len(wdc_product_data)}")

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
    results_file = f"experiments/logs/row_clustering_{args.model}_{args.domain.lower()}_{timestamp}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # Create JSON-safe version of results
    json_safe_results = create_json_safe_results(results)
    
    with open(results_file, 'w') as f:
        json.dump(json_safe_results, f, indent=2)
    
    print(f"\n✅ Clustering results saved to {results_file}")

if __name__ == "__main__":
    main()