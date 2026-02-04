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

# Quantitative analysis imports
from sklearn.metrics.pairwise import cosine_similarity

# Utility imports from your existing experiment scripts and utils
from experiments.experiment_utils import (
    load_data, 
    run_row_clustering,
    get_cls_embedding,
    get_meanpooled_embedding
)
from row_classification import (
    verify_target_columns_present,
    preprocess_wdc_movie,
    preprocess_wdc_product,
    stratified_sample,
    remove_target_column,
    print_class_distribution
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

    # NAVI models - use navi_movie and navi_product_default_3epoch
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
    
    # NAVI Movie - use navi_movie
    navi_movie_base = './models/navi_movie_default'
    navi_movie_path = find_epoch2_path(navi_movie_base)
    if navi_movie_path:
        models['navi_movie'] = NaviForMaskedLM(navi_movie_path)
        models['navi_movie'] = models['navi_movie'].to(device)
        models['navi_movie'].eval()
        print(f"✓ Loaded NAVI Movie from: {navi_movie_path}")
    else:
        print(f"⚠️  NAVI Movie model not found: {navi_movie_base}/*epoch_2")
    
    # NAVI Product - use navi_product_default_3epoch
    navi_product_base = './models/navi_product_default'
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
    
    domains = ['movie', 'product']
    ablation_values = ['woSSI', 'woMSM', 'woESA']

    for domain in domains:
        for ablation in ablation_values:
            model_path = f'./models/navi_{domain}_{ablation}/{ablation}_HVB_seed42_cleaned_tau0.02_0.14_percentile_epoch_2'
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

def evaluate_hyperparam_sensitivity(data, target_col, tokenizer, domain, embedding_type="cls"):
    """Evaluate hyperparameter sensitivity analysis: only align*, ethresh*, and tau* variants"""
    print(f"\n{domain} - Hyperparameter Sensitivity Analysis")
    results = {}
    
    # Load all hyperparameter sensitivity models
    models = load_hyperparam_sensitivity_models(tokenizer, domain)
    
    # Evaluate each model
    for model_name, model in models.items():
        print(f"\n{model_name}")
        results[model_name] = run_model_clustering(data, target_col, model, model_name, domain, embedding_type)
    
    # Clear memory
    del models
    clear_memory()
    
    return results

def evaluate_tau_align_ethresh(data, target_col, tokenizer, domain, embedding_type="cls"):
    """Evaluate tau/align/ethresh variant models"""
    print(f"\n{domain} - Tau/Align/Ethresh Variants")
    results = {}
    
    # Load tau/align/ethresh models
    models = load_tau_align_ethresh_models(tokenizer, domain)
    
    # Evaluate each model
    for model_name, model in models.items():
        print(f"\n{model_name}")
        results[model_name] = run_model_clustering(data, target_col, model, model_name, domain, embedding_type)
    
    # Clear memory
    del models
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
    parser.add_argument('--model', choices=['baselines', 'ablations', 'hyperparams', 'hyperparam_sensitivity', 'tau_align_ethresh'], 
                       required=True, help='Type of models to evaluate')
    parser.add_argument('--domain', choices=['Movie', 'Product'], 
                       required=True, help='Domain to evaluate on')
    parser.add_argument("--embedding_type", type=str, default="cls", choices=["cls", "mean"], 
                       help="Type of embedding to use")

    args = parser.parse_args()
    
    print(f"Evaluating {args.model} models on {args.domain} domain for clustering")
    
    # Load and process data for the specified domain only
    if args.domain == 'Product':
        # Load Product classification dataset
        wdc_product_data = load_data("data/cleaned/Product/test/WDC_product_for_cls.jsonl")
        
        # Verify target columns are present (data already cleaned in preprocess.py)
        print("Verifying target columns in product data...")
        wdc_product_data = verify_target_columns_present(wdc_product_data, target_columns=['category'])
        
        # Preprocess and filter data to create a manageable set for evaluation
        wdc_product_data = preprocess_wdc_product(wdc_product_data)
        
        # Stratified sampling to ensure class representation while keeping dataset size manageable
        wdc_product_data = stratified_sample(wdc_product_data, "category", sample_size=500)
        
        print(f"WDC Product data sample size: {len(wdc_product_data)}")
        print_class_distribution(wdc_product_data, "category", "WDC Product")
        
        data = wdc_product_data
        target_col = 'category'
    else:  # Movie
        # Load Movie classification dataset
        wdc_movie_data = load_data("data/cleaned/Movie/test/WDC_movie_for_cls.jsonl")
        
        # Verify target columns are present (data already cleaned in preprocess.py)
        print("Verifying target columns in movie data...")
        wdc_movie_data = verify_target_columns_present(wdc_movie_data, target_columns=['genres'])
        
        # Preprocess and filter data to create a manageable set for evaluation
        wdc_movie_data = preprocess_wdc_movie(wdc_movie_data)
        
        # Stratified sampling to ensure class representation while keeping dataset size manageable
        wdc_movie_data = stratified_sample(wdc_movie_data, "genres", sample_size=500)
        
        print(f"WDC Movie data sample size: {len(wdc_movie_data)}")
        print_class_distribution(wdc_movie_data, "genres", "WDC Movie")
        
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
    
    elif args.model == 'hyperparam_sensitivity':
        results = evaluate_hyperparam_sensitivity(data, target_col, tokenizer, args.domain, embedding_type=args.embedding_type)
    
    elif args.model == 'tau_align_ethresh':
        results = evaluate_tau_align_ethresh(data, target_col, tokenizer, args.domain, embedding_type=args.embedding_type)

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