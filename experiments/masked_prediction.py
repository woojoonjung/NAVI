# Device

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and will be used.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU.")

# Imports

import sys
import os
import argparse
import gc
import glob
import random
sys.path.append(os.path.abspath(".."))

import warnings
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

from experiment_utils import evaluate_masked_prediction, load_data
from dataset.dataset import NaviDataset, BertDataset, HaetaeDataset, TapasDataset
from dataset.collator import CollatorForMaskedPrediction

from model.navi import NaviForMaskedLM
from baselines.haetae.model import HAETAE
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer, TapasForMaskedLM, BertConfig


# -- helper functions -- #

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

    # NAVI models - use find_epoch2_path to locate the actual epoch directory
    navi_movie_base = './models/navi_movie'
    navi_movie_path = find_epoch2_path(navi_movie_base)
    if navi_movie_path:
        models['navi_movie'] = NaviForMaskedLM(navi_movie_path)
        models['navi_movie'] = models['navi_movie'].to(device)
        models['navi_movie'].eval()
    else:
        print(f"⚠️  NAVI Movie model not found: {navi_movie_base}/*epoch_2")
    
    navi_product_base = './models/navi_product'
    navi_product_path = find_epoch2_path(navi_product_base)
    if navi_product_path:
        models['navi_product'] = NaviForMaskedLM(navi_product_path)
        models['navi_product'] = models['navi_product'].to(device)
        models['navi_product'].eval()
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

def find_epoch2_path(base_path):
    """Find epoch_2 directory using glob pattern"""
    pattern = os.path.join(base_path, "*epoch_2")
    matches = glob.glob(pattern)
    if matches:
        # Prefer the most specific match (longest path)
        matches.sort(key=len, reverse=True)
        return matches[0]
    return None

def load_ablation_variants_models(tokenizer, domain):
    """Load ablation variant models for a specific domain"""
    models = {}
    domain_lower = domain.lower()
    
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

def group_data_by_table(data):
    """
    Group data by table based on row_id patterns.
    Each table contains 10 consecutive rows with row_id 450-459.
    
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
        
        # Check if this is the start of a new table (row_id 450)
        if row_id == 450 and current_table:
            # Save the previous table
            if len(current_table) == 10:  # Ensure we have a complete table
                grouped_tables.append((table_idx, current_table))
                table_idx += 1
            current_table = []
        
        current_table.append(row)
    
    # Don't forget the last table
    if current_table and len(current_table) == 10:
        grouped_tables.append((table_idx, current_table))
    
    print(f"Grouped {len(data)} rows into {len(grouped_tables)} tables")
    return grouped_tables

def evaluate_baselines(bert_dataset, haetae_dataset, tapas_dataset, navi_dataset, models, tokenizers, collators, domain):
    """Evaluate baseline models (BERT, HAETAE, TAPAS)"""
    print(f"\n{domain} - Baselines")
    
    # BERT
    print("\nBERT")
    bert_key = f'bert_{domain.lower()}'
    print("Header:")
    evaluate_masked_prediction(bert_dataset, models[bert_key], tokenizers[0], collators[0], epoch=1)
    print("Value:")
    evaluate_masked_prediction(bert_dataset, models[bert_key], tokenizers[0], collators[0], epoch=5)
    
    # TAPAS
    print("\nTAPAS")
    tapas_key = f'tapas_{domain.lower()}'
    print("Header:")
    evaluate_masked_prediction(tapas_dataset, models[tapas_key], tokenizers[1], collators[1], epoch=1)
    print("Value:")
    evaluate_masked_prediction(tapas_dataset, models[tapas_key], tokenizers[1], collators[1], epoch=5)
    
    # HAETAE
    print("\nHAETAE")
    haetae_key = f'haetae_{domain.lower()}'
    print("Header:")
    evaluate_masked_prediction(haetae_dataset, models[haetae_key], tokenizers[0], collators[0], epoch=1)
    print("Value:")
    evaluate_masked_prediction(haetae_dataset, models[haetae_key], tokenizers[0], collators[0], epoch=5)
    
    # NAVI
    print("\nNAVI")
    navi_key = f'navi_{domain.lower()}'
    print("Header:")
    evaluate_masked_prediction(navi_dataset, models[navi_key], tokenizers[0], collators[0], epoch=1)
    print("Value:")
    evaluate_masked_prediction(navi_dataset, models[navi_key], tokenizers[0], collators[0], epoch=5)

def evaluate_ablations(dataset, models, tokenizer, collator, domain):
    """Evaluate ablation models"""
    print(f"\n{domain} - Ablations")
    
    # Filter models for this domain
    domain_models = {k: v for k, v in models.items() if f"_{domain.lower()}_" in k}
    
    for model_name, model in domain_models.items():
        print(f"\n{model_name}")
        print("Header:")
        evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch=1)
        print("Value:")
        evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch=5)

def evaluate_hyperparams(dataset, tokenizer, collator, domain):
    """Evaluate hyperparameter variations with memory management"""
    print(f"\n{domain} - Hyperparameter Variations")
    
    results = {}
    
    # First batch: hv0p4 (15 variants)
    print("\n=== Evaluating hv0p4 variants ===")
    hv0p4_models = load_hyperparam_models_batch(tokenizer, '0p4', domain)
    
    for model_name, model in hv0p4_models.items():
        print(f"\n{model_name}")
        print("Header:")
        header_acc = evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch=1)
        print("Value:")
        value_acc = evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch=5)
        
        results[model_name] = {
            'header_accuracy': header_acc,
            'value_accuracy': value_acc
        }
    
    # Clear memory after first batch
    del hv0p4_models
    clear_memory()
    
    # Second batch: hv0p8 (15 variants)
    print("\n=== Evaluating hv0p8 variants ===")
    hv0p8_models = load_hyperparam_models_batch(tokenizer, '0p8', domain)
    
    for model_name, model in hv0p8_models.items():
        print(f"\n{model_name}")
        print("Header:")
        header_acc = evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch=1)
        print("Value:")
        value_acc = evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch=5)
        
        results[model_name] = {
            'header_accuracy': header_acc,
            'value_accuracy': value_acc
        }
    
    # Clear memory after second batch
    del hv0p8_models
    clear_memory()
    
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
        
        model_path = find_epoch2_path(base_path)
        if model_path:
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {model_path}")
        else:
            print(f"⚠️  Model not found: {base_path}/*epoch_2")
    
    return models

def evaluate_ablation_variants(dataset, tokenizer, collator, domain):
    """Evaluate ablation variants (entropy threshold, temperature, gradient accumulation)"""
    print(f"\n{domain} - Ablation Variants")
    
    results = {}
    
    # Load all ablation variant models
    models = load_ablation_variants_models(tokenizer, domain)
    
    # Evaluate each model (Value prediction only, epoch=5)
    for model_name, model in models.items():
        print(f"\n{model_name}")
        print("Value:")
        value_acc = evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch=5)
        results[model_name] = {
            'value_accuracy': value_acc
        }
    
    # Clear memory
    del models
    clear_memory()
    
    return results

def evaluate_tau_align_ethresh(dataset, tokenizer, collator, domain):
    """Evaluate tau/align/ethresh variant models"""
    print(f"\n{domain} - Tau/Align/Ethresh Variants")
    
    results = {}
    
    # Load tau/align/ethresh models
    models = load_tau_align_ethresh_models(tokenizer, domain)
    
    # Evaluate each model (Header and Value prediction)
    for model_name, model in models.items():
        print(f"\n{model_name}")
        print("Header:")
        header_acc = evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch=1)
        print("Value:")
        value_acc = evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch=5)
        results[model_name] = {
            'header_accuracy': header_acc,
            'value_accuracy': value_acc
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
        """Find epoch directory using glob pattern, filtered to paths ending with *_epoch2 and excluding tau-related matches"""
        pattern = os.path.join(base_path, "*epoch*")
        matches = glob.glob(pattern)
        # Filter to only include paths ending with _epoch2 or _epoch_2, and exclude tau-related matches
        matches = [m for m in matches if (os.path.basename(m).endswith("_epoch2") or os.path.basename(m).endswith("_epoch_2")) and "_tau" not in os.path.basename(m)]
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

def evaluate_hyperparam_sensitivity(dataset, tokenizer, collator, domain):
    """Evaluate hyperparameter sensitivity analysis for masked prediction"""
    print(f"\n{domain} - Hyperparameter Sensitivity Analysis")
    
    results = {}
    
    # Load all hyperparameter sensitivity models
    models = load_hyperparam_sensitivity_models(tokenizer, domain)
    
    # Evaluate each model (Header and Value prediction)
    for model_name, model in models.items():
        print(f"\n{model_name}")
        print("Header:")
        header_acc = evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch=1)
        print("Value:")
        value_acc = evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch=5)
        results[model_name] = {
            'header_accuracy': header_acc,
            'value_accuracy': value_acc
        }
    
    # Clear memory
    del models
    clear_memory()
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate masked prediction models')
    parser.add_argument('--model', choices=['baselines', 'ablations', 'hyperparams', 'ablation_variants', 'tau_align_ethresh', 'hyperparam_sensitivity'], 
                       required=True, help='Type of models to evaluate')
    parser.add_argument('--domain', choices=['Movie', 'Product'], 
                       required=True, help='Domain to evaluate on')
    
    args = parser.parse_args()
    
    print(f"Evaluating {args.model} models on {args.domain} domain")
    
    # Data loading - only load data for the specified domain
    domain_lower = args.domain.lower()
    data_path = f"./data/cleaned/{args.domain}/test/WDC_{domain_lower}_for_mp.jsonl"
    print(f"Loading data from: {data_path}")
    raw_json = load_data(data_path)
    print(f"Loaded {len(raw_json)} rows")
    
    # Sample 1000 rows with fixed random seed for reproducibility
    random.seed(42)
    if len(raw_json) > 1000:
        raw_json = random.sample(raw_json, 1000)
        print(f"Sampled 1000 rows (seed=42)")
    else:
        print(f"Using all {len(raw_json)} rows (less than 1000 available)")

    # Create tuples with sequential table IDs (0, 1, 2, ...) for NaviDataset
    # NaviDataset can handle tuple format (table_id, row)
    json_data = [(i, row) for i, row in enumerate(raw_json)]

    # Extract flat list of rows for other datasets
    json_data_flat = [row for _, row in json_data]

    # Prepare datasets
    navi_dataset = NaviDataset(json_data)  # NaviDataset can handle tuple format
    bert_dataset = BertDataset(json_data_flat, mode="masked_prediction")
    haetae_dataset = HaetaeDataset(json_data_flat)
    tapas_dataset = TapasDataset(json_data_flat, mode="masked_prediction")

    # Tokenizers
    AHB_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tapas_tokenizer = AutoTokenizer.from_pretrained("google/tapas-base")
    tokenizers = [AHB_tokenizer, tapas_tokenizer]

    # Collators
    AHB_collator = CollatorForMaskedPrediction(tokenizer=tokenizers[0])
    tapas_collator = CollatorForMaskedPrediction(tokenizer=tokenizers[1])
    collators = [AHB_collator, tapas_collator]

    # Load models based on type
    if args.model == 'baselines':
        models = load_baseline_models(tokenizers[0])
        evaluate_baselines(bert_dataset, haetae_dataset, tapas_dataset, navi_dataset, models, tokenizers, collators, args.domain)
        
    elif args.model == 'ablations':
        models = load_ablation_models(tokenizers[0])
        evaluate_ablations(navi_dataset, models, tokenizers[0], collators[0], args.domain)
        
    elif args.model == 'hyperparams':
        results = evaluate_hyperparams(navi_dataset, tokenizers[0], collators[0], args.domain)
        
        # Print summary of results
        print(f"\n=== Summary for {args.domain} Hyperparameter Evaluation ===")
        for model_name, metrics in results.items():
            print(f"{model_name}: Header={metrics['header_accuracy']:.4f}, Value={metrics['value_accuracy']:.4f}")
    
    elif args.model == 'ablation_variants':
        results = evaluate_ablation_variants(navi_dataset, tokenizers[0], collators[0], args.domain)
        
        # Print summary of results
        print(f"\n=== Summary for {args.domain} Ablation Variants Evaluation ===")
        print("\nValue Prediction Accuracy (epoch=5):")
        for model_name, metrics in results.items():
            print(f"{model_name}: {metrics['value_accuracy']:.4f}")
    
    elif args.model == 'tau_align_ethresh':
        results = evaluate_tau_align_ethresh(navi_dataset, tokenizers[0], collators[0], args.domain)
        
        # Print summary of results
        print(f"\n=== Summary for {args.domain} Tau/Align/Ethresh Variants Evaluation ===")
        for model_name, metrics in results.items():
            print(f"{model_name}: Header={metrics['header_accuracy']:.4f}, Value={metrics['value_accuracy']:.4f}")
    
    elif args.model == 'hyperparam_sensitivity':
        results = evaluate_hyperparam_sensitivity(navi_dataset, tokenizers[0], collators[0], args.domain)
        
        # Print summary of results
        print(f"\n=== Summary for {args.domain} Hyperparameter Sensitivity Analysis ===")
        print("\nHeader Prediction Accuracy (epoch=1):")
        for model_name, metrics in results.items():
            print(f"{model_name}: {metrics['header_accuracy']:.4f}")
        print("\nValue Prediction Accuracy (epoch=5):")
        for model_name, metrics in results.items():
            print(f"{model_name}: {metrics['value_accuracy']:.4f}")

if __name__ == "__main__":
    main()