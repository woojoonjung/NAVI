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
sys.path.append(os.path.abspath(".."))

import warnings
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

from experiment_utils import evaluate_masked_prediction, load_data
from dataset.dataset import AtlasDataset, BertDataset, HaetaeDataset, TapasDataset
from dataset.collator import CollatorForMaskedPrediction
from dataset.preprocess import clean_table_data

from model.atlas import AtlasForMaskedLM
from baselines.haetae.model import HAETAE
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer, TapasForMaskedLM, BertConfig


# -- helper functions -- #

def load_baseline_models(tokenizer):
    """Load baseline models (BERT, HAETAE, TAPAS)"""
    models = {}
    
    # Config
    config = BertConfig.from_pretrained('bert-base-uncased')

    # Baseline models - using trained versions
    models['bert_movie'] = BertForMaskedLM.from_pretrained('./models/bert_quarter_movie/epoch_2', local_files_only=True)
    models['bert_movie'] = models['bert_movie'].to(device)
    models['bert_movie'].eval()

    models['bert_product'] = BertForMaskedLM.from_pretrained('./models/bert_quarter_product/epoch_2', local_files_only=True)
    models['bert_product'] = models['bert_product'].to(device)
    models['bert_product'].eval()

    models['haetae_product'] = HAETAE(config, tokenizer, "./models/haetae_quarter_product/epoch_2")
    models['haetae_product'] = models['haetae_product'].to(device)
    models['haetae_product'].eval()

    models['haetae_movie'] = HAETAE(config, tokenizer, "./models/haetae_quarter_movie/epoch_2")
    models['haetae_movie'] = models['haetae_movie'].to(device)
    models['haetae_movie'].eval()

    models['tapas_movie'] = TapasForMaskedLM.from_pretrained('./models/tapas_quarter_movie/epoch_2', local_files_only=True)
    models['tapas_movie'] = models['tapas_movie'].to(device)
    models['tapas_movie'].eval()

    models['tapas_product'] = TapasForMaskedLM.from_pretrained('./models/tapas_quarter_product/epoch_2', local_files_only=True)
    models['tapas_product'] = models['tapas_product'].to(device)
    models['tapas_product'].eval()
    
    return models

def load_ablation_models(tokenizer):
    """Load ablation models"""
    models = {}
    config = BertConfig.from_pretrained('bert-base-uncased')
    
    domains = ['Movie', 'Product']
    hv_values = ['0p8']
    align_values = ['0p5']
    vr_values = ['0p5']
    ablation_values = ['woSI', 'woSMLM', 'woED']

    for domain in domains:
        for hv in hv_values:
            for align in align_values:
                for vr in vr_values:
                    for ablation in ablation_values:
                        model_path = f'./models/{ablation}_Quarter_{domain}_HVB_hv{hv}_align{align}_vr{vr}/epoch_2'
                        if os.path.exists(model_path):
                            model_name = f'{ablation}_{domain.lower()}_hv{hv}_align{align}_vr{vr}'
                            models[model_name] = AtlasForMaskedLM(model_path, ablation_mode=ablation)
                            models[model_name] = models[model_name].to(device)
                            models[model_name].eval()
                        else:
                            print(f"⚠️  Model not found: {model_path}")
    
    return models

def load_hyperparam_models_batch(tokenizer, hv_value, domain):
    """Load hyperparameter models for a specific hv value and domain"""
    models = {}
    
    # Based on the images, we have these parameter combinations:
    align_values = ['0p5']
    vr_values = ['0p5']
    
    for align in align_values:
        for vr in vr_values:
            model_path = f'./models/full_Quarter_{domain}_HVB_hv{hv_value}_align{align}_vr{vr}/epoch_2'
            if os.path.exists(model_path):
                model_name = f'atlas_{domain.lower()}_hv{hv_value}_align{align}_vr{vr}'
                models[model_name] = AtlasForMaskedLM(model_path)
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

def evaluate_baselines(bert_dataset, haetae_dataset, tapas_dataset, models, tokenizers, collators, domain):
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
    
    # # First batch: hv0p4 (15 variants)
    # print("\n=== Evaluating hv0p4 variants ===")
    # hv0p4_models = load_hyperparam_models_batch(tokenizer, '0p4', domain)
    
    # for model_name, model in hv0p4_models.items():
    #     print(f"\n{model_name}")
    #     print("Header:")
    #     header_acc = evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch=1)
    #     print("Value:")
    #     value_acc = evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch=5)
        
    #     results[model_name] = {
    #         'header_accuracy': header_acc,
    #         'value_accuracy': value_acc
    #     }
    
    # # Clear memory after first batch
    # del hv0p4_models
    # clear_memory()
    
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

def main():
    parser = argparse.ArgumentParser(description='Evaluate masked prediction models')
    parser.add_argument('--model', choices=['baselines', 'ablations', 'hyperparams'], 
                       required=True, help='Type of models to evaluate')
    parser.add_argument('--domain', choices=['Movie', 'Product'], 
                       required=True, help='Domain to evaluate on')
    
    args = parser.parse_args()
    
    print(f"Evaluating {args.model} models on {args.domain} domain")
    
    # Data loading
    product_path = "./data/WDC_product_for_mp.jsonl"
    movie_path = "./data/WDC_movie_for_mp.jsonl"
    raw_product_json = load_data(product_path)
    raw_movie_json = load_data(movie_path)

    grouped_product_json = group_data_by_table(raw_product_json)
    grouped_movie_json = group_data_by_table(raw_movie_json)

    product_json = []
    movie_json = []

    for idx, table in grouped_product_json:
        for row in table:
            product_json.append((idx, row))
    for idx, table in grouped_movie_json:
        for row in table:
            movie_json.append((idx, row))

    product_json = clean_table_data(product_json)
    movie_json = clean_table_data(movie_json)

    product_json_flat = []
    movie_json_flat = []

    for idx, row in product_json:
        product_json_flat.append(row)
    for idx, row in movie_json:
        movie_json_flat.append(row)

    # Prepare datasets
    if args.domain == 'Product':
        atlas_dataset = AtlasDataset(product_json)
        bert_dataset = BertDataset(product_json_flat, mode="masked_prediction")
        haetae_dataset = HaetaeDataset(product_json_flat)
        tapas_dataset = TapasDataset(product_json_flat, mode="masked_prediction")
    else:  # Movie
        atlas_dataset = AtlasDataset(movie_json)
        bert_dataset = BertDataset(movie_json_flat, mode="masked_prediction")
        haetae_dataset = HaetaeDataset(movie_json_flat)
        tapas_dataset = TapasDataset(movie_json_flat, mode="masked_prediction")

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
        evaluate_baselines(bert_dataset, haetae_dataset, tapas_dataset, models, tokenizers, collators, args.domain)
        
    elif args.model == 'ablations':
        models = load_ablation_models(tokenizers[0])
        evaluate_ablations(atlas_dataset, models, tokenizers[0], collators[0], args.domain)
        
    elif args.model == 'hyperparams':
        results = evaluate_hyperparams(atlas_dataset, tokenizers[0], collators[0], args.domain)
        
        # Print summary of results
        print(f"\n=== Summary for {args.domain} Hyperparameter Evaluation ===")
        for model_name, metrics in results.items():
            print(f"{model_name}: Header={metrics['header_accuracy']:.4f}, Value={metrics['value_accuracy']:.4f}")

if __name__ == "__main__":
    main()