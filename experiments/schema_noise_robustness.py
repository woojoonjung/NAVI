#!/usr/bin/env python3
"""
Schema Noise Robustness Experiment
Evaluates NAVI's performance on classification and imputation tasks under schema perturbations.
"""

import torch
import numpy as np
import json
import os
import argparse
import gc
import random
from datetime import datetime
from pathlib import Path

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Imports
import warnings
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from experiments.experiment_utils import (
    load_data,
    evaluate_masked_prediction
)
from experiments.row_classification import (
    run_repeated_classification,
    preprocess_wdc_movie,
    preprocess_wdc_product,
    stratified_sample,
    verify_target_columns_present
)
from dataset.dataset import NaviDataset
from dataset.collator import CollatorForMaskedPrediction
from model.navi import NaviForMaskedLM
from transformers import BertTokenizer

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def load_models():
    """
    Load NAVI models for Movie and Product domains.
    
    Returns:
        dict: Dictionary with 'movie' and 'product' model keys
    """
    models = {}
    
    # Movie model
    movie_model_path = "./models/navi_movie/full_HVB_seed42_cleaned_tau0.02_0.14_percentile_epoch_2"
    if os.path.exists(movie_model_path):
        print(f"Loading Movie model from: {movie_model_path}")
        models['movie'] = NaviForMaskedLM(movie_model_path)
        models['movie'] = models['movie'].to(device)
        models['movie'].eval()
        print("✓ Movie model loaded")
    else:
        raise FileNotFoundError(f"Movie model not found at: {movie_model_path}")
    
    # Product model
    product_model_path = "./models/navi_product/full_HVB_seed42_cleaned_tau0.02_0.14_percentile_epoch_2"
    if os.path.exists(product_model_path):
        print(f"Loading Product model from: {product_model_path}")
        models['product'] = NaviForMaskedLM(product_model_path)
        models['product'] = models['product'].to(device)
        models['product'].eval()
        print("✓ Product model loaded")
    else:
        raise FileNotFoundError(f"Product model not found at: {product_model_path}")
    
    return models


def load_classification_data(domain: str, perturbation: str = "default"):
    """
    Load classification test data for a domain with optional perturbation.
    
    Args:
        domain: 'Movie' or 'Product'
        perturbation: 'default', 'synonym', 'typo', or 'reordered'
    
    Returns:
        tuple: (data, target_col) where data is preprocessed list of dicts
    """
    domain_lower = domain.lower()
    
    # Determine file path based on perturbation
    if perturbation == "default":
        data_path = f"data/cleaned/{domain}/test/WDC_{domain_lower}_for_cls.jsonl"
    else:
        data_path = f"data/cleaned/{domain}/test/WDC_{domain_lower}_for_cls_{perturbation}.jsonl"
    
    if domain == 'Product':
        target_col = 'category'
        print(f"\nLoading Product classification data from: {data_path}")
        wdc_data = load_data(data_path)
        print(f"Loaded {len(wdc_data)} rows")
        
        # Verify target columns (may need to check for synonym/typo variants)
        # For synonym/typo, the target column name might have changed
        # Try to find the target column by checking common variants
        if target_col not in wdc_data[0] if wdc_data else {}:
            # Try to find category column (might be corrupted or replaced)
            for key in wdc_data[0].keys() if wdc_data else []:
                if 'categor' in key.lower() or 'categ' in key.lower():
                    target_col = key
                    print(f"Using target column: {target_col}")
                    break
        
        wdc_data = verify_target_columns_present(wdc_data, target_columns=[target_col])
        
        # Preprocess
        wdc_data = preprocess_wdc_product(wdc_data)
        wdc_data = stratified_sample(wdc_data, target_col)
        
        print(f"Final Product data: {len(wdc_data)} rows")
        return wdc_data, target_col
    
    else:  # Movie
        target_col = 'genres'
        print(f"\nLoading Movie classification data from: {data_path}")
        wdc_data = load_data(data_path)
        print(f"Loaded {len(wdc_data)} rows")
        
        # Verify target columns (may need to check for synonym/typo variants)
        if target_col not in wdc_data[0] if wdc_data else {}:
            # Try to find genres column
            for key in wdc_data[0].keys() if wdc_data else []:
                if 'genre' in key.lower() or 'genr' in key.lower():
                    target_col = key
                    print(f"Using target column: {target_col}")
                    break
        
        wdc_data = verify_target_columns_present(wdc_data, target_columns=[target_col])
        
        # Preprocess
        wdc_data = preprocess_wdc_movie(wdc_data)
        wdc_data = stratified_sample(wdc_data, target_col)
        
        print(f"Final Movie data: {len(wdc_data)} rows")
        return wdc_data, target_col


def load_masked_prediction_data(domain: str, perturbation: str = "default"):
    """
    Load masked prediction test data for a domain with optional perturbation.
    
    Args:
        domain: 'Movie' or 'Product'
        perturbation: 'default', 'synonym', 'typo', or 'reordered'
    
    Returns:
        NaviDataset: Dataset for masked prediction evaluation
    """
    domain_lower = domain.lower()
    
    # Determine file path based on perturbation
    if perturbation == "default":
        data_path = f"./data/cleaned/{domain}/test/WDC_{domain_lower}_for_mp.jsonl"
    else:
        data_path = f"./data/cleaned/{domain}/test/WDC_{domain_lower}_for_mp_{perturbation}.jsonl"
    
    print(f"\nLoading {domain} masked prediction data from: {data_path}")
    raw_json = load_data(data_path)
    print(f"Loaded {len(raw_json)} rows")
    
    # Sample 1000 rows with fixed random seed for reproducibility
    random.seed(42)
    if len(raw_json) > 1000:
        raw_json = random.sample(raw_json, 1000)
        print(f"Sampled 1000 rows (seed=42)")
    else:
        print(f"Using all {len(raw_json)} rows (less than 1000 available)")
    
    # Create tuples with sequential table IDs for NaviDataset
    json_data = [(i, row) for i, row in enumerate(raw_json)]
    navi_dataset = NaviDataset(json_data)
    
    return navi_dataset


def evaluate_classification(model, model_name, test_data, test_domain, target_col, n_runs=5):
    """
    Evaluate classification task.
    
    Args:
        model: The model to evaluate
        model_name: Name of the model (for logging)
        test_data: Test dataset
        test_domain: Domain of test data ('Movie' or 'Product')
        target_col: Target column name
        n_runs: Number of evaluation runs
    
    Returns:
        float: Mean F1 score
    """
    print(f"\n{'='*60}")
    print(f"Classification: {model_name} on {test_domain} data")
    print(f"{'='*60}")
    
    result = run_repeated_classification(
        dataset=test_data,
        target_col=target_col,
        model=model,
        model_name=model_name,
        domain=test_domain.lower(),
        ml_model="xgboost",  # Use XGBoost as specified
        n_runs=n_runs,
        embedding_type="cls"
    )
    
    return result['mean']


def evaluate_imputation(model, model_name, test_dataset, test_domain, epoch=5):
    """
    Evaluate masked prediction (imputation) task.
    
    Args:
        model: The model to evaluate
        model_name: Name of the model (for logging)
        test_dataset: NaviDataset for masked prediction
        test_domain: Domain of test data ('Movie' or 'Product')
        epoch: Epoch number for collator (5 for value prediction)
    
    Returns:
        float: Accuracy score
    """
    print(f"\n{'='*60}")
    print(f"Imputation: {model_name} on {test_domain} data (epoch={epoch})")
    print(f"{'='*60}")
    
    collator = CollatorForMaskedPrediction(tokenizer=tokenizer)
    accuracy = evaluate_masked_prediction(
        dataset=test_dataset,
        model=model,
        tokenizer=tokenizer,
        collator=collator,
        epoch=epoch
    )
    
    return accuracy


def run_schema_noise_experiment():
    """
    Run schema noise robustness experiments.
    
    Returns:
        dict: Results dictionary with all metrics
    """
    results = {}
    
    # Load models
    print("\n" + "="*60)
    print("Loading Models")
    print("="*60)
    models = load_models()
    
    # Perturbations to evaluate
    perturbations = ['default', 'synonym', 'typo', 'reordered']
    perturbation_names = {
        'default': 'Default',
        'synonym': 'Synonym',
        'typo': 'Typo',
        'reordered': 'Column Reordered'
    }
    
    # Domains
    domains = ['Product', 'Movie']
    
    # Run evaluations
    print("\n" + "="*60)
    print("Running Evaluations")
    print("="*60)
    
    for perturbation in perturbations:
        print(f"\n{'='*80}")
        print(f"Perturbation: {perturbation_names[perturbation]}")
        print(f"{'='*80}")
        
        for domain in domains:
            domain_lower = domain.lower()
            model = models[domain_lower]
            model_name = f'navi_{domain_lower}'
            
            # Classification evaluation
            try:
                cls_data, target_col = load_classification_data(domain, perturbation)
                cls_result = evaluate_classification(
                    model=model,
                    model_name=model_name,
                    test_data=cls_data,
                    test_domain=domain,
                    target_col=target_col,
                    n_runs=5
                )
                
                # Store result
                key = f"{domain_lower}_cls_{perturbation}"
                results[key] = cls_result
                
                # Clear memory
                del cls_data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"⚠️  Error in classification evaluation: {e}")
                key = f"{domain_lower}_cls_{perturbation}"
                results[key] = None
            
            # Imputation evaluation
            try:
                mp_dataset = load_masked_prediction_data(domain, perturbation)
                imp_result = evaluate_imputation(
                    model=model,
                    model_name=model_name,
                    test_dataset=mp_dataset,
                    test_domain=domain,
                    epoch=5
                )
                
                # Store result
                key = f"{domain_lower}_imp_{perturbation}"
                results[key] = imp_result
                
                # Clear memory
                del mp_dataset
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"⚠️  Error in imputation evaluation: {e}")
                key = f"{domain_lower}_imp_{perturbation}"
                results[key] = None
    
    return results


def print_results_table(results):
    """
    Print results in table format matching Table 5.
    
    Args:
        results: Dictionary with evaluation results
    """
    print("\n" + "="*80)
    print("Schema Noise Robustness Results (Table 5)")
    print("="*80)
    
    perturbations = ['default', 'synonym', 'typo', 'reordered']
    perturbation_names = {
        'default': 'Default',
        'synonym': 'Synonym',
        'typo': 'Typo',
        'reordered': 'Column Reordered'
    }
    
    # Print header
    print(f"\n{'Perturbation':<20} {'Prod (Cls)':<15} {'Prod (Imp)':<15} {'Mov (Cls)':<15} {'Mov (Imp)':<15}")
    print("-" * 80)
    
    # Print rows
    for perturbation in perturbations:
        name = perturbation_names[perturbation]
        prod_cls = results.get(f'product_cls_{perturbation}', None)
        prod_imp = results.get(f'product_imp_{perturbation}', None)
        mov_cls = results.get(f'movie_cls_{perturbation}', None)
        mov_imp = results.get(f'movie_imp_{perturbation}', None)
        
        prod_cls_str = f"{prod_cls:.4f}" if prod_cls is not None else "N/A"
        prod_imp_str = f"{prod_imp:.4f}" if prod_imp is not None else "N/A"
        mov_cls_str = f"{mov_cls:.4f}" if mov_cls is not None else "N/A"
        mov_imp_str = f"{mov_imp:.4f}" if mov_imp is not None else "N/A"
        
        print(f"{name:<20} {prod_cls_str:<15} {prod_imp_str:<15} {mov_cls_str:<15} {mov_imp_str:<15}")


def save_results(results, output_path):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Schema Noise Robustness Experiment'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='experiments/logs',
        help='Directory to save results (default: experiments/logs)'
    )
    
    args = parser.parse_args()
    
    print("="*80)
    print("Schema Noise Robustness Experiment")
    print("="*80)
    
    # Run experiments
    results = run_schema_noise_experiment()
    
    # Print results table
    print_results_table(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"schema_noise_robustness_{timestamp}.json")
    save_results(results, output_path)
    
    print("\n" + "="*80)
    print("Experiment completed!")
    print("="*80)


if __name__ == "__main__":
    main()

