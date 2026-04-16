#!/usr/bin/env python3
"""
Cross-domain evaluation script for NAVI models.
Evaluates models trained on one domain (Movie/Product) on test data from the other domain.
"""

import torch
import numpy as np
import json
import random
import os
import argparse
import gc
from datetime import datetime
from pathlib import Path

DATA_ROOT = "data/"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Imports
import warnings
from transformers import logging as transformers_logging
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from experiments.masked_prediction.mp_utils import (
    load_data,
    evaluate_masked_prediction
)
from experiments.downstream_tasks.row_classification import (
    run_repeated_classification,
    preprocess_wdc_movie,
    preprocess_wdc_product,
    stratified_sample,
    verify_target_columns_present
)
from dataset.dataset import NaviDataset, BertDataset
from dataset.collator import CollatorForMaskedPrediction
from model.navi import NaviForMaskedLM
from transformers import BertTokenizer
from config import config as app_config

# Initialize tokenizer (local or online per RUN_ONLINE)
tokenizer = BertTokenizer.from_pretrained(
    app_config.get_bert_name(),
    local_files_only=app_config.use_local_files_only()
)


def load_cross_domain_models():
    """
    Load last-epoch checkpoints (config.CHECKPOINT_EPOCH) from navi_product and navi_movie.
    
    Returns:
        dict: Dictionary with 'product' and 'movie' model keys
    """
    try:
        from config import config as app_config
        checkpoint_epoch = getattr(app_config, "CHECKPOINT_EPOCH", 4)
    except ImportError:
        checkpoint_epoch = 4
    models = {}
    
    # Product model
    product_model_path = f"./models/navi_product_default_3epoch/full_HVB_seed42_cleaned_tau0.07_0.13_quartile_epoch_{checkpoint_epoch}"
    if os.path.exists(product_model_path):
        print(f"Loading Product model from: {product_model_path}")
        models['product'] = NaviForMaskedLM(product_model_path)
        models['product'] = models['product'].to(device)
        models['product'].eval()
        print("✓ Product model loaded")
    else:
        raise FileNotFoundError(f"Product model not found at: {product_model_path}")
    
    # Movie model
    movie_model_path = f"./models/navi_movie/full_HVB_seed42_cleaned_tau0.07_0.13_quartile_epoch_{checkpoint_epoch}"
    if os.path.exists(movie_model_path):
        print(f"Loading Movie model from: {movie_model_path}")
        models['movie'] = NaviForMaskedLM(movie_model_path)
        models['movie'] = models['movie'].to(device)
        models['movie'].eval()
        print("✓ Movie model loaded")
    else:
        raise FileNotFoundError(f"Movie model not found at: {movie_model_path}")
    
    return models


def load_classification_data(domain):
    """
    Load and preprocess classification test data for a domain.
    
    Args:
        domain: 'Movie' or 'Product'
    
    Returns:
        tuple: (data, target_col) where data is preprocessed list of dicts
    """
    if domain == 'Product':
        data_path = Path(DATA_ROOT) / "cleaned" / "Product" / "test" / "WDC_product_for_cls.jsonl"
        target_col = 'category'
        print(f"\nLoading Product classification data from: {data_path}")
        wdc_data = load_data(str(data_path))
        print(f"Loaded {len(wdc_data)} rows")
        
        # Verify target columns
        wdc_data = verify_target_columns_present(wdc_data, target_columns=['category'])
        
        # Preprocess
        wdc_data = preprocess_wdc_product(wdc_data)
        wdc_data = stratified_sample(wdc_data, "category")
        
        print(f"Final Product data: {len(wdc_data)} rows")
        return wdc_data, target_col
    
    else:  # Movie
        data_path = Path(DATA_ROOT) / "cleaned" / "Movie" / "test" / "WDC_movie_for_cls.jsonl"
        target_col = 'genres'
        print(f"\nLoading Movie classification data from: {data_path}")
        wdc_data = load_data(str(data_path))
        print(f"Loaded {len(wdc_data)} rows")
        
        # Verify target columns
        wdc_data = verify_target_columns_present(wdc_data, target_columns=['genres'])
        
        # Preprocess
        wdc_data = preprocess_wdc_movie(wdc_data)
        wdc_data = stratified_sample(wdc_data, "genres")
        
        print(f"Final Movie data: {len(wdc_data)} rows")
        return wdc_data, target_col


def load_masked_prediction_data(domain):
    """
    Load masked prediction test data for a domain.
    
    Args:
        domain: 'Movie' or 'Product'
    
    Returns:
        NaviDataset: Dataset for masked prediction evaluation
    """
    domain_lower = domain.lower()
    data_path = Path(DATA_ROOT) / "cleaned" / domain / "test" / f"WDC_{domain_lower}_for_mp.jsonl"
    
    print(f"\nLoading {domain} masked prediction data from: {data_path}")
    raw_json = load_data(str(data_path))
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


def evaluate_classification_cross_domain(model, model_name, test_data, test_domain, target_col, n_runs=5):
    """
    Evaluate classification task with cross-domain or same-domain setup.
    
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
        ml_model="xgboost",  # Using XGBoost for cross-domain classification
        n_runs=n_runs,
        embedding_type="cls"
    )
    
    return result['mean']


def evaluate_masked_prediction_cross_domain(model, model_name, test_dataset, test_domain, epoch=5):
    """
    Evaluate masked prediction task with cross-domain or same-domain setup.
    
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
    print(f"Masked Prediction (Imp): {model_name} on {test_domain} data (epoch={epoch})")
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


def run_cross_domain_evaluation():
    """
    Run all cross-domain and same-domain evaluations.
    
    Returns:
        dict: Results dictionary with all metrics
    """
    results = {}
    
    # Load models
    print("\n" + "="*60)
    print("Loading Models")
    print("="*60)
    models = load_cross_domain_models()
    
    # Load test data for both domains
    print("\n" + "="*60)
    print("Loading Test Data")
    print("="*60)
    
    # Classification data
    product_cls_data, product_target_col = load_classification_data('Product')
    movie_cls_data, movie_target_col = load_classification_data('Movie')
    
    # Masked prediction data
    product_mp_dataset = load_masked_prediction_data('Product')
    movie_mp_dataset = load_masked_prediction_data('Movie')
    
    # Run evaluations
    print("\n" + "="*60)
    print("Running Evaluations")
    print("="*60)
    
    # Classification evaluations
    print("\n>>> CLASSIFICATION EVALUATIONS <<<")
    
    # M->P (Cls): Movie model on Product classification data
    results['M->P (Cls)'] = evaluate_classification_cross_domain(
        model=models['movie'],
        model_name='navi_movie',
        test_data=product_cls_data,
        test_domain='Product',
        target_col=product_target_col,
        n_runs=5
    )
    
    # P->M (Cls): Product model on Movie classification data
    results['P->M (Cls)'] = evaluate_classification_cross_domain(
        model=models['product'],
        model_name='navi_product',
        test_data=movie_cls_data,
        test_domain='Movie',
        target_col=movie_target_col,
        n_runs=5
    )
    
    # P->P (Cls): Product model on Product classification data (same-domain)
    results['P->P (Cls)'] = evaluate_classification_cross_domain(
        model=models['product'],
        model_name='navi_product',
        test_data=product_cls_data,
        test_domain='Product',
        target_col=product_target_col,
        n_runs=5
    )
    
    # M->M (Cls): Movie model on Movie classification data (same-domain)
    results['M->M (Cls)'] = evaluate_classification_cross_domain(
        model=models['movie'],
        model_name='navi_movie',
        test_data=movie_cls_data,
        test_domain='Movie',
        target_col=movie_target_col,
        n_runs=5
    )
    
    # Masked prediction evaluations
    print("\n>>> MASKED PREDICTION EVALUATIONS <<<")
    
    # M->P (Imp): Movie model on Product masked prediction data
    results['M->P (Imp)'] = evaluate_masked_prediction_cross_domain(
        model=models['movie'],
        model_name='navi_movie',
        test_dataset=product_mp_dataset,
        test_domain='Product',
        epoch=5
    )
    
    # P->M (Imp): Product model on Movie masked prediction data
    results['P->M (Imp)'] = evaluate_masked_prediction_cross_domain(
        model=models['product'],
        model_name='navi_product',
        test_dataset=movie_mp_dataset,
        test_domain='Movie',
        epoch=5
    )
    
    # P->P (Imp): Product model on Product masked prediction data (same-domain)
    results['P->P (Imp)'] = evaluate_masked_prediction_cross_domain(
        model=models['product'],
        model_name='navi_product',
        test_dataset=product_mp_dataset,
        test_domain='Product',
        epoch=5
    )
    
    # M->M (Imp): Movie model on Movie masked prediction data (same-domain)
    results['M->M (Imp)'] = evaluate_masked_prediction_cross_domain(
        model=models['movie'],
        model_name='navi_movie',
        test_dataset=movie_mp_dataset,
        test_domain='Movie',
        epoch=5
    )
    
    return results


def print_results_table(results):
    """
    Print results in a table format matching Table 4.
    
    Args:
        results: Dictionary with evaluation results
    """
    print("\n" + "="*80)
    print("CROSS-DOMAIN EVALUATION RESULTS (Table 4)")
    print("="*80)
    print()
    print(f"{'Metric':<20} {'Value':<15}")
    print("-" * 35)
    print(f"{'M->P (Cls)':<20} {results.get('M->P (Cls)', 'N/A'):<15.4f}" if 'M->P (Cls)' in results else f"{'M->P (Cls)':<20} {'N/A':<15}")
    print(f"{'P->P (Cls)':<20} {results.get('P->P (Cls)', 'N/A'):<15.4f}" if 'P->P (Cls)' in results else f"{'P->P (Cls)':<20} {'N/A':<15}")
    print(f"{'M->P (Imp)':<20} {results.get('M->P (Imp)', 'N/A'):<15.4f}" if 'M->P (Imp)' in results else f"{'M->P (Imp)':<20} {'N/A':<15}")
    print(f"{'P->P (Imp)':<20} {results.get('P->P (Imp)', 'N/A'):<15.4f}" if 'P->P (Imp)' in results else f"{'P->P (Imp)':<20} {'N/A':<15}")
    print(f"{'P->M (Cls)':<20} {results.get('P->M (Cls)', 'N/A'):<15.4f}" if 'P->M (Cls)' in results else f"{'P->M (Cls)':<20} {'N/A':<15}")
    print(f"{'M->M (Cls)':<20} {results.get('M->M (Cls)', 'N/A'):<15.4f}" if 'M->M (Cls)' in results else f"{'M->M (Cls)':<20} {'N/A':<15}")
    print(f"{'P->M (Imp)':<20} {results.get('P->M (Imp)', 'N/A'):<15.4f}" if 'P->M (Imp)' in results else f"{'P->M (Imp)':<20} {'N/A':<15}")
    print(f"{'M->M (Imp)':<20} {results.get('M->M (Imp)', 'N/A'):<15.4f}" if 'M->M (Imp)' in results else f"{'M->M (Imp)':<20} {'N/A':<15}")
    print()
    print("="*80)


def save_results(results, output_file=None):
    """
    Save results to JSON file.
    
    Args:
        results: Dictionary with evaluation results
        output_file: Output file path (optional)
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"experiments/logs/cross_domain_evaluation_{timestamp}.json"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    json_results = {k: convert_numpy(v) for k, v in results.items()}
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_file}")


def main():
    """
    Main function to run cross-domain evaluation.
    """
    parser = argparse.ArgumentParser(description='Run cross-domain evaluation for NAVI models')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (optional)')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CROSS-DOMAIN EVALUATION FOR NAVI MODELS")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print("="*80)
    
    # Run evaluations
    results = run_cross_domain_evaluation()
    
    # Print results table
    print_results_table(results)
    
    # Save results
    save_results(results, args.output)
    
    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    print("\n✅ Cross-domain evaluation completed!")


if __name__ == "__main__":
    main()

