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
import json
import random
import numpy as np
from pathlib import Path
sys.path.append(os.path.abspath(".."))

DATA_ROOT = "data/"

import warnings
from transformers import logging as transformers_logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast

transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

from experiments.masked_prediction.mp_utils import load_data
from dataset.dataset import NaviDataset
from dataset.collator import NaviCollator
from dataset.preprocess import load_jsonl
from model.navi import NaviForMaskedLM
from model.loss import (
    MLMLoss,
    EntropyAwareContrastiveLoss,
    get_loss_weights,
    extract_header_value_embeddings
)
from transformers import BertTokenizer, BertConfig
from config import config

_checkpoint_epoch = getattr(config, "CHECKPOINT_EPOCH", 4)


def find_epoch_path(base_path, epoch):
    """Find epoch directory using glob pattern"""
    pattern = os.path.join(base_path, f"*epoch_{epoch}")
    matches = glob.glob(pattern)
    if matches:
        # Prefer the most specific match (longest path)
        matches.sort(key=len, reverse=True)
        return matches[0]
    return None


def load_training_variant_model(domain, epoch=None, header_encoder_mode='full', seed=None):
    """
    Load a training variant model.
    
    Args:
        domain: Domain name ('movie' or 'product')
        epoch: Epoch number (1, 2, 3, or 4). If None, uses config CHECKPOINT_EPOCH.
        header_encoder_mode: 'full', 'frozen', or 'partial'
        seed: Training seed (None, 0, 1, or 2). If provided, loads from seed-specific directory.
    
    Returns:
        Model instance or None if not found
    """
    domain_lower = domain.lower()
    
    # Determine model directory based on variant
    if seed is not None:
        # Seed-specific models: navi_{domain}_seed{seed}
        base_path = f'./models/navi_{domain_lower}_seed{seed}'
        if epoch is not None and epoch in [1, 2, 3, 4]:
            model_name = f'navi_{domain_lower}_seed{seed}_epoch{epoch}'
        else:
            model_name = f'navi_{domain_lower}_seed{seed}'
    elif header_encoder_mode == 'frozen':
        base_path = f'./models/navi_{domain_lower}_hefrozen'
        model_name = f'navi_{domain_lower}_hefrozen'
    elif header_encoder_mode == 'partial':
        base_path = f'./models/navi_{domain_lower}_hepartial'
        model_name = f'navi_{domain_lower}_hepartial'
    elif epoch is not None and epoch in [1, 2, 3, 4]:
        # E1, E2, E3, or E4 from default_3epoch (or 4-epoch run)
        base_path = f'./models/navi_{domain_lower}_default_3epoch'
        model_name = f'navi_{domain_lower}_epoch{epoch}'
    else:
        # Default E2
        base_path = f'./models/navi_{domain_lower}'
        model_name = f'navi_{domain_lower}'
    
    # Find epoch directory
    target_epoch = epoch if epoch is not None else _checkpoint_epoch
    
    # For seed-specific models, use pattern matching for checkpoint paths
    if seed is not None:
        # Pattern: {base_path}/full_HVB_seed{seed}_*_epoch_{target_epoch}
        pattern = os.path.join(base_path, f"*seed{seed}*_epoch_{target_epoch}")
        matches = glob.glob(pattern)
        if matches:
            # Prefer the most specific match (longest path)
            matches.sort(key=len, reverse=True)
            model_path = matches[0]
        else:
            # Fallback to find_epoch_path
            model_path = find_epoch_path(base_path, target_epoch)
    else:
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


def load_validation_data(validation_dir, sample_ratio=1.0):
    """
    Load validation data from JSON files in validation directory.
    
    Args:
        validation_dir: Path to validation directory
        sample_ratio: Ratio of data to sample (default: 1.0 for all data)
    
    Returns:
        List of tuples (table_id, row_dict)
    """
    if not os.path.exists(validation_dir):
        print(f"⚠️  Validation directory not found: {validation_dir}")
        return []
    
    # Load all JSON files
    val_files = sorted([
        os.path.join(validation_dir, f)
        for f in os.listdir(validation_dir)
        if f.endswith(".json") or f.endswith(".jsonl")
    ])
    
    if not val_files:
        print("⚠️  No validation files found")
        return []
    
    print(f"📂 Found {len(val_files)} validation files")
    
    # Load data from all files
    val_data = []
    loaded_count = 0
    for val_file in val_files:
        try:
            # Try JSONL format first (many .json files are actually JSONL format)
            data = load_jsonl(val_file)
            val_data.extend([(0, row) for row in data])
            loaded_count += 1
        except Exception as e:
            # If JSONL fails, try JSON format (single JSON object or array)
            try:
                with open(val_file, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                    if isinstance(content, list):
                        data = content
                    else:
                        data = [content]
                val_data.extend([(0, row) for row in data])
                loaded_count += 1
            except Exception as e2:
                # Skip files that fail both formats
                continue
    
    if loaded_count < len(val_files):
        print(f"⚠️  Successfully loaded {loaded_count}/{len(val_files)} validation files")
    
    # Sample data if needed
    total_samples = len(val_data)
    if val_data and sample_ratio < 1.0:
        num_samples = max(1, int(len(val_data) * sample_ratio))
        random.seed(42)
        val_data = random.sample(val_data, num_samples)
        print(f"📊 Sampled {len(val_data)}/{total_samples} validation samples ({sample_ratio*100:.0f}% of total)")
    else:
        print(f"📊 Loaded {len(val_data)} validation samples")
    
    return val_data


def calculate_validation_loss(model, validation_data, device, loss_fns, collator, ablation_type="full", 
                             batch_size=16, low_entropy_tau=0.13, high_entropy_tau=0.07, alignment_weight=0.5,
                             entropy_threshold_method="quartile", low_threshold_percentile=25, high_threshold_percentile=75):
    """
    Calculate validation loss on validation data.
    
    Args:
        model: Model instance
        validation_data: List of tuples (table_id, row_dict)
        device: Device to run on
        loss_fns: Dictionary with 'mlm_loss' and 'entropy_loss' functions
        collator: Data collator
        ablation_type: Ablation type ('full', 'woSSI', 'woMSM', 'woESA', 'woGHA', 'woGHC')
        batch_size: Batch size for evaluation
        low_entropy_tau: Temperature for low entropy loss
        high_entropy_tau: Temperature for high entropy loss
        alignment_weight: Alignment weight for loss combination
        entropy_threshold_method: Method for calculating entropy thresholds ('quartile' or 'percentile')
        low_threshold_percentile: Low threshold percentile for entropy categorization
        high_threshold_percentile: High threshold percentile for entropy categorization
    
    Returns:
        Dictionary with validation metrics
    """
    if not validation_data:
        return None
    
    # Create validation dataset with field entropy computation enabled
    val_dataset = NaviDataset(
        validation_data,
        tokenizer_name=config.BERT_NAME,
        max_length=config.MAX_SEQ_LENGTH,
        ablation_mode=ablation_type,
        compute_field_entropy=True,  # Enable field entropy computation for entropy loss
        entropy_threshold_method=entropy_threshold_method,
        low_threshold_percentile=low_threshold_percentile,
        high_threshold_percentile=high_threshold_percentile
    )
    
    # Get field categories from the dataset
    field_categories = val_dataset.get_field_categories()
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )
    
    total_loss = 0.0
    mlm_loss_total = 0.0
    entropy_loss_total = 0.0
    num_batches = 0
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Calculating validation loss", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            position_ids = batch["position_ids"].to(device)
            labels = batch["labels"].to(device)
            segment_ids = batch.get("segment_ids", None)
            if segment_ids is not None:
                segment_ids = segment_ids.to(device)
            
            header_strings = batch.get("header_strings")
            unmasked_input_ids = batch["unmasked_input_ids"].to(device)
            
            with autocast("cuda"):
                # Forward pass for MLM
                _, mlm_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                    header_strings=header_strings,
                )
                
                # Forward pass for contextualized embeddings
                contextualized_embeddings, _ = model(
                    input_ids=unmasked_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                    header_strings=header_strings,
                )
                
                # Extract embeddings
                header_embeds, val_embeds = extract_header_value_embeddings(
                    contextualized_embeddings, batch["header_positions"], batch["value_positions"]
                )
                
                header_encoder = model.bert.embeddings.header_encoder
                header_anchor_embeds, _ = header_encoder(header_strings)
                
                # Compute MLM loss
                mlm_loss = loss_fns["mlm_loss"](mlm_logits, labels)
                
                # Compute entropy loss (if not woESA)
                entropy_contrast_loss = None
                if ablation_type != "woESA":
                    table_ids = batch.get("table_ids")
                    entropy_loss_result = loss_fns["entropy_loss"](
                        E_univ=header_anchor_embeds,
                        H_ctx=header_embeds,
                        V_ctx=val_embeds,
                        header_strings=header_strings,
                        table_ids=table_ids,
                        field_categories=field_categories,  # Use computed field categories
                        return_dict=True
                    )
                    
                    if isinstance(entropy_loss_result, dict):
                        entropy_contrast_loss = entropy_loss_result["total"]
                    else:
                        entropy_contrast_loss = entropy_loss_result
                
                # Combine losses
                all_losses = [mlm_loss, entropy_contrast_loss]
                weights = get_loss_weights(collator.get_current_strategy(), ablation_type, alignment_weight)
                total_batch_loss = sum(w * l for w, l in zip(weights, all_losses) if l is not None)
                
                total_loss += total_batch_loss.item()
                mlm_loss_total += mlm_loss.item() if mlm_loss is not None else 0.0
                entropy_loss_total += entropy_contrast_loss.item() if entropy_contrast_loss is not None else 0.0
                num_batches += 1
    
    if num_batches == 0:
        return None
    
    avg_total_loss = total_loss / num_batches
    avg_mlm_loss = mlm_loss_total / num_batches
    avg_entropy_loss = entropy_loss_total / num_batches
    
    return {
        'total_loss': avg_total_loss,
        'mlm_loss': avg_mlm_loss,
        'entropy_loss': avg_entropy_loss,
        'num_batches': num_batches
    }


def evaluate_training_variants(validation_dir, tokenizer, domain):
    """
    Evaluate all training variants for validation loss.
    
    Args:
        validation_dir: Path to validation directory
        tokenizer: Tokenizer instance
        domain: Domain name ('Movie' or 'Product')
    
    Returns:
        Dictionary mapping variant names to validation loss
    """
    domain_lower = domain.lower()
    results = {}
    
    # Load validation data once
    validation_data = load_validation_data(validation_dir, sample_ratio=1.0)
    if not validation_data:
        print("⚠️  No validation data loaded")
        return results
    
    # Initialize loss functions (will be recreated for each model)
    # Default training parameters
    low_entropy_tau = 0.13
    high_entropy_tau = 0.07
    alignment_weight = 0.5
    
    # Variants to evaluate
    variants = [
        ('Default (E2)', None, 'full'),
        # ('E1', 1, 'full'),
        # ('E3', 3, 'full'),
        # ('Header Enc Frozen', None, 'frozen'),
        # ('Header Enc Partial', None, 'partial'),
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
        
        # Initialize loss functions for this model
        loss_fns = {
            "mlm_loss": MLMLoss(),
            "entropy_loss": EntropyAwareContrastiveLoss(model, low_entropy_tau=low_entropy_tau, high_entropy_tau=high_entropy_tau)
        }
        
        # Create collator (use HVB strategy for validation, matching training)
        collator = NaviCollator(
            tokenizer=tokenizer,
            stage_config=[('HVB', 1)],  # Use HVB strategy (matches training)
            word_level_mask_fraction=0.25,
            token_length_threshold=8,
            ablation_mode="full",
            field_categories=None,
            hv_weight=0.8,
            value_ratio=0.5
        )
        collator.set_epoch(0)  # Set epoch for strategy selection
        
        # Calculate validation loss
        # Use default entropy threshold parameters (matching training defaults)
        val_metrics = calculate_validation_loss(
            model, validation_data, device, loss_fns, collator,
            ablation_type="full",
            batch_size=16,
            low_entropy_tau=low_entropy_tau,
            high_entropy_tau=high_entropy_tau,
            alignment_weight=alignment_weight,
            entropy_threshold_method="quartile",  # Default method
            low_threshold_percentile=25,  # Default percentile
            high_threshold_percentile=75  # Default percentile
        )
        
        if val_metrics:
            results[variant_name] = val_metrics
            print(f"✅ Validation Loss: {val_metrics['total_loss']:.4f}")
            print(f"   MLM Loss: {val_metrics['mlm_loss']:.4f}")
            print(f"   Entropy Loss: {val_metrics['entropy_loss']:.4f}")
        else:
            print(f"⚠️  Failed to calculate validation loss")
        
        # Clear memory
        del model
        del loss_fns
        del collator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def evaluate_seed_variance(validation_dir, tokenizer, domain, seeds=[0, 1, 2], epoch=2):
    """
    Evaluate training variants across multiple seeds and calculate variance statistics.
    
    Args:
        validation_dir: Path to validation directory
        tokenizer: Tokenizer instance
        domain: Domain name ('Movie' or 'Product')
        seeds: List of seeds to evaluate (default: [0, 1, 2])
        epoch: Epoch number to evaluate (default: 2)
    
    Returns:
        Dictionary mapping variant names to seed variance results
    """
    domain_lower = domain.lower()
    results = {}
    
    # Load validation data once
    validation_data = load_validation_data(validation_dir, sample_ratio=1.0)
    if not validation_data:
        print("⚠️  No validation data loaded")
        return results
    
    # Default training parameters
    low_entropy_tau = 0.13
    high_entropy_tau = 0.07
    alignment_weight = 0.5
    
    # Variants to evaluate (only Default E2 for seed variance)
    variants = [
        ('Default (E2)', epoch, 'full'),
    ]
    
    for variant_name, variant_epoch, header_mode in variants:
        print(f"\n{'='*60}")
        print(f"Evaluating Seed Variance: {variant_name}")
        print(f"Seeds: {seeds}")
        print(f"{'='*60}")
        
        # Collect results for each seed
        seed_results = {}
        per_seed_metrics = {}
        
        for seed in seeds:
            print(f"\n  Evaluating seed {seed}...")
            
            # Load model for this seed
            model, model_name = load_training_variant_model(
                domain_lower, 
                epoch=variant_epoch, 
                header_encoder_mode=header_mode,
                seed=seed
            )
            
            if model is None:
                print(f"  ⚠️  Skipping seed {seed} - model not found")
                continue
            
            # Initialize loss functions for this model
            loss_fns = {
                "mlm_loss": MLMLoss(),
                "entropy_loss": EntropyAwareContrastiveLoss(model, low_entropy_tau=low_entropy_tau, high_entropy_tau=high_entropy_tau)
            }
            
            # Create collator (use HVB strategy for validation, matching training)
            collator = NaviCollator(
                tokenizer=tokenizer,
                stage_config=[('HVB', 1)],  # Use HVB strategy (matches training)
                word_level_mask_fraction=0.25,
                token_length_threshold=8,
                ablation_mode="full",
                field_categories=None,
                hv_weight=0.8,
                value_ratio=0.5
            )
            collator.set_epoch(0)  # Set epoch for strategy selection
            
            # Calculate validation loss
            val_metrics = calculate_validation_loss(
                model, validation_data, device, loss_fns, collator,
                ablation_type="full",
                batch_size=16,
                low_entropy_tau=low_entropy_tau,
                high_entropy_tau=high_entropy_tau,
                alignment_weight=alignment_weight,
                entropy_threshold_method="quartile",
                low_threshold_percentile=25,
                high_threshold_percentile=75
            )
            
            if val_metrics:
                seed_results[seed] = val_metrics
                per_seed_metrics[seed] = val_metrics
                print(f"  ✅ Seed {seed} - Total Loss: {val_metrics['total_loss']:.4f}")
            else:
                print(f"  ⚠️  Failed to calculate validation loss for seed {seed}")
            
            # Clear memory
            del model
            del loss_fns
            del collator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if not seed_results:
            print(f"⚠️  No valid results for {variant_name}")
            continue
        
        # Calculate statistics across seeds
        metrics_to_aggregate = ['total_loss', 'mlm_loss', 'entropy_loss']
        aggregated = {}
        
        for metric_name in metrics_to_aggregate:
            values = [seed_results[seed][metric_name] for seed in seed_results.keys() if metric_name in seed_results[seed]]
            if values:
                aggregated[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values,
                    'num_seeds': len(values)
                }
        
        # Store results
        results[variant_name] = {
            'per_seed': per_seed_metrics,
            'aggregated': aggregated
        }
        
        # Print summary for this variant
        if 'total_loss' in aggregated:
            stats = aggregated['total_loss']
            print(f"\n  📊 Seed Variance Statistics for {variant_name}:")
            print(f"     Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"     Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"     Seeds evaluated: {stats['num_seeds']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Calculate validation loss for training variants')
    parser.add_argument('--domain', choices=['Movie', 'Product'], 
                       required=True, help='Domain to evaluate on')
    parser.add_argument('--seeds', type=int, nargs='+', default=None,
                       help='List of seeds to evaluate for seed variance (e.g., --seeds 0 1 2). If provided, evaluates seed variance instead of single-seed variants.')
    parser.add_argument('--seed_variance', action='store_true', default=False,
                       help='Enable seed variance evaluation (uses default seeds [0, 1, 2])')
    parser.add_argument('--epoch', type=int, default=2,
                       help='Epoch number to evaluate (default: 2)')
    
    args = parser.parse_args()
    
    # Validation data path
    domain_lower = args.domain.lower()
    validation_dir = str(Path(DATA_ROOT) / "cleaned" / args.domain / "validation")
    
    # Tokenizer (local or online per RUN_ONLINE)
    try:
        from config import config as app_config
        tokenizer = BertTokenizer.from_pretrained(
            app_config.get_bert_name(),
            local_files_only=app_config.use_local_files_only()
        )
    except ImportError:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Determine if we should evaluate seed variance
    if args.seed_variance or args.seeds is not None:
        # Seed variance evaluation
        seeds = args.seeds if args.seeds is not None else [0, 1, 2]
        print(f"Calculating seed variance for {args.domain} domain")
        print(f"Seeds: {seeds}, Epoch: {args.epoch}")
        
        results = evaluate_seed_variance(validation_dir, tokenizer, args.domain, seeds=seeds, epoch=args.epoch)
        
        # Print seed variance summary
        print(f"\n{'='*80}")
        print(f"Seed Variance Results for {args.domain}")
        print(f"{'='*80}")
        for variant_name, variant_results in results.items():
            if 'aggregated' in variant_results and 'total_loss' in variant_results['aggregated']:
                stats = variant_results['aggregated']['total_loss']
                print(f"\n{variant_name}:")
                print(f"  Mean ± Std:  {stats['mean']:.4f} ± {stats['std']:.4f}")
                print(f"  Range:       [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"  Seeds:       {stats['num_seeds']}")
                
                # Print per-seed breakdown
                if 'per_seed' in variant_results:
                    print(f"  Per-seed breakdown:")
                    for seed in sorted(variant_results['per_seed'].keys()):
                        seed_metrics = variant_results['per_seed'][seed]
                        print(f"    Seed {seed}: {seed_metrics['total_loss']:.4f}")
            else:
                print(f"{variant_name}: No aggregated statistics available")
    else:
        # Single-seed evaluation (original behavior)
        print(f"Calculating validation loss for {args.domain} domain")
        results = evaluate_training_variants(validation_dir, tokenizer, args.domain)
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"Summary: Validation Loss for {args.domain}")
        print(f"{'='*80}")
        for variant_name, metrics in results.items():
            if isinstance(metrics, dict) and 'total_loss' in metrics:
                print(f"{variant_name:30s}: {metrics['total_loss']:.4f}")
            else:
                print(f"{variant_name:30s}: {metrics}")


if __name__ == "__main__":
    main()

