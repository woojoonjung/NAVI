import argparse
from tqdm import tqdm
import os
from multiprocessing import Pool
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import pickle
import random
import numpy as np
from datetime import datetime
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from dataset.dataset import NaviDataset, StratifiedBatchSampler
from dataset.collator import NaviCollator
from dataset.preprocess import load_jsonl, clean_table_data
from model.navi import NaviForMaskedLM
from config import config
from model.loss import (
    MLMLoss,
    EntropyAwareContrastiveLoss,
    get_loss_weights,
    extract_header_value_embeddings
)

# Move these functions to the top level (outside of any other function)
def count_lines(file_path):
    try:
        with open(file_path, 'r') as f:
            return (file_path, sum(1 for _ in f))
    except Exception as e:
        print(f"⚠️ Error reading {file_path}: {e}")
        return (file_path, 0)

def load_single_file(args):
    table_idx, jsonl_file = args
    try:
        data = load_jsonl(jsonl_file)
        table_data = [(table_idx, row) for row in data]
        return table_data
    except Exception as e:
        print(f"⚠️ Skipping file {jsonl_file} due to error: {e}")
        return []

def create_combined_dataset(jsonl_files, tokenizer_name, max_length, ablation_mode, compute_field_entropy=True,
                           entropy_threshold_method="quartile", low_threshold_percentile=25, high_threshold_percentile=75):
    """Create a combined dataset by merging multiple JSONL files into a single list of tuples."""
    all_data = []
    
    # Create arguments for multiprocessing
    file_args = [(idx, file) for idx, file in enumerate(jsonl_files)]
    
    # Use multiprocessing for parallel loading
    with Pool(processes=min(4, os.cpu_count())) as pool:
        results = pool.map(load_single_file, file_args)
    
    # Combine results
    for result in results:
        all_data.extend(result)
    
    if not all_data:
        raise ValueError("No valid data found from any files")

    # all_data = clean_table_data(
    #     all_data,
    #     tokenizer_name=tokenizer_name,
    #     max_tokens=500,
    #     max_indexed_fields=3,
    #     max_tokens_per_field=20
    # )
    
    print(f"🔄 Total combined data: {len(all_data)} rows")
    
    # Create single NaviDataset with all merged data as tuples
    dataset = NaviDataset(
        all_data, 
        tokenizer_name=tokenizer_name, 
        max_length=max_length, 
        ablation_mode=ablation_mode, 
        compute_field_entropy=compute_field_entropy,
        entropy_threshold_method=entropy_threshold_method,
        low_threshold_percentile=low_threshold_percentile,
        high_threshold_percentile=high_threshold_percentile
    )
    return dataset

def load_or_tokenize(jsonl_file, tokenizer_name, max_length, ablation_mode, cache_dir=".cache_tokenized",
                    compute_field_entropy=False, entropy_threshold_method="quartile", 
                    low_threshold_percentile=25, high_threshold_percentile=75):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, os.path.basename(jsonl_file).replace(".jsonl", ".pkl"))

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load cache: {cache_file} — {e}, regenerating...")

    data = load_jsonl(jsonl_file)
    dataset = NaviDataset(
        data, 
        tokenizer_name=tokenizer_name, 
        max_length=max_length, 
        ablation_mode=ablation_mode,
        compute_field_entropy=compute_field_entropy,
        entropy_threshold_method=entropy_threshold_method,
        low_threshold_percentile=low_threshold_percentile,
        high_threshold_percentile=high_threshold_percentile
    )
    with open(cache_file, "wb") as f:
        pickle.dump(dataset, f)
    return dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train ATLAS with Masked Language Modeling")
    parser.add_argument("--data_path", type=str, required=True, help="Path to directory containing JSONL files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--masking_strategy", type=str, required=True, 
                       choices=['HV', 'B', 'HVB'],
                       help="Masking strategy to use")
    parser.add_argument("--ablation_type", type=str, required=True,
                       choices=['full', 'woESA', 'woSSI', "woMSM"],
                       help="Ablation type to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--tables_per_batch", type=int, default=4, help="Number of tables to process simultaneously")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Number of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--log_steps", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--hv_weight", type=float, default=0.5, help="Weight for header-value masking in HVB strategy")
    parser.add_argument("--alignment_weight", type=float, default=0.15, help="Weight for alignment loss")
    parser.add_argument("--value_ratio", type=float, default=0.5, help="Ratio of value segments to header segments for HV masking")
    parser.add_argument("--low_entropy_tau", type=float, default=0.13, help="Temperature parameter for low entropy contrastive loss")
    parser.add_argument("--high_entropy_tau", type=float, default=0.07, help="Temperature parameter for high entropy contrastive loss")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--validation_dir", type=str, default=None, help="Path to validation data directory (optional)")
    parser.add_argument("--data_type", type=str, default="cleaned", choices=["raw", "cleaned"], help="Data type used for training (for model naming)")
    
    # Entropy threshold parameters
    parser.add_argument("--entropy_threshold_method", type=str, default="quartile", 
                       choices=["quartile", "percentile"], 
                       help="Method for calculating entropy thresholds (default: quartile)")
    parser.add_argument("--low_threshold_percentile", type=int, default=25, 
                       help="Low threshold percentile for entropy categorization (default: 25)")
    parser.add_argument("--high_threshold_percentile", type=int, default=75, 
                       help="High threshold percentile for entropy categorization (default: 75)")
    parser.add_argument("--validation_sample_ratio", type=float, default=1.0,
                       help="Ratio of validation data to use (1.0 = full dataset, default: 1.0)")
    parser.add_argument("--header_encoder_mode", type=str, default="full",
                       choices=["full", "frozen", "partial"],
                       help="Header encoder training mode: full (all trainable), frozen (all frozen), partial (only layer_2 trainable)")
    
    return parser.parse_args()

def get_stage_config(masking_strategy, ablation_type):
    # For woMSM ablation, force BERT-style masking regardless of input strategy
    if ablation_type == 'woMSM':
        return [('B', 2)]
    if masking_strategy == 'HV':
        return [('HV', 2)]
    elif masking_strategy == 'B':
        return [('B', 2)]
    elif masking_strategy == 'HVB':
        return [('HVB', 2)]
    else:
        raise ValueError(f"Unknown masking strategy: {masking_strategy}")

class LossLogger:
    """Logger for tracking training metrics in a structured format."""
    def __init__(self, output_dir, ablation_type, masking_strategy, seed=None, data_type="cleaned", 
                 low_entropy_tau=None, high_entropy_tau=None, entropy_threshold_method=None):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.ablation_type = ablation_type
        self.masking_strategy = masking_strategy
        self.seed = seed
        self.data_type = data_type
        self.low_entropy_tau = low_entropy_tau
        self.high_entropy_tau = high_entropy_tau
        self.entropy_threshold_method = entropy_threshold_method
        
        # Build model identifier for file naming
        model_id_parts = [ablation_type, masking_strategy]
        if seed is not None:
            model_id_parts.append(f"seed{seed}")
        if data_type:
            model_id_parts.append(data_type)
        if low_entropy_tau is not None and high_entropy_tau is not None:
            model_id_parts.append(f"tau{high_entropy_tau}_{low_entropy_tau}")
        if entropy_threshold_method:
            model_id_parts.append(entropy_threshold_method)
        
        model_id = "_".join(model_id_parts)
        
        self.metrics = {
            'training_info': {
                'ablation_type': ablation_type,
                'masking_strategy': masking_strategy,
                'seed': seed,
                'data_type': data_type,
                'low_entropy_tau': low_entropy_tau,
                'high_entropy_tau': high_entropy_tau,
                'entropy_threshold_method': entropy_threshold_method,
                'start_time': datetime.now().isoformat(),
                'epochs': [],
                'validation': [],
                'compute_metrics': {},
                'dataset_stats': {}
            },
            'steps': [] 
        }
        self.metrics_file = os.path.join(output_dir, f"training_metrics_{model_id}.json")
        self.plot_file = os.path.join(output_dir, f"learning_curve_{model_id}.png")

    def log_epoch(self, epoch, epoch_metrics, stage_info, current_strategy):
        epoch_data = {
            'epoch': epoch + 1,
            'stage': stage_info['current_stage'],
            'total_stages': stage_info['total_stages'],
            'strategy': current_strategy,
            'epochs_in_stage': stage_info['epochs_in_stage'],
            'total_epochs_in_stage': stage_info['total_epochs_in_stage'],
            'metrics': epoch_metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.metrics['training_info']['epochs'].append(epoch_data)
        self.save_metrics()

    def log_step(self, global_step, step_metrics):
        step_data = {
            'step': global_step,
            'metrics': step_metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.metrics['steps'].append(step_data)
        self.save_metrics()
    
    def log_compute_metrics(self, compute_metrics):
        """Log compute metrics like timing, throughput, etc."""
        self.metrics['training_info']['compute_metrics'].update(compute_metrics)
        self.save_metrics()

    def save_metrics(self):
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def get_metrics_file(self):
        return self.metrics_file
    
    def log_validation(self, epoch, validation_metrics):
        """Log validation metrics after each epoch."""
        val_data = {
            'epoch': epoch + 1,
            'metrics': validation_metrics,
            'timestamp': datetime.now().isoformat()
        }
        self.metrics['training_info']['validation'].append(val_data)
        self.save_metrics()
    
    def log_validation_step(self, step, validation_metrics):
        """Log validation metrics at a specific training step."""
        val_data = {
            'step': step,
            'metrics': validation_metrics,
            'timestamp': datetime.now().isoformat()
        }
        # Store step-based validations separately
        if 'validation_steps' not in self.metrics['training_info']:
            self.metrics['training_info']['validation_steps'] = []
        self.metrics['training_info']['validation_steps'].append(val_data)
        self.save_metrics()
    
    def plot_learning_curve(self, smoothing=0.1):
        """Plot learning curve with optional smoothing, including validation losses."""
        if not self.metrics['steps']:
            print("⚠️ No step data available for plotting")
            return
        
        # Extract training data
        steps = [s['step'] for s in self.metrics['steps']]
        total_losses = [s['metrics'].get('total_loss', 0) for s in self.metrics['steps']]
        mlm_losses = [s['metrics'].get('mlm_loss', 0) for s in self.metrics['steps'] if s['metrics'].get('mlm_loss') is not None]
        entropy_losses = [s['metrics'].get('entropy_contrast_loss', 0) for s in self.metrics['steps'] if s['metrics'].get('entropy_contrast_loss') is not None]
        low_entropy_losses = [s['metrics'].get('low_entropy_loss', 0) for s in self.metrics['steps'] if s['metrics'].get('low_entropy_loss') is not None]
        high_entropy_losses = [s['metrics'].get('high_entropy_loss', 0) for s in self.metrics['steps'] if s['metrics'].get('high_entropy_loss') is not None]
        
        # Extract step-based validation data (every 1000 steps) - filter out None values properly
        validation_steps_data = self.metrics['training_info'].get('validation_steps', [])
        
        # Extract validation data with proper filtering for each metric
        val_step_data = {
            'steps': [],
            'total_loss': [],
            'mlm_loss': [],
            'entropy_contrast_loss': [],
            'low_entropy_loss': [],
            'high_entropy_loss': []
        }
        
        for v in validation_steps_data:
            step = v['step']
            metrics = v['metrics']
            val_step_data['steps'].append(step)
            val_step_data['total_loss'].append(metrics.get('total_loss', 0))
            
            if metrics.get('mlm_loss') is not None:
                val_step_data['mlm_loss'].append((step, metrics['mlm_loss']))
            if metrics.get('entropy_contrast_loss') is not None:
                val_step_data['entropy_contrast_loss'].append((step, metrics['entropy_contrast_loss']))
            if metrics.get('low_entropy_loss') is not None:
                val_step_data['low_entropy_loss'].append((step, metrics['low_entropy_loss']))
            if metrics.get('high_entropy_loss') is not None:
                val_step_data['high_entropy_loss'].append((step, metrics['high_entropy_loss']))
        
        # Unpack paired data for entropy losses
        val_step_low_entropy_steps = [x[0] for x in val_step_data['low_entropy_loss']]
        val_step_low_entropy_vals = [x[1] for x in val_step_data['low_entropy_loss']]
        val_step_high_entropy_steps = [x[0] for x in val_step_data['high_entropy_loss']]
        val_step_high_entropy_vals = [x[1] for x in val_step_data['high_entropy_loss']]
        
        # Unpack for other metrics
        val_step_mlm_steps = [x[0] for x in val_step_data['mlm_loss']]
        val_step_mlm_vals = [x[1] for x in val_step_data['mlm_loss']]
        val_step_entropy_steps = [x[0] for x in val_step_data['entropy_contrast_loss']]
        val_step_entropy_vals = [x[1] for x in val_step_data['entropy_contrast_loss']]
        
        # Apply exponential moving average smoothing
        def smooth(values, alpha):
            if not values:
                return values
            smoothed = [values[0]]
            for v in values[1:]:
                smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
            return smoothed
        
        if smoothing > 0:
            total_losses = smooth(total_losses, 1 - smoothing)
            if mlm_losses:
                mlm_losses = smooth(mlm_losses, 1 - smoothing)
            if entropy_losses:
                entropy_losses = smooth(entropy_losses, 1 - smoothing)
            if low_entropy_losses:
                low_entropy_losses = smooth(low_entropy_losses, 1 - smoothing)
            if high_entropy_losses:
                high_entropy_losses = smooth(high_entropy_losses, 1 - smoothing)
        
        # Store plotting data for reproducibility
        plotting_data = {
            'training': {
                'steps': steps,
                'total_loss': total_losses,
                'mlm_loss': {
                    'steps': [s['step'] for s in self.metrics['steps'] if s['metrics'].get('mlm_loss') is not None],
                    'values': mlm_losses
                },
                'entropy_contrast_loss': {
                    'steps': [s['step'] for s in self.metrics['steps'] if s['metrics'].get('entropy_contrast_loss') is not None],
                    'values': entropy_losses
                },
                'low_entropy_loss': {
                    'steps': [s['step'] for s in self.metrics['steps'] if s['metrics'].get('low_entropy_loss') is not None],
                    'values': low_entropy_losses
                },
                'high_entropy_loss': {
                    'steps': [s['step'] for s in self.metrics['steps'] if s['metrics'].get('high_entropy_loss') is not None],
                    'values': high_entropy_losses
                }
            },
            'validation': {
                'total_loss': {
                    'steps': val_step_data['steps'],
                    'values': val_step_data['total_loss']
                },
                'mlm_loss': {
                    'steps': val_step_mlm_steps,
                    'values': val_step_mlm_vals
                },
                'entropy_contrast_loss': {
                    'steps': val_step_entropy_steps,
                    'values': val_step_entropy_vals
                },
                'low_entropy_loss': {
                    'steps': val_step_low_entropy_steps,
                    'values': val_step_low_entropy_vals
                },
                'high_entropy_loss': {
                    'steps': val_step_high_entropy_steps,
                    'values': val_step_high_entropy_vals
                }
            },
            'smoothing': smoothing
        }
        
        # Save plotting data to metrics
        self.metrics['training_info']['plotting_data'] = plotting_data
        self.save_metrics()
        
        # Create figure with subplots for all 4 losses
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Plot 1: Total Loss
        plot_idx = 0
        axes[plot_idx].plot(steps, total_losses, label='Train Total Loss', linewidth=2, color='blue')
        if val_step_data['steps']:
            axes[plot_idx].plot(val_step_data['steps'], val_step_data['total_loss'], 
                              label='Val Total Loss (Step)', linewidth=2, linestyle='--', 
                              marker='o', markersize=6, color='red', zorder=5)
        axes[plot_idx].set_xlabel('Training Step')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].set_title('Total Loss')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        
        # Plot 2: MLM Loss
        plot_idx = 1
        if mlm_losses:
            mlm_steps = [s['step'] for s in self.metrics['steps'] if s['metrics'].get('mlm_loss') is not None]
            axes[plot_idx].plot(mlm_steps, mlm_losses, label='Train MLM Loss', linewidth=2, color='cyan')
        if val_step_mlm_steps:
            axes[plot_idx].plot(val_step_mlm_steps, val_step_mlm_vals, label='Val MLM Loss (Step)', 
                              linewidth=2, linestyle='--', marker='o', markersize=6, color='red', zorder=5)
        axes[plot_idx].set_xlabel('Training Step')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].set_title('MLM Loss')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        
        # Plot 3: Low Entropy Loss
        plot_idx = 2
        if low_entropy_losses:
            low_entropy_steps = [s['step'] for s in self.metrics['steps'] if s['metrics'].get('low_entropy_loss') is not None]
            axes[plot_idx].plot(low_entropy_steps, low_entropy_losses, label='Train Low Entropy Loss', 
                              linewidth=2, color='green')
        if val_step_low_entropy_steps:
            assert len(val_step_low_entropy_steps) == len(val_step_low_entropy_vals), "Validation steps and values must be aligned"
            axes[plot_idx].plot(val_step_low_entropy_steps, val_step_low_entropy_vals, 
                              label='Val Low Entropy Loss (Step)', linewidth=2, linestyle='--', 
                              marker='o', markersize=6, color='red', zorder=5)
        axes[plot_idx].set_xlabel('Training Step')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].set_title('Low Entropy Loss')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        
        # Plot 4: High Entropy Loss
        plot_idx = 3
        if high_entropy_losses:
            high_entropy_steps = [s['step'] for s in self.metrics['steps'] if s['metrics'].get('high_entropy_loss') is not None]
            axes[plot_idx].plot(high_entropy_steps, high_entropy_losses, label='Train High Entropy Loss', 
                              linewidth=2, color='purple')
        if val_step_high_entropy_steps:
            assert len(val_step_high_entropy_steps) == len(val_step_high_entropy_vals), "Validation steps and values must be aligned"
            axes[plot_idx].plot(val_step_high_entropy_steps, val_step_high_entropy_vals, 
                              label='Val High Entropy Loss (Step)', linewidth=2, linestyle='--', 
                              marker='o', markersize=6, color='red', zorder=5)
        axes[plot_idx].set_xlabel('Training Step')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].set_title('High Entropy Loss')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Learning curve saved to: {self.plot_file}")

def train_atlas(
    model,
    epoch,
    dataloader,
    loss_fns,
    optimizer,
    scheduler,
    device,
    collator,
    scaler,
    ablation_type="full",
    args=None,
    logger=None,
    global_step=0,
    field_categories=None,
    validation_dir=None,
    stage_config=None,
):
    model.train()
    collator.set_epoch(epoch)

    current_strategy = collator.get_current_strategy()
    stage_info = collator.get_stage_info()

    print(f"\n📂 Training Batch — Epoch {epoch + 1} | Stage {stage_info['current_stage']}/{stage_info['total_stages']} | Strategy: {current_strategy}")

    total_loss = 0.0
    mlm_loss_total = 0.0
    dist_loss_total = 0.0
    low_entropy_loss_total = 0.0
    high_entropy_loss_total = 0.0
    num_batches = 0
    
    # Timing tracking
    epoch_start_time = time.time()
    step_times = []

    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        step_start_time = time.time()
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
            # Forward
            _, mlm_logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                segment_ids=segment_ids,
                header_strings=header_strings,
            )

            contextualized_embeddings, _ = model(
                input_ids=unmasked_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                segment_ids=segment_ids,
                header_strings=header_strings,
            )

            extracted_embeddings = {}
            
            # Extract contextualized header and value embeddings
            header_embeds, val_embeds = extract_header_value_embeddings(
                contextualized_embeddings, batch["header_positions"], batch["value_positions"]
            )

            # Extract header anchor embeddings
            header_encoder = model.bert.embeddings.header_encoder
            header_anchor_embeds, _ = header_encoder(header_strings)

            if torch.isnan(header_anchor_embeds).any() or torch.isnan(header_embeds).any() or torch.isnan(val_embeds).any():
                print("🚨 NaN detected in embeddings BEFORE loss calculation!")
                print(f"NaN in anchor embeds: {torch.isnan(header_anchor_embeds).any()}")
                print(f"NaN in header embeds: {torch.isnan(header_embeds).any()}")
                print(f"NaN in value embeds: {torch.isnan(val_embeds).any()}")

            extracted_embeddings.update({
                'header_anchor_embeds': header_anchor_embeds,
                'header_embeds': header_embeds,
                'val_embeds': val_embeds
            })

            # Losses
            mlm_loss = loss_fns["mlm_loss"](mlm_logits, labels)

            entropy_contrast_loss = None
            low_entropy_loss = None
            high_entropy_loss = None

            if ablation_type != "woESA":
                table_ids = batch.get("table_ids")
                header_strings = batch.get("header_strings")

                entropy_loss_result = loss_fns["entropy_loss"](
                    E_univ=header_anchor_embeds,     # (B, H, D)
                    H_ctx=header_embeds,             # (B, H, D)
                    V_ctx=val_embeds,                # (B, H, D)
                    header_strings=header_strings,   # List[List[str]]
                    table_ids=table_ids,             # List[int]
                    field_categories=field_categories,  # from main()
                    return_dict=True  # Return dict with individual losses
                )
                
                # Handle both dict and scalar returns for backward compatibility
                if isinstance(entropy_loss_result, dict):
                    entropy_contrast_loss = entropy_loss_result["total"]
                    low_entropy_loss = entropy_loss_result.get("low_entropy", torch.tensor(0.0, device=header_anchor_embeds.device))
                    high_entropy_loss = entropy_loss_result.get("high_entropy", torch.tensor(0.0, device=header_anchor_embeds.device))
                else:
                    entropy_contrast_loss = entropy_loss_result
                    # If scalar return, set individual losses to 0.0 for tracking
                    low_entropy_loss = torch.tensor(0.0, device=header_anchor_embeds.device)
                    high_entropy_loss = torch.tensor(0.0, device=header_anchor_embeds.device)

            all_losses = [mlm_loss, entropy_contrast_loss]
            loss_values = [l.item() if l is not None else 0.0 for l in all_losses]
            weights = get_loss_weights(current_strategy, ablation_type, args.alignment_weight)

            print(f"loss_values: {loss_values}")

            total_batch_loss = sum(w * l for w, l in zip(weights, all_losses) if l is not None)
            total_batch_loss = total_batch_loss / args.gradient_accumulation_steps

        scaler.scale(total_batch_loss).backward()

        # Update parameters every gradient_accumulation_steps
        if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            optimizer.zero_grad()

        total_loss += total_batch_loss.item() * args.gradient_accumulation_steps  # Unscale for logging
        mlm_loss_total += mlm_loss.item() if mlm_loss is not None else 0.0
        dist_loss_total += entropy_contrast_loss.item() if entropy_contrast_loss else 0.0
        # Always accumulate entropy losses (they may be 0.0 if no valid fields)
        if ablation_type != "woESA":
            if low_entropy_loss is not None:
                low_entropy_loss_total += low_entropy_loss.item()
            else:
                low_entropy_loss_total += 0.0
            if high_entropy_loss is not None:
                high_entropy_loss_total += high_entropy_loss.item()
            else:
                high_entropy_loss_total += 0.0

        num_batches += 1
        
        # Only increment global_step when we actually update parameters
        if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
            global_step += 1

            if logger and global_step % args.log_steps == 0:
                step_time = time.time() - step_start_time
                samples_per_sec = args.batch_size / step_time if step_time > 0 else 0
                logger.log_step(global_step, {
                    "total_loss": float(total_batch_loss.item() * args.gradient_accumulation_steps),
                    "mlm_loss": float(mlm_loss.item()) if mlm_loss is not None else None,
                    "entropy_contrast_loss": float(entropy_contrast_loss.item()) if entropy_contrast_loss else None,
                    "low_entropy_loss": float(low_entropy_loss.item()) if low_entropy_loss is not None else None,
                    "high_entropy_loss": float(high_entropy_loss.item()) if high_entropy_loss is not None else None,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "loss_weights": [float(w) for w in weights],
                    "samples_per_sec": float(samples_per_sec),
                    "step_time": float(step_time)
                })
            
            # Run validation every 1000 steps
            if validation_dir and logger and global_step % 1000 == 0:
                # Get tokenizer from dataloader's dataset
                tokenizer = dataloader.dataset.tokenizer if hasattr(dataloader.dataset, 'tokenizer') else collator.tokenizer
                # Create a validation collator (reuse tokenizer from dataset or collator)
                val_collator = NaviCollator(
                    tokenizer=tokenizer,
                    stage_config=stage_config,
                    word_level_mask_fraction=0.25,
                    token_length_threshold=8,
                    ablation_mode=ablation_type,
                    field_categories=None,  # Validation doesn't need field categories
                    hv_weight=args.hv_weight,
                    value_ratio=args.value_ratio
                )
                val_metrics = run_validation(
                    model, validation_dir, device, loss_fns, val_collator, 
                    ablation_type, args, step_sample_ratio=0.2  # Randomly sample 20% for validation
                )
                if val_metrics:
                    logger.log_validation_step(global_step, val_metrics)
        
        step_time = time.time() - step_start_time
        step_times.append(step_time)

    # Calculate epoch metrics
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    avg_total_loss = total_loss / num_batches
    avg_mlm_loss = mlm_loss_total / num_batches
    avg_entropy_contrast_loss = dist_loss_total / num_batches
    # Calculate averages - show 0.0 if calculated, None only if never calculated
    avg_low_entropy_loss = low_entropy_loss_total / num_batches if ablation_type != "woESA" else None
    avg_high_entropy_loss = high_entropy_loss_total / num_batches if ablation_type != "woESA" else None
    
    # Calculate throughput
    total_samples = num_batches * args.batch_size
    samples_per_sec = total_samples / epoch_duration if epoch_duration > 0 else 0
    avg_step_time = np.mean(step_times) if step_times else 0
    
    # Log epoch metrics (ensure JSON serializable)
    epoch_metrics = {
        "total_loss": float(avg_total_loss),
        "mlm_loss": float(avg_mlm_loss),
        "entropy_contrast_loss": float(avg_entropy_contrast_loss) if ablation_type != "woESA" and avg_entropy_contrast_loss is not None else None,
        "low_entropy_loss": float(avg_low_entropy_loss) if ablation_type != "woESA" and avg_low_entropy_loss is not None else None,
        "high_entropy_loss": float(avg_high_entropy_loss) if ablation_type != "woESA" and avg_high_entropy_loss is not None else None,
        "samples_processed": int(total_samples),
        "num_batches": int(num_batches),
        "epoch_duration_seconds": float(epoch_duration),
        "samples_per_sec": float(samples_per_sec),
        "avg_step_time": float(avg_step_time)
    }
    
    if logger:
        logger.log_epoch(epoch, epoch_metrics, stage_info, current_strategy)
    
    print(f"\n[Epoch {epoch+1}]")
    print(f"📉 Total Loss: {avg_total_loss:.4f}")
    print(f"🔠 MLM Loss: {avg_mlm_loss:.4f}")
    if ablation_type != "woESA":
        print(f"🧬 Entropy Contrastive Loss: {avg_entropy_contrast_loss:.4f}")
        if avg_low_entropy_loss is not None:
            print(f"   └─ Low Entropy Loss: {avg_low_entropy_loss:.4f}")
        if avg_high_entropy_loss is not None:
            print(f"   └─ High Entropy Loss: {avg_high_entropy_loss:.4f}")
    print(f"⏱️  Epoch Duration: {epoch_duration:.2f}s | Throughput: {samples_per_sec:.2f} samples/sec")
    
    return global_step

def run_validation(model, validation_dir, device, loss_fns, collator, ablation_type, args, step_sample_ratio=0.2):
    """
    Run validation on validation dataset.
    Returns validation metrics including losses and optionally other metrics.
    
    Args:
        step_sample_ratio: Ratio to sample validation data (default: 0.2 for 20%)
    """
    if validation_dir is None or not os.path.exists(validation_dir):
        return None
    
    print(f"\n🔍 Running validation (randomly sampling {step_sample_ratio*100:.0f}% of data)...")
    model.eval()
    
    # Load validation files
    val_files = sorted([
        os.path.join(validation_dir, f)
        for f in os.listdir(validation_dir)
        if f.endswith(".json") or f.endswith(".jsonl")
    ])
    
    if not val_files:
        print("⚠️ No validation files found")
        return None
    
    # Load validation data - load all data first, then randomly sample
    val_data = []
    for val_file in val_files:
        try:
            data = load_jsonl(val_file)
            val_data.extend([(0, row) for row in data])
        except Exception as e:
            print(f"⚠️ Error loading {val_file}: {e}")
            continue
    
    # Randomly sample the specified ratio of the total data
    if val_data:
        total_samples = len(val_data)
        num_samples = max(1, int(total_samples * step_sample_ratio))
        val_data = random.sample(val_data, num_samples)
        print(f"📊 Randomly sampled {len(val_data)}/{total_samples} validation samples ({step_sample_ratio*100:.0f}% of total)")
    
    if not val_data:
        print("⚠️ No validation data loaded")
        return None
    
    # Create validation dataset
    val_dataset = NaviDataset(
        val_data,
        tokenizer_name=config.BERT_NAME,
        max_length=config.MAX_SEQ_LENGTH,
        ablation_mode=ablation_type,
        compute_field_entropy=False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4,
        pin_memory=True
    )
    
    total_loss = 0.0
    mlm_loss_total = 0.0
    entropy_loss_total = 0.0
    low_entropy_loss_total = 0.0
    high_entropy_loss_total = 0.0
    num_batches = 0
    
    val_start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation", leave=False):
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
                # Forward
                _, mlm_logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    segment_ids=segment_ids,
                    header_strings=header_strings,
                )
                
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
                
                # Compute losses
                mlm_loss = loss_fns["mlm_loss"](mlm_logits, labels)
                
                entropy_contrast_loss = None
                low_entropy_loss = None
                high_entropy_loss = None
                if ablation_type != "woESA":
                    table_ids = batch.get("table_ids")
                    # For validation, we don't compute field entropy, so field_categories is None
                    # The loss function should handle this gracefully
                    entropy_loss_result = loss_fns["entropy_loss"](
                        E_univ=header_anchor_embeds,
                        H_ctx=header_embeds,
                        V_ctx=val_embeds,
                        header_strings=header_strings,
                        table_ids=table_ids,
                        field_categories=None,  # Validation doesn't use field categories
                        return_dict=True
                    )
                    
                    # Handle both dict and scalar returns for backward compatibility
                    if isinstance(entropy_loss_result, dict):
                        entropy_contrast_loss = entropy_loss_result["total"]
                        low_entropy_loss = entropy_loss_result.get("low_entropy", torch.tensor(0.0, device=header_anchor_embeds.device))
                        high_entropy_loss = entropy_loss_result.get("high_entropy", torch.tensor(0.0, device=header_anchor_embeds.device))
                    else:
                        entropy_contrast_loss = entropy_loss_result
                        # If scalar return, set individual losses to 0.0 for tracking
                        low_entropy_loss = torch.tensor(0.0, device=header_anchor_embeds.device)
                        high_entropy_loss = torch.tensor(0.0, device=header_anchor_embeds.device)
                
                all_losses = [mlm_loss, entropy_contrast_loss]
                weights = get_loss_weights(collator.get_current_strategy(), ablation_type, args.alignment_weight)
                total_batch_loss = sum(w * l for w, l in zip(weights, all_losses) if l is not None)
                
                total_loss += total_batch_loss.item()
                mlm_loss_total += mlm_loss.item() if mlm_loss is not None else 0.0
                entropy_loss_total += entropy_contrast_loss.item() if entropy_contrast_loss else 0.0
                # Always accumulate entropy losses (they may be 0.0 if no valid fields)
                if ablation_type != "woESA":
                    if low_entropy_loss is not None:
                        low_entropy_loss_total += low_entropy_loss.item()
                    else:
                        low_entropy_loss_total += 0.0
                    if high_entropy_loss is not None:
                        high_entropy_loss_total += high_entropy_loss.item()
                    else:
                        high_entropy_loss_total += 0.0
                num_batches += 1
    
    model.train()
    
    val_end_time = time.time()
    val_duration = val_end_time - val_start_time
    total_val_samples = num_batches * args.batch_size
    val_samples_per_sec = total_val_samples / val_duration if val_duration > 0 else 0
    
    val_metrics = {
        'total_loss': float(total_loss / num_batches if num_batches > 0 else 0.0),
        'mlm_loss': float(mlm_loss_total / num_batches if num_batches > 0 else 0.0),
        'entropy_contrast_loss': float(entropy_loss_total / num_batches if num_batches > 0 else 0.0),
        'low_entropy_loss': float(low_entropy_loss_total / num_batches) if num_batches > 0 else None,
        'high_entropy_loss': float(high_entropy_loss_total / num_batches) if num_batches > 0 else None,
        'num_samples': int(total_val_samples),
        'num_batches': int(num_batches),
        'duration_seconds': float(val_duration),
        'samples_per_sec': float(val_samples_per_sec)
    }
    
    print(f"📊 Validation - Total Loss: {val_metrics['total_loss']:.4f}, MLM Loss: {val_metrics['mlm_loss']:.4f}")
    if ablation_type != "woESA":
        print(f"   Entropy Contrastive Loss: {val_metrics['entropy_contrast_loss']:.4f}")
        if val_metrics['low_entropy_loss'] is not None:
            print(f"   └─ Low Entropy Loss: {val_metrics['low_entropy_loss']:.4f}")
        if val_metrics['high_entropy_loss'] is not None:
            print(f"   └─ High Entropy Loss: {val_metrics['high_entropy_loss']:.4f}")
    print(f"⏱️  Validation Duration: {val_duration:.2f}s | Throughput: {val_samples_per_sec:.2f} samples/sec")
    
    return val_metrics

def main(args):
    # Set seeds for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Random seed set to: {seed}")
    
    print("🔄 Loading all JSONL files...")
    jsonl_files = sorted([
        os.path.join(args.data_path, f)
        for f in os.listdir(args.data_path)
        if f.endswith(".json")
    ])

    # Sort files by number of lines (rows) as per flowchart
    print("📊 Sorting files by row count...")
    
    # Use multiprocessing for faster file size counting
    with Pool(processes=min(8, os.cpu_count())) as pool:
        file_sizes = pool.map(count_lines, jsonl_files)
    
    # Filter out failed files and sort
    file_sizes = [(f, s) for f, s in file_sizes if s > 0]
    file_sizes.sort(key=lambda x: x[1], reverse=True)
    jsonl_files = [f[0] for f in file_sizes]
    
    # Calculate dataset statistics
    total_training_rows = sum(s for _, s in file_sizes)
    file_row_counts = [s for _, s in file_sizes]
    dataset_stats = {
        'num_files': len(jsonl_files),
        'total_training_rows': total_training_rows,
        'min_rows_per_file': min(file_row_counts) if file_row_counts else 0,
        'max_rows_per_file': max(file_row_counts) if file_row_counts else 0,
        'mean_rows_per_file': np.mean(file_row_counts) if file_row_counts else 0,
        'median_rows_per_file': np.median(file_row_counts) if file_row_counts else 0
    }
    
    print(f"📈 Found {len(jsonl_files)} files, sorted by size")
    print(f"📊 Dataset stats: {total_training_rows:,} total rows | "
          f"Mean: {dataset_stats['mean_rows_per_file']:.0f} | "
          f"Min: {dataset_stats['min_rows_per_file']} | "
          f"Max: {dataset_stats['max_rows_per_file']}")
    
    # Group into 25 groups of 4 tables each (as per flowchart)
    tables_per_group = 4
    num_groups = len(jsonl_files) // tables_per_group
    print(f"📦 Creating {num_groups} groups with {tables_per_group} tables each")

    masking_strategy = args.masking_strategy
    ablation_type = args.ablation_type

    # For woMSM ablation, override masking strategy to B
    if ablation_type == 'woMSM':
        masking_strategy = 'B'
        print(f"woMSM ablation: Forcing masking strategy to 'B' (BERT-style)")

    stage_config = get_stage_config(masking_strategy, ablation_type)
    total_epochs = args.num_epochs
    num_files = len(jsonl_files)

    print("🚀 Initializing model...")
    device = torch.device(config.DEVICE)
    model = NaviForMaskedLM(
        bert_name=config.BERT_NAME,
        ablation_mode=ablation_type,
        header_encoder_mode=args.header_encoder_mode
    ).to(device)
    
    # Log header encoder mode and parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params_count
    print(f"📊 Header Encoder Mode: {args.header_encoder_mode}")
    print(f"📊 Total Parameters: {total_params:,} | Trainable: {trainable_params_count:,} | Frozen: {frozen_params:,}")

    loss_fns = {
        "mlm_loss": MLMLoss(),
        "entropy_loss": EntropyAwareContrastiveLoss(model, low_entropy_tau=args.low_entropy_tau, high_entropy_tau=args.high_entropy_tau)
    }

    # Filter out frozen parameters from optimizer
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params_list, lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda")

    # ✅ Estimate total training steps based on dataset size
    print("📏 Estimating total training steps...")
    
    # Group files into batches for estimation
    file_groups = [jsonl_files[i:i+args.tables_per_batch] for i in range(0, len(jsonl_files), args.tables_per_batch)]
    steps_per_epoch = 0
    
    for file_group in tqdm(file_groups, desc="Estimating batch lengths", leave=False):
        try:
            combined_dataset = create_combined_dataset(
                file_group, 
                config.BERT_NAME, 
                config.MAX_SEQ_LENGTH, 
                ablation_type,
                compute_field_entropy=False,  # Skip entropy computation for batch length estimation
                entropy_threshold_method=args.entropy_threshold_method,
                low_threshold_percentile=args.low_threshold_percentile,
                high_threshold_percentile=args.high_threshold_percentile
            )
            steps_in_group = len(combined_dataset) // args.batch_size + 1
            # Account for gradient accumulation
            steps_per_epoch += steps_in_group // args.gradient_accumulation_steps + 1
        except Exception as e:
            print(f"⚠️ Skipping group due to error: {e}")
            continue
    
    total_steps = total_epochs * steps_per_epoch
    print(f"📈 Total estimated training steps: {total_steps}")

    from torch.optim.lr_scheduler import LambdaLR
    def get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
        return LambdaLR(optimizer, lr_lambda)

    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)
    logger = LossLogger(
        args.output_dir, 
        ablation_type, 
        masking_strategy,
        seed=args.seed,
        data_type=args.data_type,
        low_entropy_tau=args.low_entropy_tau,
        high_entropy_tau=args.high_entropy_tau,
        entropy_threshold_method=args.entropy_threshold_method
    )
    
    # Log header encoder mode and parameter counts (ensure JSON serializable)
    logger.metrics['training_info']['header_encoder_mode'] = args.header_encoder_mode
    logger.metrics['training_info']['total_parameters'] = int(total_params)
    logger.metrics['training_info']['trainable_parameters'] = int(trainable_params_count)
    logger.metrics['training_info']['frozen_parameters'] = int(frozen_params)
    
    # Log dataset statistics
    logger.metrics['training_info']['dataset_stats'] = dataset_stats
    
    # Get validation dataset stats if available
    if args.validation_dir and os.path.exists(args.validation_dir):
        val_files = sorted([
            os.path.join(args.validation_dir, f)
            for f in os.listdir(args.validation_dir)
            if f.endswith(".json") or f.endswith(".jsonl")
        ])
        if val_files:
            with Pool(processes=min(4, os.cpu_count())) as pool:
                val_file_sizes = pool.map(count_lines, val_files)
            val_file_sizes = [(f, s) for f, s in val_file_sizes if s > 0]
            total_val_rows = sum(s for _, s in val_file_sizes)
            val_row_counts = [s for _, s in val_file_sizes]
            dataset_stats['num_validation_files'] = len(val_files)
            dataset_stats['total_validation_rows'] = total_val_rows
            dataset_stats['min_val_rows_per_file'] = min(val_row_counts) if val_row_counts else 0
            dataset_stats['max_val_rows_per_file'] = max(val_row_counts) if val_row_counts else 0
            dataset_stats['mean_val_rows_per_file'] = np.mean(val_row_counts) if val_row_counts else 0
            print(f"📊 Validation stats: {total_val_rows:,} total rows from {len(val_files)} files")
    
    logger.save_metrics()

    print("🎯 Starting training...")
    print(f"📊 Masking strategy: {masking_strategy}")
    print(f"🔧 Ablation: {ablation_type}")
    print(f"📈 Stage config: {stage_config}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📚 Tables per batch: {args.tables_per_batch}")
    print(f"🔄 Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"📝 Logging to: {logger.get_metrics_file()}")

    global_step = 0
    training_start_time = time.time()

    # Global Epoch Loop (as per flowchart)
    for epoch in range(total_epochs):
        print(f"\n🗓️ Global Epoch {epoch + 1}/{total_epochs}")
        
        # Shuffle group order at the beginning of each epoch (as per flowchart)
        group_indices = list(range(num_groups))
        random.shuffle(group_indices)
        
        # Process groups sequentially (as per flowchart)
        cnt = 0
        for group_idx in group_indices:
            start_idx = group_idx * tables_per_group
            end_idx = start_idx + tables_per_group
            group_files = jsonl_files[start_idx:end_idx]
            
            print(f"\n Processing Group {cnt + 1}/{num_groups}: Tables {start_idx + 1}-{end_idx}")
            print(f" Files: {[os.path.basename(f) for f in group_files]}")

            # Create combined dataset from multiple files
            combined_dataset = create_combined_dataset(
                group_files,
                tokenizer_name=config.BERT_NAME,
                max_length=config.MAX_SEQ_LENGTH,
                ablation_mode=ablation_type,
                compute_field_entropy=True,
                entropy_threshold_method=args.entropy_threshold_method,
                low_threshold_percentile=args.low_threshold_percentile,
                high_threshold_percentile=args.high_threshold_percentile
            )

            # Get field categories for the collator
            field_categories = combined_dataset.get_field_categories()

            collator = NaviCollator(
                tokenizer=combined_dataset.tokenizer,
                stage_config=stage_config,
                word_level_mask_fraction=0.25,
                token_length_threshold=8,
                ablation_mode=ablation_type,
                field_categories=field_categories,
                hv_weight=args.hv_weight,
                value_ratio=args.value_ratio
            )

            sampler = StratifiedBatchSampler(combined_dataset, args.batch_size)
            dataloader = DataLoader(
                combined_dataset,
                batch_size=args.batch_size,
                num_workers=12,
                sampler=sampler,
                collate_fn=collator,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )

            global_step = train_atlas(
                model=model,
                epoch=epoch,
                dataloader=dataloader,
                loss_fns=loss_fns,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                collator=collator,
                scaler=scaler,
                ablation_type=ablation_type,
                args=args,
                logger=logger,
                global_step=global_step,
                field_categories=field_categories,
                validation_dir=args.validation_dir if hasattr(args, 'validation_dir') else None,
                stage_config=stage_config,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"✅ Group {group_idx + 1} Complete")
            cnt += 1
        
        # Note: Validation is only performed at step intervals (every 1000 steps) during training
        # No epoch-based validation is performed
        
        # Save checkpoint after each epoch
        # Update model save path to include seed and data type
        model_name_parts = [ablation_type, masking_strategy]
        if args.seed is not None:
            model_name_parts.append(f"seed{args.seed}")
        if args.data_type:
            model_name_parts.append(args.data_type)
        if args.low_entropy_tau is not None and args.high_entropy_tau is not None:
            model_name_parts.append(f"tau{args.high_entropy_tau}_{args.low_entropy_tau}")
        if args.entropy_threshold_method:
            model_name_parts.append(args.entropy_threshold_method)
        
        model_name = "_".join(model_name_parts)
        epoch_checkpoint_path = os.path.join(args.output_dir, f"{model_name}_epoch_{epoch + 1}")
        model.save_pretrained(epoch_checkpoint_path)
        print(f"💾 Epoch {epoch + 1} checkpoint saved: {epoch_checkpoint_path}")
    
    # Calculate total training time and compute metrics
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    total_training_hours = total_training_time / 3600
    
    # Calculate effective batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        effective_batch_size *= num_gpus
    else:
        num_gpus = 0
    
    # Estimate total samples processed
    total_samples_processed = total_training_rows * total_epochs
    
    # Log compute metrics (ensure JSON serializable)
    compute_metrics = {
        'total_training_time_seconds': float(total_training_time),
        'total_training_time_hours': float(total_training_hours),
        'total_training_steps': int(global_step),
        'total_epochs': int(total_epochs),
        'steps_per_epoch': int(steps_per_epoch),
        'effective_batch_size': int(effective_batch_size),
        'batch_size': int(args.batch_size),
        'gradient_accumulation_steps': int(args.gradient_accumulation_steps),
        'num_gpus': int(num_gpus),
        'total_samples_processed': int(total_samples_processed),
        'end_time': datetime.now().isoformat()
    }
    
    logger.log_compute_metrics(compute_metrics)
    
    print(f"\n⏱️  Total Training Time: {total_training_hours:.2f} hours ({total_training_time:.2f} seconds)")
    print(f"📊 Total Steps: {global_step} | Effective Batch Size: {effective_batch_size}")
    
    # Plot learning curve at the end
    logger.plot_learning_curve(smoothing=0.1)
    
    # Final model save (same as last checkpoint, but kept for consistency)
    final_save_path = os.path.join(args.output_dir, f"{model_name}_epoch_{total_epochs}")
    if not os.path.exists(final_save_path):
        model.save_pretrained(final_save_path)
        print(f"💾 Final model saved: {final_save_path}")
    else:
        print(f"✓ Final model already saved as checkpoint: {final_save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)