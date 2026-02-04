import argparse
from tqdm import tqdm
import os
from multiprocessing import Pool
import torch
import torch.nn as nn
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
from transformers import BertConfig, get_scheduler
from transformers.utils.logging import set_verbosity_info, enable_propagation
from dataset import JSONDataset, JSONDataCollator
from model import HAETAE
from transformers import BertTokenizer

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

def load_jsonl(file_path):
    """Load data from a JSONL file"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def clean_table_data(json_data, tokenizer_name="bert-base-uncased", 
                         max_tokens=500, max_indexed_fields=3, max_tokens_per_field=20):
    """
    Clean table data.
    """
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    processed_data = []
    
    print("🔄 Preprocessing table data...")
    
    for table_id, table_dict in json_data:
        # Step 1: Handle indexed fields
        processed_table = handle_indexed_fields(table_dict, max_indexed_fields)
        
        # Step 2: Truncate long text fields
        processed_table = truncate_long_fields(processed_table, max_tokens_per_field, tokenizer)
        
        # Step 3: Select fields that fit in token limit
        processed_table = select_fields_for_tokenization(processed_table, tokenizer, max_tokens)
        
        processed_data.append((table_id, processed_table))
        
        if len(processed_data) % 1000 == 0:
            print(f"   Processed {len(processed_data)} instances...")
    
    print(f"✅ Preprocessing complete. Processed {len(processed_data)} instances.")
    return processed_data

def handle_indexed_fields(table_dict, max_indexed_fields=3):
    """Handle indexed fields by sampling representative ones."""
    processed_table = {}
    field_groups = {}
    
    # Group indexed fields
    for field_name, field_value in table_dict.items():
        if '.' in field_name and any(char.isdigit() for char in field_name):
            # Extract field type (first word) and check if there's an index anywhere
            field_type = field_name.split('.')[0]  # First word is always the type
            
            # Check if this field has an index (anywhere after the first word)
            has_index = False
            for part in field_name.split('.')[1:]:  # Check all parts after the first word
                if part.isdigit():
                    has_index = True
                    break
            
            if has_index:
                if field_type not in field_groups:
                    field_groups[field_type] = []
                field_groups[field_type].append((field_name, field_value))
            else:
                # Field has dots but no index (e.g., "aggregaterating.bestrating")
                processed_table[field_name] = field_value
        else:
            # Field has no dots (e.g., "name", "description")
            processed_table[field_name] = field_value
    
    # Sample indexed fields
    for field_type, fields in field_groups.items():
        if len(fields) > max_indexed_fields:
            # Sample first few fields (0, 1, 2, ...)
            sampled = fields[:max_indexed_fields]
            for field_name, field_value in sampled:
                processed_table[field_name] = field_value
        else:
            for field_name, field_value in fields:
                processed_table[field_name] = field_value
    
    return processed_table

def truncate_long_fields(table_dict, max_tokens_per_field=20, tokenizer=None):
    """Truncate fields that exceed token limit based on actual tokenization."""
    if tokenizer is None:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    processed_table = {}
    
    for field_name, field_value in table_dict.items():
        # Estimate token count for this field
        estimated_tokens = estimate_token_count(field_name, field_value, tokenizer)
        
        if estimated_tokens > max_tokens_per_field and isinstance(field_value, str):
            # Field exceeds token limit, need to truncate
            # Try to truncate the value while keeping the field name
            truncated_value = truncate_value_to_token_limit(
                field_name, field_value, tokenizer, max_tokens_per_field
            )
            processed_table[field_name] = truncated_value
        else:
            # Field is within token limit, keep as-is
            processed_table[field_name] = field_value
    
    return processed_table

def truncate_value_to_token_limit(field_name, field_value, tokenizer, max_tokens_per_field):
    """Truncate field value to fit within token limit."""
    # Reserve tokens for field name and separator
    # Format: "field_name : field_value"
    name_tokens = tokenizer.tokenize(field_name)
    separator_tokens = tokenizer.tokenize(" : ")
    reserved_tokens = len(name_tokens) + len(separator_tokens)
    
    # Calculate how many tokens we can use for the value
    available_tokens = max_tokens_per_field - reserved_tokens
    
    if available_tokens <= 0:
        # Field name itself is too long, return minimal value
        return "..."
    
    # Tokenize the value
    value_tokens = tokenizer.tokenize(str(field_value))
    
    if len(value_tokens) <= available_tokens:
        # Value fits within limit
        return field_value
    else:
        # Truncate value tokens
        truncated_tokens = value_tokens[:available_tokens]
        # Convert back to text (approximate)
        truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
        # Add ellipsis to indicate truncation
        return truncated_text + "..."

def estimate_token_count(field_name, field_value, tokenizer):
    """Estimate token count for a field."""
    text = f"{field_name} : {field_value}"
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def select_fields_for_tokenization(table_dict, tokenizer, max_tokens=400):
    """Select fields that will fit within token limit."""
    selected_fields = {}
    current_tokens = 0
    
    # Sort fields by priority (shorter fields first)
    field_items = list(table_dict.items())
    field_items.sort(key=lambda x: len(str(x[1])))
    
    for field_name, field_value in field_items:
        estimated_tokens = estimate_token_count(field_name, field_value, tokenizer)
        
        if current_tokens + estimated_tokens <= max_tokens:
            selected_fields[field_name] = field_value
            current_tokens += estimated_tokens
        else:
            break  # Stop when we would exceed limit
    
    return selected_fields

def create_combined_dataset(jsonl_files, tokenizer, max_length, cache_dir=".cache_tokenized"):
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

    all_data = clean_table_data(
        all_data
    )
    
    print(f"🔄 Total combined data: {len(all_data)} rows")
    
    # Create single JSONDataset with all merged data as tuples
    # We need to extract just the json_objects for JSONDataset
    json_objects = [item[1] for item in all_data]
    dataset = JSONDataset(
        path=None,  # We'll pass data directly
        tokenizer=tokenizer,
        max_length=max_length,
        path_is="json"
    )
    # Manually set the data
    dataset.data = json_objects
    
    # Add table_ids as an attribute for potential future use
    dataset.table_ids = [item[0] for item in all_data]
    
    return dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train HAETAE with Masked Language Modeling")
    parser.add_argument("--data_path", type=str, required=True, help="Path to directory containing JSONL files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--pretrained_model", type=str, default="bert-base-uncased", help="Pretrained model name or path")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of gradient accumulation steps")
    parser.add_argument("--tables_per_batch", type=int, default=4, help="Number of tables to process simultaneously")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Number of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--log_steps", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Probability for masking tokens in MLM")
    parser.add_argument("--key_mask_probability", type=float, default=0.24, help="Probability for masking keys")
    parser.add_argument("--nonkey_mask_probability", type=float, default=0.17, help="Probability for masking nonkeys")
    parser.add_argument("--validation_dir", type=str, default=None, help="Path to validation data directory (optional)")
    parser.add_argument("--validation_sample_ratio", type=float, default=0.2, help="Ratio of validation data to use (1.0 = full dataset, default: 0.2)")
    return parser.parse_args()

class LossLogger:
    """Logger for tracking training metrics in a structured format."""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.metrics = {
            'training_info': {
                'start_time': datetime.now().isoformat(),
                'epochs': [],
                'validation': [],
                'compute_metrics': {},
                'dataset_stats': {}
            },
            'steps': [] 
        }
        self.metrics_file = os.path.join(output_dir, "training_metrics_haetae.json")
        self.plot_file = os.path.join(output_dir, "learning_curve_haetae.png")

    def log_epoch(self, epoch, epoch_metrics):
        epoch_data = {
            'epoch': epoch + 1,
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
    
    def log_compute_metrics(self, compute_metrics):
        """Log compute metrics like timing, throughput, etc."""
        self.metrics['training_info']['compute_metrics'].update(compute_metrics)
        self.save_metrics()
    
    def plot_learning_curve(self, smoothing=0.1):
        """Plot learning curve with optional smoothing, including validation losses."""
        if not self.metrics['steps']:
            print("⚠️ No step data available for plotting")
            return
        
        # Extract training data
        steps = [s['step'] for s in self.metrics['steps']]
        mlm_losses = [s['metrics'].get('mlm_loss', 0) for s in self.metrics['steps'] if s['metrics'].get('mlm_loss') is not None]
        alignment_losses = [s['metrics'].get('alignment_loss', 0) for s in self.metrics['steps'] if s['metrics'].get('alignment_loss') is not None]
        total_losses = []
        for s in self.metrics['steps']:
            mlm = s['metrics'].get('mlm_loss', 0)
            align = s['metrics'].get('alignment_loss', 0)
            total_losses.append(mlm + align)
        
        # Extract step-based validation data (every 1000 steps)
        validation_steps_data = self.metrics['training_info'].get('validation_steps', [])
        
        # Extract validation data
        val_step_data = {
            'steps': [],
            'total_loss': [],
            'mlm_loss': []
        }
        
        for v in validation_steps_data:
            step = v['step']
            metrics = v['metrics']
            val_step_data['steps'].append(step)
            val_step_data['total_loss'].append(metrics.get('total_loss', 0))
            if metrics.get('mlm_loss') is not None:
                val_step_data['mlm_loss'].append((step, metrics['mlm_loss']))
        
        # Unpack paired data for mlm loss
        val_step_mlm_steps = [x[0] for x in val_step_data['mlm_loss']]
        val_step_mlm_vals = [x[1] for x in val_step_data['mlm_loss']]
        
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
            if alignment_losses:
                alignment_losses = smooth(alignment_losses, 1 - smoothing)
        
        # Create figure with subplots
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
        
        # Plot 3: Alignment Loss
        plot_idx = 2
        if alignment_losses:
            align_steps = [s['step'] for s in self.metrics['steps'] if s['metrics'].get('alignment_loss') is not None]
            axes[plot_idx].plot(align_steps, alignment_losses, label='Train Alignment Loss', linewidth=2, color='green')
        axes[plot_idx].set_xlabel('Training Step')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].set_title('Alignment Loss')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        
        # Plot 4: Learning Rate (if available)
        plot_idx = 3
        lr_steps = [s['step'] for s in self.metrics['steps'] if s['metrics'].get('learning_rate') is not None]
        lr_values = [s['metrics'].get('learning_rate') for s in self.metrics['steps'] if s['metrics'].get('learning_rate') is not None]
        if lr_steps:
            axes[plot_idx].plot(lr_steps, lr_values, label='Learning Rate', linewidth=2, color='purple')
        axes[plot_idx].set_xlabel('Training Step')
        axes[plot_idx].set_ylabel('Learning Rate')
        axes[plot_idx].set_title('Learning Rate Schedule')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Learning curve saved to: {self.plot_file}")

    def save_metrics(self):
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def get_metrics_file(self):
        return self.metrics_file

def train_haetae(
    model,
    epoch,
    dataloader,
    optimizer,
    scheduler,
    device,
    data_collator,
    args=None,
    logger=None,
    global_step=0,
    validation_dir=None,
):
    model.train()

    print(f"\n Training Batch — Epoch {epoch + 1}")

    total_mlm_loss = 0.0
    total_alignment_loss = 0.0
    num_batches = 0
    
    # Timing tracking
    epoch_start_time = time.time()
    step_times = []

    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        step_start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        key_positions = batch["key_positions"]

        # Only compute centroid loss at the end of an epoch or every 500 steps
        compute_alignment_loss = (step % 500 == 0 or step == len(dataloader) - 1)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            key_positions=key_positions,
            compute_alignment_loss=compute_alignment_loss
        )
        
        mlm_loss = outputs["mlm_loss"]

        if compute_alignment_loss:
            centroid_alignment_loss = outputs["centroid_alignment_loss"]
            mlm_loss = mlm_loss / args.gradient_accumulation_steps
            centroid_alignment_loss = centroid_alignment_loss / args.gradient_accumulation_steps
            mlm_loss.backward(retain_graph=compute_alignment_loss)
            centroid_alignment_loss.backward()
            loss = mlm_loss + centroid_alignment_loss
            total_alignment_loss += centroid_alignment_loss.item() * args.gradient_accumulation_steps
        else:
            loss = mlm_loss / args.gradient_accumulation_steps
            loss.backward()

        total_mlm_loss += mlm_loss.item() * args.gradient_accumulation_steps

        # Update parameters every gradient_accumulation_steps
        if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        num_batches += 1
        
        # Only increment global_step when we actually update parameters
        if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
            global_step += 1

            if logger and global_step % args.log_steps == 0:
                step_time = time.time() - step_start_time
                samples_per_sec = args.batch_size / step_time if step_time > 0 else 0
                step_metrics = {
                    "mlm_loss": mlm_loss.item() * args.gradient_accumulation_steps,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "samples_per_sec": samples_per_sec,
                    "step_time": step_time
                }
                if compute_alignment_loss:
                    step_metrics["alignment_loss"] = centroid_alignment_loss.item() * args.gradient_accumulation_steps
                logger.log_step(global_step, step_metrics)
            
            # Run validation every 1000 steps
            if validation_dir and logger and global_step % 1000 == 0:
                tokenizer = dataloader.dataset.tokenizer if hasattr(dataloader.dataset, 'tokenizer') else data_collator.tokenizer
                val_collator = JSONDataCollator(
                    tokenizer=tokenizer,
                    mlm_probability=args.mlm_probability,
                    key_mask_probability=args.key_mask_probability,
                    nonkey_mask_probability=args.nonkey_mask_probability,
                    hybrid_epochs=6, 
                    total_epochs=args.num_epochs,
                )
                val_metrics = run_validation(
                    model, validation_dir, device, val_collator, 
                    args, step_sample_ratio=args.validation_sample_ratio
                )
                if val_metrics:
                    logger.log_validation_step(global_step, val_metrics)
        
        step_time = time.time() - step_start_time
        step_times.append(step_time)

    # Calculate epoch metrics
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    avg_mlm_loss = total_mlm_loss / num_batches
    avg_alignment_loss = total_alignment_loss / num_batches if total_alignment_loss > 0 else 0
    
    # Calculate throughput
    total_samples = num_batches * args.batch_size
    samples_per_sec = total_samples / epoch_duration if epoch_duration > 0 else 0
    avg_step_time = np.mean(step_times) if step_times else 0
    
    # Log epoch metrics
    epoch_metrics = {
        "mlm_loss": float(avg_mlm_loss),
        "alignment_loss": float(avg_alignment_loss) if avg_alignment_loss > 0 else 0.0,
        "total_loss": float(avg_mlm_loss + avg_alignment_loss),
        "samples_processed": int(total_samples),
        "num_batches": int(num_batches),
        "epoch_duration_seconds": float(epoch_duration),
        "samples_per_sec": float(samples_per_sec),
        "avg_step_time": float(avg_step_time)
    }
    
    if logger:
        logger.log_epoch(epoch, epoch_metrics)

    print(f"\n[Epoch {epoch+1}]")
    print(f"�� MLM Loss: {avg_mlm_loss:.4f}")
    if avg_alignment_loss > 0:
        print(f"📉 Alignment Loss: {avg_alignment_loss:.4f}")
    print(f"⏱️  Epoch Duration: {epoch_duration:.2f}s | Throughput: {samples_per_sec:.2f} samples/sec")

    return global_step

def run_validation(model, validation_dir, device, data_collator, args, step_sample_ratio=0.2):
    """
    Run validation on validation dataset.
    Returns validation metrics including losses.
    
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
    
    # Clean and create validation dataset
    processed_val_data = clean_table_data(val_data, tokenizer_name="bert-base-uncased")
    json_objects = [item[1] for item in processed_val_data]
    val_dataset = JSONDataset(
        path=None,
        tokenizer=data_collator.tokenizer,
        max_length=args.max_length,
        path_is="json"
    )
    val_dataset.data = json_objects
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True
    )
    
    total_loss = 0.0
    mlm_loss_total = 0.0
    num_batches = 0
    
    val_start_time = time.time()
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validation", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            key_positions = batch["key_positions"]
            
            # Forward pass (no alignment loss for validation)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                key_positions=key_positions,
                compute_alignment_loss=False
            )
            
            mlm_loss = outputs["mlm_loss"]
            total_loss += mlm_loss.item()
            mlm_loss_total += mlm_loss.item()
            num_batches += 1
    
    model.train()
    
    val_end_time = time.time()
    val_duration = val_end_time - val_start_time
    total_val_samples = num_batches * args.batch_size
    val_samples_per_sec = total_val_samples / val_duration if val_duration > 0 else 0
    
    val_metrics = {
        'total_loss': float(total_loss / num_batches if num_batches > 0 else 0.0),
        'mlm_loss': float(mlm_loss_total / num_batches if num_batches > 0 else 0.0),
        'num_samples': int(total_val_samples),
        'num_batches': int(num_batches),
        'duration_seconds': float(val_duration),
        'samples_per_sec': float(val_samples_per_sec)
    }
    
    print(f"📊 Validation - Total Loss: {val_metrics['total_loss']:.4f}, MLM Loss: {val_metrics['mlm_loss']:.4f}")
    print(f"⏱️  Validation Duration: {val_duration:.2f}s | Throughput: {val_samples_per_sec:.2f} samples/sec")
    
    return val_metrics

def main(args):
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
    
    print(f"📈 Found {len(jsonl_files)} files, sorted by size")
    
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
    
    print(f"📊 Dataset stats: {total_training_rows:,} total rows | "
          f"Mean: {dataset_stats['mean_rows_per_file']:.0f} | "
          f"Min: {dataset_stats['min_rows_per_file']} | "
          f"Max: {dataset_stats['max_rows_per_file']}")
    
    # Group into groups of 4 tables each (as per flowchart)
    tables_per_group = 4
    num_groups = len(jsonl_files) // tables_per_group
    print(f"📦 Creating {num_groups} groups with {tables_per_group} tables each")

    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {args.data_path}")

    print(" Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)

    # Initialize model
    config = BertConfig.from_pretrained(args.pretrained_model)
    model = HAETAE(config, tokenizer)
    model.to(device)

    # Data collator
    data_collator = JSONDataCollator(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
        key_mask_probability=args.key_mask_probability,
        nonkey_mask_probability=args.nonkey_mask_probability,
        hybrid_epochs=6, 
        total_epochs=args.num_epochs,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # ✅ Estimate total training steps based on dataset size
    print("📏 Estimating total training steps...")
    
    # Group files into batches for estimation
    file_groups = [jsonl_files[i:i+args.tables_per_batch] for i in range(0, len(jsonl_files), args.tables_per_batch)]
    steps_per_epoch = 0
    
    for file_group in tqdm(file_groups, desc="Estimating batch lengths", leave=False):
        try:
            combined_dataset = create_combined_dataset(file_group, tokenizer, args.max_length)
            steps_in_group = len(combined_dataset) // args.batch_size + 1
            # Account for gradient accumulation
            steps_per_epoch += steps_in_group // args.gradient_accumulation_steps + 1
        except Exception as e:
            print(f"⚠️ Skipping group due to error: {e}")
            continue
    
    total_steps = args.num_epochs * steps_per_epoch
    print(f"📈 Total estimated training steps: {total_steps}")

    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )
    
    logger = LossLogger(args.output_dir)
    
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
    print(f"📊 Total training steps: {total_steps}")
    print(f"📈 Warmup steps: {args.warmup_steps}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📚 Tables per batch: {args.tables_per_batch}")
    print(f"🔄 Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"📝 Logging to: {logger.get_metrics_file()}")

    global_step = 0
    training_start_time = time.time()

    # Global Epoch Loop (as per flowchart)
    for epoch in range(args.num_epochs):
        print(f"\n🗓️ Global Epoch {epoch + 1}/{args.num_epochs}")
        
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
                tokenizer=tokenizer,
                max_length=args.max_length
            )

            # Set epoch for data collator
            data_collator.set_epoch(epoch)

            # Data loader for current group
            dataloader = DataLoader(
                combined_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=data_collator,
                num_workers=12,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )

            global_step = train_haetae(
                model=model,
                epoch=epoch,
                dataloader=dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                data_collator=data_collator,
                args=args,
                logger=logger,
                global_step=global_step,
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"✅ Group {group_idx + 1} Complete")
            cnt += 1

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
    total_samples_processed = total_training_rows * args.num_epochs
    
    # Log compute metrics (ensure JSON serializable)
    compute_metrics = {
        'total_training_time_seconds': float(total_training_time),
        'total_training_time_hours': float(total_training_hours),
        'total_training_steps': int(global_step),
        'total_epochs': int(args.num_epochs),
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
    
    save_path = os.path.join(args.output_dir, f"epoch_{args.num_epochs}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"�� Model saved: {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)