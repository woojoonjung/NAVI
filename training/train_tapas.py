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
from datetime import datetime
from transformers import TapasForMaskedLM, AutoTokenizer, get_scheduler
from transformers.utils.logging import set_verbosity_info, enable_propagation
from transformers import DataCollatorForLanguageModeling
from dataset.dataset import TapasDataset
from dataset.preprocess import load_jsonl
from dataset.collator import TapasCollator

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

def create_combined_dataset(jsonl_files, max_length, cache_dir=".cache_tokenized"):
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
    
    print(f"🔄 Total combined data: {len(all_data)} rows")
    
    # Create single TapasDataset with all merged data as tuples
    # We need to extract just the json_objects for TapasDataset
    json_objects = [item[1] for item in all_data]
    dataset = TapasDataset(json_objects, max_length=max_length)
    
    # Add table_ids as an attribute for potential future use
    dataset.table_ids = [item[0] for item in all_data]
    
    return dataset

def load_or_tokenize(jsonl_file, max_length, cache_dir=".cache_tokenized"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, os.path.basename(jsonl_file).replace(".jsonl", ".pkl"))

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load cache: {cache_file} — {e}, regenerating...")

    data = load_jsonl(jsonl_file)
    dataset = TapasDataset(data, max_length=max_length)
    with open(cache_file, "wb") as f:
        pickle.dump(dataset, f)
    return dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train TAPAS with NaviCollator")
    parser.add_argument("--data_path", type=str, required=True, help="Path to directory containing JSONL files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--pretrained_model", type=str, default="google/tapas-base-masklm", help="Pretrained model name or path")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of gradient accumulation steps")
    parser.add_argument("--tables_per_batch", type=int, default=4, help="Number of tables to process simultaneously")
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=50, help="Number of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--log_steps", type=int, default=100, help="Evaluate every N steps")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Probability for masking tokens in MLM")
    parser.add_argument("--hv_weight", type=float, default=1.0, help="Weight for HV masking in HVB strategy")
    parser.add_argument("--max_tokens_to_mask", type=int, default=80, help="Maximum tokens to mask per batch")
    return parser.parse_args()

class LossLogger:
    """Logger for tracking training metrics in a structured format."""
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.metrics = {
            'training_info': {
                'start_time': datetime.now().isoformat(),
                'epochs': []
            },
            'steps': [] 
        }
        self.metrics_file = os.path.join(output_dir, "training_metrics_tapas.json")

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

    def save_metrics(self):
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def get_metrics_file(self):
        return self.metrics_file

def train_tapas(
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
):
    model.train()

    print(f"\n Training Batch — Epoch {epoch + 1}")

    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        
        loss = outputs.loss
        
        # Scale loss by gradient accumulation steps
        loss = loss / args.gradient_accumulation_steps
        loss.backward()

        # Update parameters every gradient_accumulation_steps
        if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * args.gradient_accumulation_steps  # Unscale for logging
        num_batches += 1
        
        # Only increment global_step when we actually update parameters
        if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
            global_step += 1

            if logger and global_step % args.log_steps == 0:
                logger.log_step(global_step, {
                    "total_loss": loss.item() * args.gradient_accumulation_steps,
                    "learning_rate": scheduler.get_last_lr()[0],
                })

    # Print avg loss info for this epoch
    avg_total_loss = total_loss / num_batches

    print(f"\n[Epoch {epoch+1}]")
    print(f" Total Loss: {avg_total_loss:.4f}")

    return global_step

def main(args):
    print("🔄 Loading all JSONL files...")
    jsonl_files = sorted([
        os.path.join(args.data_path, f)
        for f in os.listdir(args.data_path)
        if f.endswith(".jsonl")
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
    
    # Group into groups of 4 tables each (as per flowchart)
    tables_per_group = 4
    num_groups = len(jsonl_files) // tables_per_group
    print(f"📦 Creating {num_groups} groups with {tables_per_group} tables each")

    if not jsonl_files:
        raise ValueError(f"No JSONL files found in {args.data_path}")

    print(" Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)

    # Initialize model
    model = TapasForMaskedLM.from_pretrained(args.pretrained_model)
    model.to(device)

    # TapasCollator with HVB strategy and hv_weight=1
    data_collator = TapasCollator(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
        stage_config=[('HVB', args.num_epochs)],  # Use HVB strategy for all epochs
        hv_weight=args.hv_weight,  # Use the hv_weight from args
        token_length_threshold=8,
        mask_replace_prob=0.8,
        random_replace_prob=0.1,
        max_tokens_to_mask=args.max_tokens_to_mask,
        value_ratio=0.5
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
            combined_dataset = create_combined_dataset(file_group, args.max_length)
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

    print("🎯 Starting training...")
    print(f"📊 Total training steps: {total_steps}")
    print(f"📈 Warmup steps: {args.warmup_steps}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📚 Tables per batch: {args.tables_per_batch}")
    print(f"🔄 Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"🎯 Using NaviCollator with HVB strategy, hv_weight={args.hv_weight}")
    print(f" Max tokens to mask: {args.max_tokens_to_mask}")
    print(f" Logging to: {logger.get_metrics_file()}")

    global_step = 0

    # Global Epoch Loop (as per flowchart)
    for epoch in range(args.num_epochs):
        print(f"\n🗓️ Global Epoch {epoch + 1}/{args.num_epochs}")
        
        # Set the current epoch for the NaviCollator
        data_collator.set_epoch(epoch)
        
        # Print current strategy info
        stage_info = data_collator.get_stage_info()
        print(f"🎯 Current strategy: {stage_info['current_strategy']}")
        
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
                max_length=args.max_length
            )

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

            global_step = train_tapas(
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

    save_path = os.path.join(args.output_dir, f"epoch_{args.num_epochs}")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"💾 Model saved: {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)