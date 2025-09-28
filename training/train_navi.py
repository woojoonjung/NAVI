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
from datetime import datetime
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

def create_combined_dataset(jsonl_files, tokenizer_name, max_length, ablation_mode, compute_field_entropy=True):
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
        all_data,
        tokenizer_name=tokenizer_name,
        max_tokens=500,
        max_indexed_fields=3,
        max_tokens_per_field=20
    )
    
    print(f"🔄 Total combined data: {len(all_data)} rows")
    
    # Create single NaviDataset with all merged data as tuples
    dataset = NaviDataset(all_data, tokenizer_name=tokenizer_name, max_length=max_length, ablation_mode=ablation_mode, compute_field_entropy=compute_field_entropy)
    return dataset

def load_or_tokenize(jsonl_file, tokenizer_name, max_length, ablation_mode, cache_dir=".cache_tokenized"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, os.path.basename(jsonl_file).replace(".jsonl", ".pkl"))

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load cache: {cache_file} — {e}, regenerating...")

    data = load_jsonl(jsonl_file)
    dataset = NaviDataset(data, tokenizer_name=tokenizer_name, max_length=max_length, ablation_mode=ablation_mode)
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
                       choices=['full', 'woED', 'woSI', "woSMLM"],
                       help="Ablation type to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of gradient accumulation steps")
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
    return parser.parse_args()

def get_stage_config(masking_strategy, ablation_type):
    # For woSMLM ablation, force BERT-style masking regardless of input strategy
    if ablation_type == 'woSMLM':
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
    def __init__(self, output_dir, ablation_type, masking_strategy):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.ablation_type = ablation_type
        self.masking_strategy = masking_strategy
        self.metrics = {
            'training_info': {
                'ablation_type': ablation_type,
                'masking_strategy': masking_strategy,
                'start_time': datetime.now().isoformat(),
                'epochs': []
            },
            'steps': [] 
        }
        self.metrics_file = os.path.join(output_dir, f"training_metrics_{ablation_type}_{masking_strategy}.json")

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

    def save_metrics(self):
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def get_metrics_file(self):
        return self.metrics_file

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
):
    model.train()
    collator.set_epoch(epoch)

    current_strategy = collator.get_current_strategy()
    stage_info = collator.get_stage_info()

    print(f"\n📂 Training Batch — Epoch {epoch + 1} | Stage {stage_info['current_stage']}/{stage_info['total_stages']} | Strategy: {current_strategy}")

    total_loss = 0.0
    mlm_loss_total = 0.0
    dist_loss_total = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, desc="Training", leave=False)):
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
            if ablation_type != "woMLM":
                mlm_loss = loss_fns["mlm_loss"](mlm_logits, labels)
            else:
                mlm_loss = None

            entropy_contrast_loss = None

            if ablation_type != "woED":
                table_ids = batch.get("table_ids")
                header_strings = batch.get("header_strings")

                entropy_contrast_loss = loss_fns["entropy_loss"](
                    E_univ=header_anchor_embeds,     # (B, H, D)
                    H_ctx=header_embeds,             # (B, H, D)
                    V_ctx=val_embeds,                # (B, H, D)
                    header_strings=header_strings,   # List[List[str]]
                    table_ids=table_ids,             # List[int]
                    field_categories=field_categories  # from main()
                )

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

        num_batches += 1
        
        # Only increment global_step when we actually update parameters
        if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
            global_step += 1

            if logger and global_step % args.log_steps == 0:
                logger.log_step(global_step, {
                    "total_loss": total_batch_loss.item() * args.gradient_accumulation_steps,
                    "mlm_loss": mlm_loss.item() if mlm_loss is not None else None,
                    "entropy_contrast_loss": entropy_contrast_loss.item() if entropy_contrast_loss else None,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "loss_weights": weights
                })

    # Print avg loss info for this file
    avg_total_loss = total_loss / num_batches
    avg_mlm_loss = mlm_loss_total / num_batches
    avg_entropy_contrast_loss = dist_loss_total / num_batches
    print(f"\n[Epoch {epoch+1}]")
    print(f"📉 Total Loss: {avg_total_loss:.4f}")
    if ablation_type != "woMLM":
        print(f"🔠 MLM Loss: {avg_mlm_loss:.4f}")
    if ablation_type != "woED":
        print(f"🧬 Entropy Contrastive Loss: {avg_entropy_contrast_loss:.4f}")
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

    # Sort by line count (descending)
    file_sizes.sort(key=lambda x: x[1], reverse=True)
    jsonl_files = [f[0] for f in file_sizes]
    
    print(f"📈 Found {len(jsonl_files)} files, sorted by size")
    
    # Group into 25 groups of 4 tables each (as per flowchart)
    tables_per_group = 4
    num_groups = len(jsonl_files) // tables_per_group
    print(f"📦 Creating {num_groups} groups with {tables_per_group} tables each")

    masking_strategy = args.masking_strategy
    ablation_type = args.ablation_type

    # For woSMLM ablation, override masking strategy to B
    if ablation_type == 'woSMLM':
        masking_strategy = 'B'
        print(f"woSMLM ablation: Forcing masking strategy to 'B' (BERT-style)")

    stage_config = get_stage_config(masking_strategy, ablation_type)
    total_epochs = args.num_epochs
    num_files = len(jsonl_files)

    print("🚀 Initializing model...")
    device = torch.device(config.DEVICE)
    model = NaviForMaskedLM(
        bert_name=config.BERT_NAME,
        ablation_mode=ablation_type
    ).to(device)

    loss_fns = {
        "mlm_loss": MLMLoss(),
        "entropy_loss": EntropyAwareContrastiveLoss(model, low_entropy_tau=args.low_entropy_tau)
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler("cuda")

    # ✅ Estimate total training steps based on dataset size
    print("📏 Estimating total training steps...")
    
    # Group files into batches for estimation
    file_groups = [jsonl_files[i:i+args.tables_per_batch] for i in range(0, len(jsonl_files), args.tables_per_batch)]
    steps_per_epoch = 0
    
    for file_group in tqdm(file_groups, desc="Estimating batch lengths", leave=False):
        try:
            combined_dataset = create_combined_dataset(file_group, config.BERT_NAME, config.MAX_SEQ_LENGTH, ablation_type)
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
    logger = LossLogger(args.output_dir, ablation_type, masking_strategy)

    print("🎯 Starting training...")
    print(f"📊 Masking strategy: {masking_strategy}")
    print(f"🔧 Ablation: {ablation_type}")
    print(f"📈 Stage config: {stage_config}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📚 Tables per batch: {args.tables_per_batch}")
    print(f"🔄 Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"📝 Logging to: {logger.get_metrics_file()}")

    global_step = 0

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
                compute_field_entropy=True
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
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"✅ Group {group_idx + 1} Complete")
            cnt += 1

    save_path = os.path.join(args.output_dir, f"epoch_{total_epochs}")
    model.save_pretrained(save_path)
    print(f"Model saved: {save_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)