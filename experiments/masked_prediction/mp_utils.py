"""
Utility functions for masked prediction experiments.

Token-level accuracy: `evaluate_masked_prediction` (micro-averaged over non-special positions).

Cell-level accuracy: `evaluate_masked_prediction_cell_level` — macro-averaged over fields (columns);
only MLM targets (`labels != -100`) inside each field’s span; per-field success if the fraction of
correct masked tokens is >= τ for each threshold in the sweep.
"""

import json
from typing import Dict, List, Optional, Sequence

import pandas as pd
from io import StringIO
import torch
import numpy as np

from model.navi import NaviForMaskedLM
from baselines.haetae.model import HAETAE
from transformers import TapasForMaskedLM, BertForMaskedLM


def load_data(path, path_is="jsonl"):
    """Load data from JSONL or CSV file."""
    data = []

    if path_is == "csv":
        if path is not None:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            table = pd.read_csv(StringIO(''.join(lines)))
            if "class" in table.columns:
                table.drop(columns=["class"], inplace=True)
            for _, row in table.iterrows():
                row_json = row.to_dict() 
                data.append(row_json)

    elif path_is == "jsonl":
        if path is not None:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))

    return data


def mask_entry(example, tokenizer, collator):
    """
    Returns a batch with masked input, labels, and token-level representations.
    """
    batch = collator([example])

    # Decode tokens for visualization
    masked_input_tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][0])
    label_tokens = tokenizer.convert_ids_to_tokens(batch["labels"][0])

    return batch, masked_input_tokens, label_tokens


def forward_mlm_logits(model, batch):
    """
    Single forward pass for MLM logits (NAVI / HAETAE / TAPAS / BERT-style).
    """
    device = next(model.parameters()).device

    if isinstance(model, NaviForMaskedLM):
        input_kwargs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "position_ids": batch["position_ids"].to(device),
            "segment_ids": batch["segment_ids"].to(device) if batch.get("segment_ids") is not None else None,
            "header_strings": batch.get("header_strings"),
        }
        with torch.no_grad():
            outputs = model(**input_kwargs)
        return outputs[1]

    if isinstance(model, HAETAE):
        input_kwargs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "key_positions": batch.get("key_positions"),
        }
        with torch.no_grad():
            outputs = model(**input_kwargs)
        return outputs["logits"]

    if isinstance(model, TapasForMaskedLM):
        input_kwargs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "token_type_ids": batch["token_type_ids"].to(device) if batch.get("token_type_ids") is not None else None,
        }
        with torch.no_grad():
            outputs = model(**input_kwargs)
        return outputs.logits

    input_kwargs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
    }
    with torch.no_grad():
        outputs = model(**input_kwargs)
    return outputs.logits


def predict_masked_token_ids(model, batch) -> torch.Tensor:
    """
    Argmax token ids per position [batch, seq] on CPU (int64).
    """
    logits = forward_mlm_logits(model, batch)
    predicted_ids = torch.argmax(logits, dim=-1)
    return predicted_ids.cpu()


def predict_masked_tokens(model, tokenizer, batch):
    """
    Runs the model and predicts masked tokens in the input.
    """
    predicted_ids = predict_masked_token_ids(model, batch)
    return tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())


def _select_field_position_dict(collator, batch, batch_idx: int = 0) -> Optional[Dict[str, List[int]]]:
    """
    Header vs value field spans: must match CollatorForMaskedPrediction.set_epoch (mask_keys_only).
    """
    if not getattr(collator, "field_target_mode", True):
        return None
    if getattr(collator, "mask_keys_only", None) is None:
        return None
    hdr_list = batch.get("header_positions")
    val_list = batch.get("value_positions")
    if not hdr_list or not val_list:
        return None
    if batch_idx >= len(hdr_list) or batch_idx >= len(val_list):
        return None
    hdr = hdr_list[batch_idx]
    val = val_list[batch_idx]
    if hdr is None or val is None:
        return None
    return hdr if collator.mask_keys_only else val


def evaluate_masked_prediction_cell_level(
    dataset,
    model,
    tokenizer,
    collator,
    epoch: int,
    thresholds: Sequence[float] = (0.5, 0.6, 0.8),
) -> Dict[str, float]:
    """
    Per-field (column) accuracy with a threshold sweep on masked MLM positions only.

    For each field in the active span dict (header_positions if mask_keys_only else value_positions),
    consider only token indices where labels != -100. A field counts toward fields_total if it has at
    least one such masked position. It counts as success for threshold τ if
    (correct_masked / total_masked_in_field) >= τ.

    Requires CollatorForMaskedPrediction in field-targeting mode (epochs 1 and 5 in this repo).
    """
    thresholds = tuple(sorted(set(float(t) for t in thresholds)))
    if collator is not None:
        collator.set_epoch(epoch)

    fields_correct: Dict[float, int] = {t: 0 for t in thresholds}
    fields_total = 0

    if not getattr(collator, "field_target_mode", True):
        print(
            "⚠️  Cell-level eval: collator is not in field_targeting mode "
            "(use epoch < field_targeting_epochs, e.g. 1 or 5)."
        )
        return {f"cell_acc@{t:g}": 0.0 for t in thresholds}

    for example in dataset:
        batch = collator([example])
        field_dict = _select_field_position_dict(collator, batch, 0)
        if not field_dict:
            continue

        labels = batch["labels"][0]
        pred_ids = predict_masked_token_ids(model, batch)[0]
        seq_len = labels.shape[0]

        for _fname, positions in field_dict.items():
            if not positions:
                continue
            masked_idx: List[int] = []
            for i in positions:
                if i < 0 or i >= seq_len:
                    continue
                if labels[i].item() != -100:
                    masked_idx.append(i)
            if not masked_idx:
                continue
            gold = labels[masked_idx].long()
            pred = pred_ids[masked_idx].long()
            correct = int((gold == pred).sum().item())
            total_m = len(masked_idx)
            frac = correct / total_m
            fields_total += 1
            for tau in thresholds:
                if frac >= tau:
                    fields_correct[tau] += 1

    if fields_total == 0:
        print("⚠️  Cell-level eval: no fields with masked tokens (fields_total=0).")
        return {f"cell_acc@{t:g}": 0.0 for t in thresholds}

    out: Dict[str, float] = {}
    print(f"   Cell-level (per-field, masked tokens only): fields_total={fields_total}")
    for tau in thresholds:
        acc = fields_correct[tau] / fields_total
        key = f"cell_acc@{tau:g}"
        out[key] = float(acc)
        print(f"   {key} = {acc:.4f} ({fields_correct[tau]}/{fields_total})")
    return out


def evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch):
    """
    Original token-level masked prediction metric (unchanged): micro-averaged over non-special positions.

    Cell-level metrics are separate; see `evaluate_masked_prediction_cell_level`. Uses collator.set_epoch(epoch).
    """
    if collator is not None:
        collator.set_epoch(epoch)
    
    correct = 0
    total = 0

    for i, example in enumerate(dataset):
        batch, masked_tokens, label_tokens = mask_entry(example, tokenizer, collator)
        predicted_tokens = predict_masked_tokens(model, tokenizer, batch)

        # Evaluate prediction quality
        for label, pred in zip(label_tokens, predicted_tokens):
            if label in ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]:
                continue
            if label == pred:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"✅ Accuracy: {correct}/{total} = {accuracy:.4f}")

    return accuracy
