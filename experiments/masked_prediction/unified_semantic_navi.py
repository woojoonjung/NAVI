"""
NAVI unified semantic evaluation: Approach A (masked vs gold-from-labels forward) with per-field cosine.

Each sample is one field with at least one masked token: mean-pool hidden states at masked
positions, compare cosine(pred, gold) at field level. Uses full-field masking via collator flag.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from model.navi import NaviForMaskedLM

from experiments.masked_prediction.mp_utils import _select_field_position_dict
from experiments.masked_prediction.unified_semantic_stats import (
    print_unified_semantic_block,
    summarize_scores,
)


def forward_navi_hidden_states(model: NaviForMaskedLM, batch: dict) -> torch.Tensor:
    device = next(model.parameters()).device
    input_kwargs = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "position_ids": batch["position_ids"].to(device),
        "segment_ids": batch["segment_ids"].to(device) if batch.get("segment_ids") is not None else None,
        "header_strings": batch.get("header_strings"),
    }
    with torch.no_grad():
        hidden, _logits = model(**input_kwargs)
    return hidden


def _batch_with_gold_input_ids(batch: dict) -> dict:
    """Restore gold subwords at MLM positions (labels hold original ids)."""
    out = {k: v for k, v in batch.items() if k not in ("input_ids",)}
    labels = batch["labels"]
    input_ids = batch["input_ids"].clone()
    m = labels != -100
    input_ids[m] = labels[m]
    out["input_ids"] = input_ids
    return out


def _collect_field_cosines_for_epoch(dataset, model, collator, epoch: int) -> List[float]:
    collator.set_epoch(epoch)
    scores: List[float] = []

    for example in dataset:
        batch = collator([example])
        field_dict = _select_field_position_dict(collator, batch, 0)
        if not field_dict:
            continue

        labels = batch["labels"][0]
        seq_len = labels.shape[0]

        hidden_pred = forward_navi_hidden_states(model, batch)
        hidden_gold = forward_navi_hidden_states(model, _batch_with_gold_input_ids(batch))

        for _fname, positions in field_dict.items():
            if not positions:
                continue
            masked_idx = [
                i for i in positions
                if 0 <= i < seq_len and labels[i].item() != -100
            ]
            if not masked_idx:
                continue
            idx = torch.tensor(masked_idx, device=hidden_pred.device, dtype=torch.long)
            hp = hidden_pred[0, idx].mean(dim=0)
            hg = hidden_gold[0, idx].mean(dim=0)
            cos = F.cosine_similarity(hp.unsqueeze(0), hg.unsqueeze(0), dim=-1).item()
            scores.append(cos)

    return scores


def evaluate_navi_unified_semantic(dataset, model: NaviForMaskedLM, collator) -> None:
    print("\nNAVI — unified semantic (dual-forward, field-level cosine, full-field mask)")
    scores_h = _collect_field_cosines_for_epoch(dataset, model, collator, epoch=1)
    print_unified_semantic_block("NAVI", "Header (categorical fields)", summarize_scores(scores_h), len(scores_h))
    scores_v = _collect_field_cosines_for_epoch(dataset, model, collator, epoch=5)
    print_unified_semantic_block("NAVI", "Value (numerical fields)", summarize_scores(scores_v), len(scores_v))


def _tensor_batch_for_feature(example: dict, tokenizer, positions: List[int]) -> dict:
    """Build a single-row batch with one combined header+value feature fully masked."""
    input_ids = example["input_ids"].unsqueeze(0).clone()
    seq_len = input_ids.shape[1]
    valid_positions = sorted({int(pos) for pos in positions if 0 <= int(pos) < seq_len})

    labels = torch.full_like(input_ids, -100)
    if valid_positions:
        idx = torch.tensor(valid_positions, dtype=torch.long)
        labels[0, idx] = input_ids[0, idx]
        input_ids[0, idx] = tokenizer.mask_token_id

    batch = {
        "input_ids": input_ids,
        "attention_mask": example["attention_mask"].unsqueeze(0),
        "labels": labels,
        "header_positions": [example.get("header_positions")],
        "value_positions": [example.get("value_positions")],
    }
    for optional_key in ("position_ids", "segment_ids"):
        value = example.get(optional_key)
        if value is not None:
            batch[optional_key] = value.unsqueeze(0)
    if example.get("header_strings") is not None:
        batch["header_strings"] = [example.get("header_strings")]
    return batch


def collect_navi_feature_unified_semantic_scores(
    dataset,
    model: NaviForMaskedLM,
    tokenizer,
) -> Tuple[List[float], Dict[str, int]]:
    """
    Return feature-level cosine similarities for NAVI.

    Each feature is one column-value unit: concatenate the token positions for a
    field's header span and value span, mask that full span, then compare the
    pooled hidden state from masked input against the pooled hidden state after
    restoring the gold tokens.
    """
    scores: List[float] = []
    diagnostics = {
        "features_total": 0,
        "features_scored": 0,
        "missing_value_span": 0,
        "empty_feature_span": 0,
    }

    for example in dataset:
        header_positions = example.get("header_positions") or {}
        value_positions = example.get("value_positions") or {}
        if not header_positions:
            continue

        for field_name, header_span in header_positions.items():
            diagnostics["features_total"] += 1
            value_span = value_positions.get(field_name)
            if not value_span:
                diagnostics["missing_value_span"] += 1
                continue

            feature_positions = list(header_span or []) + list(value_span or [])
            if not feature_positions:
                diagnostics["empty_feature_span"] += 1
                continue

            batch = _tensor_batch_for_feature(example, tokenizer, feature_positions)
            labels = batch["labels"][0]
            masked_idx = [
                int(i)
                for i in feature_positions
                if 0 <= int(i) < labels.shape[0] and labels[int(i)].item() != -100
            ]
            if not masked_idx:
                diagnostics["empty_feature_span"] += 1
                continue

            hidden_pred = forward_navi_hidden_states(model, batch)
            hidden_gold = forward_navi_hidden_states(model, _batch_with_gold_input_ids(batch))
            idx = torch.tensor(masked_idx, device=hidden_pred.device, dtype=torch.long)
            hp = hidden_pred[0, idx].mean(dim=0)
            hg = hidden_gold[0, idx].mean(dim=0)
            scores.append(F.cosine_similarity(hp.unsqueeze(0), hg.unsqueeze(0), dim=-1).item())
            diagnostics["features_scored"] += 1

    return scores, diagnostics
