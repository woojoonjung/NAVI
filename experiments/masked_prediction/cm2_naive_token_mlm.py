"""
Naive token-level masked prediction for CM2 (not CM2's structured feature masking).

CM2's encoder operates on *pooled* column vectors, so it cannot emit per-token logits.
This eval instead:

  1. Tokenizes each table cell the same way as CM2FeatureExtractor (BERT tokenizer).
  2. Replaces one token with [MASK].
  3. Embeds the sequence with CM2's *value* token embedding (`word_embeddings_value`).
  4. Builds a context vector = mean embedding over other (non-mask, non-pad) tokens.
  5. Predicts the masked token id as argmax(context @ E^T) with E = value embedding table.

This is a shallow "bag-of-neighbors" MLM probe using CM2's embedding weights — comparable in spirit
to reporting token accuracy, while being honest that the full transformer encoder is not used for
the softmax over vocabulary.

Optional: same for *header* tokens (column-name strings) using `word_embeddings_header`.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from experiments.downstream_tasks.cm2_utils import json_row_to_cm2_dataframe


def _trim_pad(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """1D long tensor, drop trailing pad only (CM2 pads on the right)."""
    ids = ids.long()
    if ids.numel() == 0:
        return ids
    nonzero = (ids != pad_id).nonzero(as_tuple=True)[0]
    if nonzero.numel() == 0:
        return ids[:0]
    last = int(nonzero[-1].item()) + 1
    return ids[:last]


def _predict_masked_token_value(
    we: Any,
    ids_trim: torch.Tensor,
    pos: int,
    mask_id: int,
    pad_id: int,
    device: torch.device,
) -> Tuple[int, int]:
    """
    Returns (pred_id, true_id). Uses value embeddings only.
    """
    true_id = int(ids_trim[pos].item())
    ids_m = ids_trim.clone()
    ids_m[pos] = mask_id
    L = ids_m.numel()
    inp = ids_m.unsqueeze(0).to(device)
    emb = we(inp, emb_type="value")[0]  # [L, D]
    mask_positions = torch.tensor([pos], device=device, dtype=torch.long)
    valid = torch.ones(L, dtype=torch.bool, device=device)
    valid[mask_positions] = False
    valid &= ids_m.to(device) != pad_id
    if valid.sum() == 0:
        return -1, true_id
    ctx = emb[valid].mean(dim=0)
    w = we.word_embeddings_value.weight
    logits = F.linear(ctx, w, bias=None)
    pred_id = int(logits.argmax(-1).item())
    return pred_id, true_id


def _predict_masked_token_header(
    we: Any,
    ids_trim: torch.Tensor,
    pos: int,
    mask_id: int,
    pad_id: int,
    device: torch.device,
) -> Tuple[int, int]:
    """Column-name tokens use header embedding table."""
    true_id = int(ids_trim[pos].item())
    ids_m = ids_trim.clone()
    ids_m[pos] = mask_id
    L = ids_m.numel()
    inp = ids_m.unsqueeze(0).to(device)
    emb = we(inp, emb_type="header")[0]
    valid = torch.ones(L, dtype=torch.bool, device=device)
    valid[pos] = False
    valid &= ids_m.to(device) != pad_id
    if valid.sum() == 0:
        return -1, true_id
    ctx = emb[valid].mean(dim=0)
    w = we.word_embeddings_header.weight
    logits = F.linear(ctx, w, bias=None)
    pred_id = int(logits.argmax(-1).item())
    return pred_id, true_id


def evaluate_cm2_naive_token_mlm(
    rows: List[Dict[str, Any]],
    model: Any,
    table_flag: int = 0,
    base_seed: int = 42,
    include_column_name_tokens: bool = True,
) -> Dict[str, float]:
    """
    Per table row: tokenize cells like CM2, mask one random token per eligible cell, predict id.

    Returns dict with keys ``value_token_accuracy`` (cell content) and optionally
    ``header_name_token_accuracy`` (column-name strings) when ``include_column_name_tokens``.
    """
    model.eval()
    device = next(model.parameters()).device
    fe = model.input_encoder.feature_extractor
    we = model.input_encoder.feature_processor.word_embedding
    tok = fe.tokenizer
    mask_id = tok.mask_token_id
    pad_id = tok.pad_token_id

    v_correct, v_total = 0, 0
    h_correct, h_total = 0, 0

    for row_idx, row in enumerate(rows):
        df = json_row_to_cm2_dataframe(row)
        rng = random.Random(base_seed + row_idx * 7919)

        with torch.no_grad():
            inputs = fe(df, table_flag=table_flag)

            # --- Cell value tokens (x_cat_input_ids) ---
            x_cat = inputs.get("x_cat_input_ids")
            if x_cat is not None:
                x_cat = x_cat.to(device)
                _, n_cat, _ = x_cat.shape
                for j in range(n_cat):
                    ids = _trim_pad(x_cat[0, j], pad_id)
                    if ids.numel() < 2:
                        continue
                    pos = rng.randint(0, ids.numel() - 1)
                    pred, true = _predict_masked_token_value(we, ids, pos, mask_id, pad_id, device)
                    if pred < 0:
                        continue
                    v_total += 1
                    if pred == true:
                        v_correct += 1

            # --- Column header tokens (names of cat cols and num cols) ---
            if include_column_name_tokens:
                for key in ("col_cat_input_ids", "num_col_input_ids"):
                    t = inputs.get(key)
                    if t is None:
                        continue
                    t = t.to(device)
                    # shape [1, cols, L] or [1, L] for num cols
                    if t.dim() == 2:
                        ids = _trim_pad(t[0], pad_id)
                        if ids.numel() < 2:
                            continue
                        pos = rng.randint(0, ids.numel() - 1)
                        pred, true = _predict_masked_token_header(we, ids, pos, mask_id, pad_id, device)
                        if pred < 0:
                            continue
                        h_total += 1
                        if pred == true:
                            h_correct += 1
                    elif t.dim() == 3:
                        _, ncols, _ = t.shape
                        for j in range(ncols):
                            ids = _trim_pad(t[0, j], pad_id)
                            if ids.numel() < 2:
                                continue
                            pos = rng.randint(0, ids.numel() - 1)
                            pred, true = _predict_masked_token_header(we, ids, pos, mask_id, pad_id, device)
                            if pred < 0:
                                continue
                            h_total += 1
                            if pred == true:
                                h_correct += 1

    def _rate(c: int, t: int) -> float:
        return float(c / t) if t else 0.0

    val_acc = _rate(v_correct, v_total)
    hdr_acc = _rate(h_correct, h_total)

    print("\nCM2 naive token MLM (embedding-table softmax; encoder not used for vocab logits)")
    print("  Cell value tokens:")
    print(f"✅ Accuracy: {v_correct}/{v_total} = {val_acc:.4f}")
    if include_column_name_tokens:
        print("  Column-name (header) tokens:")
        print(f"✅ Accuracy: {h_correct}/{h_total} = {hdr_acc:.4f}")

    return {
        "value_token_accuracy": val_acc,
        "header_name_token_accuracy": hdr_acc,
        "value_correct": v_correct,
        "value_total": v_total,
        "header_correct": h_correct,
        "header_total": h_total,
    }


def print_cm2_naive_token_mlm_disclaimer() -> None:
    print(
        "\nNote: This does not use CM2’s structured mask heads or encoder for token softmax — only "
        "CM2’s tokenizer + value/header embedding tables with a mean-pooled context vector. "
        "It is meant as a naive token-id baseline comparable to “mask one token, predict id”.\n"
    )
