"""
CM2 masked-feature evaluation aligned with experiments/masked_prediction/mp_utils.py reporting.

Uses CM2ForMask tabular masking (not BERT token MLM). Maps:
  - Header (collator epoch=1)  -> mask only categorical feature positions
  - Value  (collator epoch=5)  -> mask only numerical feature positions

Accuracy:
  - Numerical: fraction of masked num cells with |minmax(pred) - minmax(target)| < tol
    (same normalization as cal_mask_num_features_loss).
  - Categorical: fraction of masked cat cells with cosine(pred, target_emb) > cat_cos_min
    (same objective family as cal_mask_cat_features_loss).
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import torch
import torch.nn.functional as F

from experiments.downstream_tasks.cm2_utils import json_row_to_cm2_dataframe
from experiments.masked_prediction.unified_semantic_stats import (
    print_unified_semantic_block,
    summarize_scores,
)

MaskScope = Literal["cat", "num"]


def _sanitize_mask_like_forward(masked_indices: torch.Tensor) -> torch.Tensor:
    """Match CM2ForMask.forward mask bookkeeping (at least one mask, not all)."""
    masked_indices = masked_indices.clone().float()
    indixes_tesor = torch.full(masked_indices.shape, False, dtype=torch.bool, device=masked_indices.device)
    sum_masked = torch.sum(masked_indices, dim=-1, keepdim=True)
    indixes_tesor[:, :1] = sum_masked == 0
    masked_indices[indixes_tesor] = 1
    indixes_tesor[:, :1] = sum_masked == masked_indices.shape[1]
    masked_indices[indixes_tesor] = 0
    return masked_indices.int()


def _sample_override_mask(
    attention_mask: torch.Tensor,
    num_cnt: int,
    cat_cnt: int,
    scope: MaskScope,
    mlm_probability: float,
    seed: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Build [1, num_cnt+cat_cnt] mask: only cat or only num positions may be non-zero."""
    torch.manual_seed(seed)
    if scope == "num":
        if num_cnt == 0:
            return None
        m = torch.bernoulli(torch.full((1, num_cnt), mlm_probability, device=device))
        c = torch.zeros(1, cat_cnt, device=device)
    else:
        if cat_cnt == 0:
            return None
        m = torch.zeros(1, num_cnt, device=device)
        c = torch.bernoulli(torch.full((1, cat_cnt), mlm_probability, device=device))
    mi = torch.cat([m, c], dim=-1)
    return _sanitize_mask_like_forward(mi)


def _count_num_correct(model: Any, aux: Dict[str, Any], num_abs_tol: float) -> tuple[int, int]:
    num_proj = aux.get("num_proj")
    num_mask = aux.get("num_mask")
    x_num = aux.get("x_num")
    if num_proj is None or num_mask is None or x_num is None:
        return 0, 0
    if not num_mask.bool().any():
        return 0, 0
    out = model._minmax_norm(num_proj)
    xn = x_num.unsqueeze(-1)
    diff = (out[num_mask.bool()] - xn[num_mask.bool()]).abs()
    correct = int((diff < num_abs_tol).sum().item())
    total = int(num_mask.bool().sum().item())
    return correct, total


def _num_similarity_scores(model: Any, aux: Dict[str, Any], num_abs_tol: float) -> List[float]:
    """
    Per masked numerical cell: clamp(1 - |pred_norm - target| / tol, 0, 1) in minmax-normalized space.
    """
    num_proj = aux.get("num_proj")
    num_mask = aux.get("num_mask")
    x_num = aux.get("x_num")
    if num_proj is None or num_mask is None or x_num is None:
        return []
    if not num_mask.bool().any():
        return []
    out = model._minmax_norm(num_proj)
    xn = x_num.unsqueeze(-1)
    diff = (out[num_mask.bool()] - xn[num_mask.bool()]).abs()
    if diff.dim() > 1:
        diff = diff.squeeze(-1)
    tol = max(float(num_abs_tol), 1e-8)
    scores = (1.0 - diff / tol).clamp(0.0, 1.0)
    return [float(x) for x in scores.detach().cpu().tolist()]


def _count_cat_correct(
    aux: Dict[str, Any], cat_cos_min: float
) -> tuple[int, int, Optional[torch.Tensor]]:
    """Returns (correct, total, cos_values) for diagnostics."""
    cat_proj = aux.get("cat_proj")
    cat_mask = aux.get("cat_mask")
    cat_tgt = aux.get("cat_bert_emb")
    if cat_proj is None or cat_mask is None or cat_tgt is None:
        return 0, 0, None
    if not cat_mask.bool().any():
        return 0, 0, None
    cos = F.cosine_similarity(cat_proj[cat_mask.bool()], cat_tgt[cat_mask.bool()], dim=-1)
    correct = int((cos > cat_cos_min).sum().item())
    total = int(cat_mask.bool().sum().item())
    return correct, total, cos


def evaluate_cm2_masked_prediction(
    rows: List[Dict[str, Any]],
    model: Any,
    table_flag: int = 0,
    mask_scope: MaskScope = "num",
    mlm_probability: float = 0.35,
    base_seed: int = 42,
    num_abs_tol: float = 0.05,
    cat_cos_min: float = 0.70,
) -> float:
    """
    One pass over rows: for each row, sample a stochastic mask (reproducible seed) on num or cat
    positions only, run CM2ForMask with masked_indices_override + return_aux, aggregate accuracy.

    mask_scope 'num' -> value-style (numerical cells); 'cat' -> header-style (categorical cells).
    """
    model.eval()
    device = next(model.parameters()).device
    correct = 0
    total = 0
    skipped_no_slots = 0
    cos_list: List[torch.Tensor] = []

    for row_idx, row in enumerate(rows):
        df = json_row_to_cm2_dataframe(row)
        seed = base_seed + row_idx * 1009 + (31 if mask_scope == "cat" else 97)

        with torch.no_grad():
            inputs = model.input_encoder.feature_extractor(df, table_flag=table_flag)
            outputs, other_info = model.input_encoder.feature_processor(**inputs)
            num_cnt = other_info["num_cnt"]
            col_emb = other_info["col_emb"]
            cat_cnt = col_emb.shape[0] - num_cnt

            if mask_scope == "num" and num_cnt == 0:
                skipped_no_slots += 1
                continue
            if mask_scope == "cat" and cat_cnt == 0:
                skipped_no_slots += 1
                continue

            mi = _sample_override_mask(
                outputs["attention_mask"],
                num_cnt,
                cat_cnt,
                mask_scope,
                mlm_probability,
                seed,
                device,
            )
            if mi is None:
                continue

            out = model(df, table_flag=table_flag, masked_indices_override=mi, return_aux=True)
            if len(out) != 3:
                continue
            _, _loss, aux = out

            if mask_scope == "num":
                c, t = _count_num_correct(model, aux, num_abs_tol)
            else:
                c, t, cos_vals = _count_cat_correct(aux, cat_cos_min)
                if cos_vals is not None and cos_vals.numel() > 0:
                    cos_list.append(cos_vals.detach().cpu())

            correct += c
            total += t

    if total == 0:
        if mask_scope == "num" and skipped_no_slots == len(rows):
            print(
                "✅ Accuracy: 0/0 = 0.0000 (CM2 has 0 numerical columns for every row — common for "
                "WDC Movie JSON; try --domain Product or data with numeric fields.)"
            )
        elif mask_scope == "cat" and skipped_no_slots == len(rows):
            print(
                "✅ Accuracy: 0/0 = 0.0000 (CM2 has 0 categorical columns for every row; check extractor/checkpoint.)"
            )
        else:
            print("✅ Accuracy: 0/0 = 0.0000 (no masked positions; check schema vs checkpoint)")
        return 0.0

    acc = correct / total
    print(f"✅ Accuracy: {correct}/{total} = {acc:.4f}")
    if mask_scope == "cat":
        print(f"   (counting masked cat cells with cosine(pred,target) > {cat_cos_min}; training minimizes 1−cos, so 0.95 is often too strict.)")
    else:
        print(f"   (counting masked num cells with |pred_norm−target| < {num_abs_tol}.)")
    if cos_list:
        all_cos = torch.cat(cos_list)
        print(
            f"   Masked cosine sim: mean={all_cos.mean().item():.4f}, min={all_cos.min().item():.4f}, max={all_cos.max().item():.4f}"
        )
    if skipped_no_slots and mask_scope == "num":
        print(f"   Rows skipped (no numerical slots in CM2): {skipped_no_slots}/{len(rows)}")
    return float(acc)


def evaluate_cm2_unified_semantic_scores(
    rows: List[Dict[str, Any]],
    model: Any,
    table_flag: int = 0,
    mlm_probability: float = 0.35,
    base_seed: int = 42,
    num_abs_tol: float = 0.05,
) -> None:
    """
    Same masking loop as evaluate_cm2_masked_prediction, but collect per-mask scores and print
    min/max/avg/Q1/Q2/Q3 like NAVI unified_semantic.

    Categorical: cosine(pred, target_emb) per masked cat cell.
    Numerical: similarity = clamp(1 - |pred_norm - target| / num_abs_tol, 0, 1) per masked num cell.
    """
    model.eval()
    device = next(model.parameters()).device
    cat_scores: List[float] = []
    num_scores: List[float] = []

    for row_idx, row in enumerate(rows):
        df = json_row_to_cm2_dataframe(row)

        for mask_scope, score_list in (("cat", cat_scores), ("num", num_scores)):
            seed = base_seed + row_idx * 1009 + (31 if mask_scope == "cat" else 97)

            with torch.no_grad():
                inputs = model.input_encoder.feature_extractor(df, table_flag=table_flag)
                outputs, other_info = model.input_encoder.feature_processor(**inputs)
                num_cnt = other_info["num_cnt"]
                col_emb = other_info["col_emb"]
                cat_cnt = col_emb.shape[0] - num_cnt

                if mask_scope == "num" and num_cnt == 0:
                    continue
                if mask_scope == "cat" and cat_cnt == 0:
                    continue

                mi = _sample_override_mask(
                    outputs["attention_mask"],
                    num_cnt,
                    cat_cnt,
                    mask_scope,  # type: ignore[arg-type]
                    mlm_probability,
                    seed,
                    device,
                )
                if mi is None:
                    continue

                out = model(df, table_flag=table_flag, masked_indices_override=mi, return_aux=True)
                if len(out) != 3:
                    continue
                _, _loss, aux = out

                if mask_scope == "num":
                    score_list.extend(_num_similarity_scores(model, aux, num_abs_tol))
                else:
                    cat_proj = aux.get("cat_proj")
                    cat_mask = aux.get("cat_mask")
                    cat_tgt = aux.get("cat_bert_emb")
                    if cat_proj is None or cat_mask is None or cat_tgt is None:
                        continue
                    if not cat_mask.bool().any():
                        continue
                    cos = F.cosine_similarity(
                        cat_proj[cat_mask.bool()], cat_tgt[cat_mask.bool()], dim=-1
                    )
                    score_list.extend([float(x) for x in cos.detach().cpu().tolist()])

    print("\nCM2 — unified semantic (tabular masked features)")
    print("  Categorical: cosine(pred, target_emb); numerical: clamp(1 - |err|/tol, 0, 1).")
    print_unified_semantic_block(
        "CM2", "Header (categorical)", summarize_scores(cat_scores), len(cat_scores)
    )
    print_unified_semantic_block(
        "CM2", "Value (numerical)", summarize_scores(num_scores), len(num_scores)
    )


def evaluate_cm2_like_baselines(
    rows: List[Dict[str, Any]],
    model: Any,
    table_flag: int = 0,
    mlm_probability: float = 0.35,
    base_seed: int = 42,
    num_abs_tol: float = 0.05,
    cat_cos_min: float = 0.70,
) -> Dict[str, float]:
    """Print Header / Value blocks in the same order as evaluate_baselines (BERT/TAPAS/NAVI)."""
    print("\nCM2 (tabular masked features)")
    print("Header (categorical):")
    header_acc = evaluate_cm2_masked_prediction(
        rows,
        model,
        table_flag=table_flag,
        mask_scope="cat",
        mlm_probability=mlm_probability,
        base_seed=base_seed,
        num_abs_tol=num_abs_tol,
        cat_cos_min=cat_cos_min,
    )
    print("Value (numerical):")
    value_acc = evaluate_cm2_masked_prediction(
        rows,
        model,
        table_flag=table_flag,
        mask_scope="num",
        mlm_probability=mlm_probability,
        base_seed=base_seed,
        num_abs_tol=num_abs_tol,
        cat_cos_min=cat_cos_min,
    )
    return {"header_accuracy": header_acc, "value_accuracy": value_acc}
