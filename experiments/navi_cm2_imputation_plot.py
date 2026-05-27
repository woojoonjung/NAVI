#!/usr/bin/env python
"""NAVI feature-level vs CM2 categorical cosine similarity distribution plot.

Computes:
  - NAVI: header+value span combined masking, cosine(masked_hidden, gold_hidden)
  - CM2:  per-cell categorical cosine(pred_proj, target_emb)

Then stratified-downsamples NAVI scores to match the CM2 count while
preserving the NAVI distribution shape, and saves a histogram PNG.

Usage (conda activate CM2, from repo root):
    python experiments/navi_cm2_imputation_plot.py \
        --domain Movie --run_id my_run --output_dir experiments/navi_cm2_results
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from config import config as app_config
except ImportError:
    app_config = None

DATA_ROOT = _REPO_ROOT / "data"

from dataset.dataset import NaviDataset
from experiments.masked_prediction.mp_utils import load_data
from experiments.masked_prediction.unified_semantic_navi import (
    collect_navi_feature_unified_semantic_scores,
    forward_navi_hidden_states,
    _batch_with_gold_input_ids,
)
from experiments.masked_prediction.cm2_mp_utils import (
    _sample_override_mask,
    _num_similarity_scores,
    json_row_to_cm2_dataframe,
)
from experiments.downstream_tasks.cm2_utils import (
    default_cm2_movie_ckpt,
    default_cm2_product_ckpt,
    load_cm2_mask_model,
)
from model.navi import NaviForMaskedLM
from transformers import BertTokenizer
from utils.navi_checkpoints import resolve_navi_epoch_dir

try:
    _CHECKPOINT_EPOCH = getattr(app_config, "CHECKPOINT_EPOCH", 2)
except Exception:
    _CHECKPOINT_EPOCH = 2

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ── helpers ─────────────────────────────────────────────────────────────────

def _find_navi_path(domain_lower: str) -> str | None:
    base = str(_REPO_ROOT / "models" / f"navi_{domain_lower}")
    p = resolve_navi_epoch_dir(base, _CHECKPOINT_EPOCH)
    return str(p) if p else None


def _load_tokenizer() -> BertTokenizer:
    if app_config:
        return BertTokenizer.from_pretrained(
            app_config.BERT_NAME, local_files_only=app_config.use_local_files_only()
        )
    return BertTokenizer.from_pretrained("bert-base-uncased")


def stratified_downsample(
    scores: np.ndarray, target_n: int, n_bins: int = 50, seed: int = 42
) -> np.ndarray:
    """Sample `target_n` items from `scores` preserving its distribution shape."""
    if len(scores) <= target_n:
        return scores
    bin_edges = np.linspace(scores.min(), scores.max() + 1e-9, n_bins + 1)
    bin_ids = np.clip(np.digitize(scores, bin_edges) - 1, 0, n_bins - 1)
    rng = np.random.default_rng(seed)
    sampled: list[int] = []
    for b in range(n_bins):
        in_bin = np.where(bin_ids == b)[0]
        if len(in_bin) == 0:
            continue
        k = max(1, int(round(len(in_bin) / len(scores) * target_n)))
        k = min(k, len(in_bin))
        sampled.extend(rng.choice(in_bin, k, replace=False).tolist())
    return scores[np.array(sampled)]


def _collect_cm2_cat_scores(
    rows: list[dict],
    model: Any,
    table_flag: int = 0,
    mlm_probability: float = 0.35,
    base_seed: int = 42,
) -> list[float]:
    """Return categorical cosine scores from CM2 (same loop as evaluate_cm2_unified_semantic_scores)."""
    model.eval()
    cat_scores: list[float] = []

    for row_idx, row in enumerate(rows):
        df = json_row_to_cm2_dataframe(row)
        seed = base_seed + row_idx * 1009 + 31  # same as cm2_mp_utils cat seed

        with torch.no_grad():
            inputs = model.input_encoder.feature_extractor(df, table_flag=table_flag)
            outputs, other_info = model.input_encoder.feature_processor(**inputs)
            num_cnt = other_info["num_cnt"]
            col_emb = other_info["col_emb"]
            cat_cnt = col_emb.shape[0] - num_cnt

            if cat_cnt == 0:
                continue

            mi = _sample_override_mask(
                outputs["attention_mask"],
                num_cnt,
                cat_cnt,
                "cat",
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

            cat_proj = aux.get("cat_proj")
            cat_mask = aux.get("cat_mask")
            cat_tgt  = aux.get("cat_bert_emb")
            if cat_proj is None or cat_mask is None or cat_tgt is None:
                continue
            if not cat_mask.bool().any():
                continue
            cos = F.cosine_similarity(
                cat_proj[cat_mask.bool()], cat_tgt[cat_mask.bool()], dim=-1
            )
            cat_scores.extend([float(x) for x in cos.detach().cpu().tolist()])

    return cat_scores


# ── plotting ─────────────────────────────────────────────────────────────────

def _plot_histogram(
    navi_scores: np.ndarray,
    cm2_scores:  np.ndarray,
    domain: str,
    out_path: Path,
    bins: int = 50,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))

    finite_navi = navi_scores[np.isfinite(navi_scores)]
    finite_cm2  = cm2_scores[np.isfinite(cm2_scores)]

    if len(finite_navi):
        ax.hist(finite_navi, bins=bins, alpha=0.55, label="NAVI")
    if len(finite_cm2):
        ax.hist(finite_cm2,  bins=bins, alpha=0.55, label="CM2")

    ax.set_title(f"{domain}", fontsize=13)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_xlabel("Cosine similarity", fontsize=11)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_ylabel("Frequency", fontsize=11)
    if len(finite_navi) or len(finite_cm2):
        ax.legend(fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"✅ Plot saved to: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def run(domain: str, run_id: str, output_dir: Path) -> None:
    domain_lower = domain.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ──
    data_path = DATA_ROOT / "cleaned" / domain / "test" / f"WDC_{domain_lower}_for_mp.jsonl"
    raw_json = load_data(str(data_path))
    random.seed(42)
    if len(raw_json) > 1000:
        raw_json = random.sample(raw_json, 1000)
    print(f"[{domain}] {len(raw_json)} rows loaded")

    json_data = [(i, row) for i, row in enumerate(raw_json)]
    navi_dataset = NaviDataset(json_data)

    # ── NAVI feature-level scores ──
    tokenizer = _load_tokenizer()
    navi_path = _find_navi_path(domain_lower)
    if navi_path is None:
        print(f"⚠️  NAVI model not found for {domain}, skipping.")
        return
    navi_model = NaviForMaskedLM(navi_path).to(device).eval()
    print(f"[{domain}] Computing NAVI feature-level scores...")
    navi_scores, navi_diag = collect_navi_feature_unified_semantic_scores(
        navi_dataset, navi_model, tokenizer
    )
    print(f"  {navi_diag}")
    del navi_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # ── CM2 categorical scores ──
    if domain == "Movie":
        cm2_ckpt = os.environ.get("CM2_MOVIE_CHECKPOINT") or default_cm2_movie_ckpt()
    else:
        cm2_ckpt = os.environ.get("CM2_PRODUCT_CHECKPOINT") or default_cm2_product_ckpt()
    print(f"[{domain}] Loading CM2 from: {cm2_ckpt}")
    cm2_model = load_cm2_mask_model(cm2_ckpt, device)
    print(f"[{domain}] Computing CM2 categorical scores...")
    cm2_scores = _collect_cm2_cat_scores(raw_json, cm2_model, table_flag=0)
    del cm2_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    navi_arr = np.array([s for s in navi_scores if np.isfinite(s)], dtype=np.float32)
    cm2_arr  = np.array([s for s in cm2_scores  if np.isfinite(s)], dtype=np.float32)
    print(f"[{domain}] NAVI: {len(navi_arr)} finite scores, CM2: {len(cm2_arr)} finite scores")

    # Stratified downsample NAVI to match CM2 count
    if len(cm2_arr) > 0 and len(navi_arr) > len(cm2_arr):
        navi_plot = stratified_downsample(navi_arr, target_n=len(cm2_arr))
        print(f"[{domain}] NAVI downsampled: {len(navi_arr)} → {len(navi_plot)}")
    else:
        navi_plot = navi_arr

    plot_path = output_dir / f"imputation_plot_{domain_lower}_{run_id}.png"
    _plot_histogram(navi_plot, cm2_arr, domain, plot_path)

    # Save raw scores JSON
    scores_path = output_dir / f"imputation_scores_{domain_lower}_{run_id}.json"
    scores_path.write_text(
        json.dumps(
            {
                "domain": domain,
                "run_id": run_id,
                "navi_scores": navi_arr.tolist(),
                "cm2_scores":  cm2_arr.tolist(),
                "navi_diagnostics": navi_diag,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"✅ Scores saved to: {scores_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NAVI vs CM2 imputation cosine similarity plot.")
    parser.add_argument("--domain", choices=["Movie", "Product"], required=True)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=_REPO_ROOT / "experiments" / "navi_cm2_results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from datetime import datetime
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run(args.domain, run_id, args.output_dir)


if __name__ == "__main__":
    main()
