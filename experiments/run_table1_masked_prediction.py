#!/usr/bin/env python
"""Table 1 — Masked Prediction driver for ICML reproducibility.

Runs token-level masked prediction (header epoch=1, value epoch=5) for
baselines (BERT, TAPAS, HAETAE, NAVI) and ablations (woSSI, woMSM, woESA,
woGHA, woGHC) over N randomised runs, aggregates mean±std, then writes a
multi-column .txt table and a JSON log.

Usage (from repo root, conda activate navi):
    python experiments/run_table1_masked_prediction.py \
        --domains Movie Product --n_runs 8 --run_id my_run
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from config import config as app_config
    _CHECKPOINT_EPOCH = getattr(app_config, "CHECKPOINT_EPOCH", 2)
except ImportError:
    app_config = None
    _CHECKPOINT_EPOCH = 2

DATA_ROOT = _REPO_ROOT / "data"

from dataset.dataset import NaviDataset, BertDataset, HaetaeDataset, TapasDataset
from dataset.collator import CollatorForMaskedPrediction
from model.navi import NaviForMaskedLM
from baselines.haetae.model import HAETAE
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer, TapasForMaskedLM, BertConfig
from experiments.masked_prediction.mp_utils import evaluate_masked_prediction, load_data
from utils.navi_checkpoints import resolve_navi_epoch_dir


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# ── checkpoint helpers ──────────────────────────────────────────────────────

def _find_epoch_path(base_path: str) -> str | None:
    p = resolve_navi_epoch_dir(base_path, _CHECKPOINT_EPOCH)
    return str(p) if p else None


def _navi_ablation_mode(ablation: str) -> str:
    return ablation if ablation in ("woGHA", "woGHC") else "full"


# ── model loaders ───────────────────────────────────────────────────────────

def _load_tokenizers():
    if app_config:
        bert_tok = BertTokenizer.from_pretrained(
            app_config.BERT_NAME, local_files_only=app_config.use_local_files_only()
        )
        tapas_tok = AutoTokenizer.from_pretrained(
            app_config.TAPAS_NAME, local_files_only=app_config.use_local_files_only()
        )
    else:
        bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
        tapas_tok = AutoTokenizer.from_pretrained("google/tapas-base-masklm")
    return bert_tok, tapas_tok


def _load_baselines(bert_tok, domain_lower: str) -> dict:
    if app_config:
        _bert_name = app_config.BERT_NAME
        _bert_kw = {"local_files_only": True}
    else:
        _bert_name = "bert-base-uncased"
        _bert_kw = {}
    bconfig = BertConfig.from_pretrained(_bert_name, **_bert_kw)
    ep = _CHECKPOINT_EPOCH
    models = {}

    for arch, key, cls, kw in [
        ("bert",  f"bert_{domain_lower}",  BertForMaskedLM,  {"local_files_only": True}),
        ("tapas", f"tapas_{domain_lower}", TapasForMaskedLM, {"local_files_only": True}),
    ]:
        path = str(_REPO_ROOT / "models" / f"{arch}_{domain_lower}" / f"epoch_{ep}")
        try:
            models[key] = cls.from_pretrained(path, **kw).to(device).eval()
        except Exception as e:
            print(f"⚠️  Could not load {key}: {e}")

    for key in (f"haetae_{domain_lower}",):
        path = str(_REPO_ROOT / "models" / key / f"epoch_{ep}")
        try:
            models[key] = HAETAE(bconfig, bert_tok, path).to(device).eval()
        except Exception as e:
            print(f"⚠️  Could not load {key}: {e}")

    navi_base = str(_REPO_ROOT / "models" / f"navi_{domain_lower}")
    navi_path = _find_epoch_path(navi_base)
    if navi_path:
        models[f"navi_{domain_lower}"] = NaviForMaskedLM(navi_path).to(device).eval()
    else:
        print(f"⚠️  NAVI not found: {navi_base}")

    return models


def _load_ablations(domain_lower: str) -> dict:
    models = {}
    for ablation in ("woSSI", "woMSM", "woESA", "woGHA", "woGHC"):
        base = str(_REPO_ROOT / "models" / f"navi_{domain_lower}_{ablation}")
        path = _find_epoch_path(base)
        if not path:
            explicit = (
                _REPO_ROOT / "models"
                / f"navi_{domain_lower}_{ablation}"
                / f"{ablation}_HVB_seed42_cleaned_tau0.02_0.14_percentile_epoch_{_CHECKPOINT_EPOCH}"
            )
            path = str(explicit) if explicit.exists() else None
        if path:
            mode = _navi_ablation_mode(ablation)
            models[f"navi_{domain_lower}_{ablation}"] = (
                NaviForMaskedLM(path, ablation_mode=mode).to(device).eval()
            )
        else:
            print(f"⚠️  Ablation not found: {base}")
    return models


def _clear():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ── per-run evaluation ──────────────────────────────────────────────────────

def _eval_model(model, datasets: dict, tokenizers, collators) -> dict[str, float]:
    """Return {header_acc, value_acc} for one model on one domain."""
    bert_tok, tapas_tok = tokenizers
    bert_col, tapas_col = collators

    from transformers import TapasForMaskedLM as _Tapas
    from baselines.haetae.model import HAETAE as _H

    # NaviForMaskedLM subclasses BertForMaskedLM — check NAVI first.
    if isinstance(model, NaviForMaskedLM):
        ds, tok, col = datasets["navi"], bert_tok, bert_col
    elif isinstance(model, _Tapas):
        ds, tok, col = datasets["tapas"], tapas_tok, tapas_col
    elif isinstance(model, _H):
        ds, tok, col = datasets["haetae"], bert_tok, bert_col
    elif isinstance(model, BertForMaskedLM):
        ds, tok, col = datasets["bert"], bert_tok, bert_col
    else:
        raise TypeError(f"Unsupported model type for masked prediction: {type(model)}")

    print("  Header:")
    h_acc = evaluate_masked_prediction(ds, model, tok, col, epoch=1)
    print("  Value:")
    v_acc = evaluate_masked_prediction(ds, model, tok, col, epoch=5)
    return {"header_accuracy": float(h_acc), "value_accuracy": float(v_acc)}


# ── multi-run driver ────────────────────────────────────────────────────────

def run_domain(domain: str, n_runs: int) -> dict[str, Any]:
    domain_lower = domain.lower()
    data_path = DATA_ROOT / "cleaned" / domain / "test" / f"WDC_{domain_lower}_for_mp.jsonl"
    raw_json = load_data(str(data_path))

    bert_tok, tapas_tok = _load_tokenizers()
    tokenizers = (bert_tok, tapas_tok)

    # all_runs[model_name] = {"header": [run0, run1, ...], "value": [...]}
    all_runs: dict[str, dict[str, list[float]]] = {}

    for run_idx in range(n_runs):
        seed = 42 + run_idx
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Fresh sample of 1000 rows per run
        rows = raw_json
        if len(rows) > 1000:
            rows = random.sample(rows, 1000)

        json_data = [(i, row) for i, row in enumerate(rows)]
        flat = [row for _, row in json_data]

        datasets = {
            "navi":   NaviDataset(json_data),
            "bert":   BertDataset(flat, mode="masked_prediction"),
            "tapas":  TapasDataset(flat, mode="masked_prediction"),
            "haetae": HaetaeDataset(flat),
        }

        bert_col  = CollatorForMaskedPrediction(tokenizer=bert_tok)
        tapas_col = CollatorForMaskedPrediction(tokenizer=tapas_tok)
        collators = (bert_col, tapas_col)

        print(f"\n[{domain}] Run {run_idx + 1}/{n_runs} (seed={seed})")

        # ── baselines ──
        baseline_models = _load_baselines(bert_tok, domain_lower)
        for model_name, model in baseline_models.items():
            print(f"  {model_name}")
            res = _eval_model(model, datasets, tokenizers, collators)
            if model_name not in all_runs:
                all_runs[model_name] = {"header": [], "value": []}
            all_runs[model_name]["header"].append(res["header_accuracy"])
            all_runs[model_name]["value"].append(res["value_accuracy"])
        del baseline_models
        _clear()

        # ── ablations ──
        ablation_models = _load_ablations(domain_lower)
        for model_name, model in ablation_models.items():
            print(f"  {model_name}")
            res = _eval_model(model, datasets, tokenizers, collators)
            if model_name not in all_runs:
                all_runs[model_name] = {"header": [], "value": []}
            all_runs[model_name]["header"].append(res["header_accuracy"])
            all_runs[model_name]["value"].append(res["value_accuracy"])
        del ablation_models
        _clear()

    # aggregate
    agg: dict[str, Any] = {}
    for model_name, runs in all_runs.items():
        h = runs["header"]
        v = runs["value"]
        agg[model_name] = {
            "header_mean": float(np.mean(h)) if h else None,
            "header_std":  float(np.std(h))  if h else None,
            "value_mean":  float(np.mean(v)) if v else None,
            "value_std":   float(np.std(v))  if v else None,
            "header_runs": h,
            "value_runs":  v,
        }
    return agg


# ── table formatting ────────────────────────────────────────────────────────

def _fmt(mean, std) -> str:
    if mean is None:
        return "   N/A   "
    return f"{mean:.4f}±{std:.4f}"


_MODEL_ORDER = [
    ("bert",    "BERT"),
    ("tapas",   "TAPAS"),
    ("haetae",  "HAETAE"),
    ("navi",    "NAVI"),
    ("navi_woSSI", "NAVI-woSSI"),
    ("navi_woMSM", "NAVI-woMSM"),
    ("navi_woESA", "NAVI-woESA"),
    ("navi_woGHA", "NAVI-woGHA"),
    ("navi_woGHC", "NAVI-woGHC"),
]

COL_W = 19  # width of each metric cell


def _agg_key(suffix: str, domain_lower: str) -> str:
    """Map _MODEL_ORDER suffix → actual key in the aggregated results dict.

    Keys stored by run_domain follow the model_name from _load_baselines /
    _load_ablations:
      bert_movie, tapas_movie, haetae_movie, navi_movie,
      navi_movie_woSSI, navi_movie_woMSM, ...

    The suffix either has no underscore ("bert") or one underscore separating
    the architecture name from the ablation tag ("navi_woSSI").  Insert the
    domain between the two parts in the latter case.
    """
    if "_" in suffix:
        head, tail = suffix.split("_", 1)
        return f"{head}_{domain_lower}_{tail}"
    return f"{suffix}_{domain_lower}"


def _write_table(results: dict[str, dict], domains: list[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cols = []
    for d in domains:
        cols.append(f"{d} Header".center(COL_W))
        cols.append(f"{d} Value".center(COL_W))
    header = f"{'Model':<20} | " + " | ".join(cols)
    sep = "-" * len(header)
    lines = [header, sep]

    for suffix, label in _MODEL_ORDER:
        row_parts = []
        for domain in domains:
            domain_lower = domain.lower()
            entry = results.get(domain, {}).get(_agg_key(suffix, domain_lower))
            if entry is None:
                row_parts.append("   N/A   ".center(COL_W))
                row_parts.append("   N/A   ".center(COL_W))
            else:
                row_parts.append(_fmt(entry["header_mean"], entry["header_std"]).center(COL_W))
                row_parts.append(_fmt(entry["value_mean"],  entry["value_std"]).center(COL_W))
        lines.append(f"{label:<20} | " + " | ".join(row_parts))

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✅ Table 1 written to: {out_path}")


# ── entry point ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Table 1: repeated masked-prediction evaluation.")
    parser.add_argument("--domains", nargs="+", default=["Movie", "Product"], choices=["Movie", "Product"])
    parser.add_argument("--n_runs", type=int, default=8)
    parser.add_argument("--run_id", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results: dict[str, Any] = {}
    for domain in args.domains:
        print(f"\n{'='*60}")
        print(f"  Domain: {domain}  ({args.n_runs} runs)")
        print(f"{'='*60}")
        all_results[domain] = run_domain(domain, args.n_runs)

    # Write JSON log
    log_path = _REPO_ROOT / "experiments" / "logs" / f"masked_prediction_table1_{run_id}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"run_id": run_id, "n_runs": args.n_runs, "results": all_results}
    log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n✅ JSON log saved to: {log_path}")

    # Write text table
    out_txt = _REPO_ROOT / "experiments" / "paper_tables" / "table1_masked_prediction.txt"
    _write_table(all_results, args.domains, out_txt)


if __name__ == "__main__":
    main()
