#!/usr/bin/env python3
"""
ICML baseline evaluation sweep (Imp + Cls).

Evaluates:
  {bert, tapas, haetae, navi}_{movie,product}

Reports:
  Imp: masked value prediction accuracy (epoch=5), mean/std over repeated runs
  Cls: row classification macro-F1 (XGBoost, cls embedding), mean/std over repeated runs
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from config import config as app_config

from dataset.dataset import NaviDataset, BertDataset, HaetaeDataset, TapasDataset
from dataset.collator import CollatorForMaskedPrediction
from model.navi import NaviForMaskedLM
from baselines.haetae.model import HAETAE
from transformers import BertTokenizer, AutoTokenizer, BertForMaskedLM, TapasForMaskedLM, BertConfig

from experiments.masked_prediction.mp_utils import evaluate_masked_prediction
from experiments.masked_prediction.mp_utils import load_data as load_mp_data

from experiments.downstream_tasks.cls_utils import load_data as load_cls_data
from experiments.downstream_tasks.row_classification import (
    preprocess_wdc_movie,
    preprocess_wdc_product,
    stratified_sample,
    verify_target_columns_present,
    run_repeated_classification,
)

from utils.navi_checkpoints import resolve_navi_epoch_dir


DATA_ROOT = "data/"


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fmt_mean_std(mean, std) -> str:
    if mean is None or std is None:
        return "---"
    try:
        return f"{float(mean):.4f} +/- {float(std):.4f}"
    except (TypeError, ValueError):
        return "---"


def _write_table_txt(out_path: Path, table_rows):
    """
    table_rows:
      (row_label,
       imp_movie_mean, imp_movie_std, cls_movie_mean, cls_movie_std,
       imp_product_mean, imp_product_std, cls_product_mean, cls_product_std)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header_1 = "            |    Movie   | Product"
    header_2 = "            | Imp | Cls | Imp | Cls"
    bar = "-" * 34

    lines = [header_1, header_2, bar]
    for (
        row_label,
        imp_m_mean,
        imp_m_std,
        cls_m_mean,
        cls_m_std,
        imp_p_mean,
        imp_p_std,
        cls_p_mean,
        cls_p_std,
    ) in table_rows:
        imp_m = _fmt_mean_std(imp_m_mean, imp_m_std)
        cls_m = _fmt_mean_std(cls_m_mean, cls_m_std)
        imp_p = _fmt_mean_std(imp_p_mean, imp_p_std)
        cls_p = _fmt_mean_std(cls_p_mean, cls_p_std)
        lines.append(f"{row_label}|{imp_m}|{cls_m}|{imp_p}|{cls_p}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✅ ICML baseline Imp/Cls table written to: {out_path}")


def _get_cls_dataset(domain: str, max_samples: int, cls_path: Path):
    raw = load_cls_data(str(cls_path))

    if domain == "Product":
        raw = verify_target_columns_present(raw, target_columns=["category"])
        raw = preprocess_wdc_product(raw)
        raw = stratified_sample(raw, "category", sample_size=max_samples)
        return raw, "category"

    raw = verify_target_columns_present(raw, target_columns=["genres"])
    raw = preprocess_wdc_movie(raw)
    raw = stratified_sample(raw, "genres", sample_size=max_samples)
    return raw, "genres"


def _get_mp_datasets(domain: str, max_samples: int, mp_path: Path):
    """
    Returns per-model masked-prediction datasets for one domain.
    """
    raw = load_mp_data(str(mp_path))

    random.seed(42)
    if len(raw) > max_samples:
        raw = random.sample(raw, max_samples)

    navi_data = [(i, row) for i, row in enumerate(raw)]
    flat = [row for _, row in navi_data]
    return {
        "navi": NaviDataset(navi_data),
        "bert": BertDataset(flat, mode="masked_prediction"),
        "haetae": HaetaeDataset(flat),
        "tapas": TapasDataset(flat, mode="masked_prediction"),
    }


def _resolve_epoch_dir(base_dir: Path, epoch: int) -> Path | None:
    pattern = sorted(base_dir.glob(f"*epoch_{epoch}"), key=lambda p: len(str(p)), reverse=True)
    if pattern:
        return pattern[0]
    direct = base_dir / f"epoch_{epoch}"
    if direct.is_dir():
        return direct
    return None


def _load_model(model_type: str, domain: str, device: torch.device, bert_tokenizer):
    domain_lower = domain.lower()
    model_name = f"{model_type}_{domain_lower}_cross_table"
    base_dir = Path("./models") / model_name
    checkpoint_epoch = getattr(app_config, "CHECKPOINT_EPOCH", 2)

    if model_type == "navi":
        epoch_dir = resolve_navi_epoch_dir(base_dir, checkpoint_epoch)
        if epoch_dir is None:
            raise FileNotFoundError(f"Missing checkpoint for {model_name} at epoch_{checkpoint_epoch}")
        model = NaviForMaskedLM(epoch_dir).to(device).eval()
        return model, model_name

    epoch_dir = _resolve_epoch_dir(base_dir, checkpoint_epoch)
    if epoch_dir is None:
        raise FileNotFoundError(f"Missing checkpoint for {model_name} at epoch_{checkpoint_epoch}")

    use_local = app_config.use_local_files_only()
    if model_type == "bert":
        model = BertForMaskedLM.from_pretrained(str(epoch_dir), local_files_only=use_local).to(device).eval()
    elif model_type == "tapas":
        model = TapasForMaskedLM.from_pretrained(str(epoch_dir), local_files_only=use_local).to(device).eval()
    elif model_type == "haetae":
        bert_name = app_config.get_bert_name()
        bconfig = BertConfig.from_pretrained(bert_name, local_files_only=use_local)
        model = HAETAE(bconfig, bert_tokenizer, str(epoch_dir)).to(device).eval()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return model, model_name


def _run_repeated_masked_value_prediction(
    dataset,
    model,
    tokenizer,
    collator,
    n_runs: int = 5,
    base_seed: int = 42,
):
    scores = []
    for run in range(n_runs):
        _set_all_seeds(base_seed + run)
        acc = evaluate_masked_prediction(
            dataset,
            model,
            tokenizer,
            collator,
            epoch=5,
        )
        scores.append(acc)
    arr = np.asarray(scores, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "scores": scores}


def run_icml_baseline_imp_cls(
    max_samples: int = 1000,
    imp_runs: int = 5,
    xgb_runs: int = 5,
    run_id: str | None = None,
    movie_cls_path: Path | None = None,
    movie_mp_path: Path | None = None,
    product_cls_path: Path | None = None,
    product_mp_path: Path | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_all_seeds(getattr(app_config, "SEED", 42))

    use_local = app_config.use_local_files_only()
    bert_name = app_config.get_bert_name()
    tapas_name = app_config.TAPAS_NAME
    bert_tokenizer = BertTokenizer.from_pretrained(bert_name, local_files_only=use_local)
    tapas_tokenizer = AutoTokenizer.from_pretrained(tapas_name, local_files_only=use_local)
    bert_collator = CollatorForMaskedPrediction(tokenizer=bert_tokenizer)
    tapas_collator = CollatorForMaskedPrediction(tokenizer=tapas_tokenizer)

    domains = ["Movie", "Product"]

    resolved_paths = {
        "Movie": {
            "mp": movie_mp_path
            or Path(DATA_ROOT) / "cleaned" / "Movie" / "test" / "WDC_movie_for_mp.jsonl",
            "cls": movie_cls_path
            or Path(DATA_ROOT) / "cleaned" / "Movie" / "test" / "WDC_movie_for_cls.jsonl",
        },
        "Product": {
            "mp": product_mp_path
            or Path(DATA_ROOT) / "cleaned" / "Product" / "test" / "WDC_product_for_mp.jsonl",
            "cls": product_cls_path
            or Path(DATA_ROOT) / "cleaned" / "Product" / "test" / "WDC_product_for_cls.jsonl",
        },
    }

    for domain in domains:
        if not resolved_paths[domain]["mp"].exists():
            raise FileNotFoundError(f"Missing MP file for {domain}: {resolved_paths[domain]['mp']}")
        if not resolved_paths[domain]["cls"].exists():
            raise FileNotFoundError(f"Missing CLS file for {domain}: {resolved_paths[domain]['cls']}")
        print(f"[{domain}] MP path:  {resolved_paths[domain]['mp']}")
        print(f"[{domain}] CLS path: {resolved_paths[domain]['cls']}")

    mp_datasets = {
        d: _get_mp_datasets(d, max_samples=max_samples, mp_path=resolved_paths[d]["mp"])
        for d in domains
    }
    cls_datasets = {
        d: _get_cls_dataset(d, max_samples=max_samples, cls_path=resolved_paths[d]["cls"])
        for d in domains
    }

    rid = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path("experiments/icml_baselines_imp_cls") / f"icml_baselines_imp_cls_{rid}.txt"

    model_types = [("BERT", "bert"), ("TAPAS", "tapas"), ("HAETAE", "haetae"), ("NAVI", "navi")]
    table_rows = []

    for row_label, model_type in model_types:
        imp_m_mean = imp_m_std = cls_m_mean = cls_m_std = None
        imp_p_mean = imp_p_std = cls_p_mean = cls_p_std = None

        for domain in domains:
            print(f"\n[{domain}] {row_label}: loading {model_type}_{domain.lower()}...")
            model = None
            try:
                model, model_name = _load_model(model_type, domain, device, bert_tokenizer)
                tokenizer = bert_tokenizer if model_type in ("bert", "haetae", "navi") else tapas_tokenizer
                collator = bert_collator if model_type in ("bert", "haetae", "navi") else tapas_collator

                imp_stats = _run_repeated_masked_value_prediction(
                    mp_datasets[domain][model_type],
                    model,
                    tokenizer,
                    collator,
                    n_runs=imp_runs,
                    base_seed=42,
                )

                cls_data, target_col = cls_datasets[domain]
                cls_stats = run_repeated_classification(
                    dataset=cls_data,
                    target_col=target_col,
                    model=model,
                    model_name=model_name,
                    domain=domain.lower(),
                    ml_model="xgboost",
                    n_runs=xgb_runs,
                    embedding_type="cls",
                )

                if domain == "Movie":
                    imp_m_mean = imp_stats.get("mean")
                    imp_m_std = imp_stats.get("std")
                    cls_m_mean = cls_stats.get("mean")
                    cls_m_std = cls_stats.get("std")
                else:
                    imp_p_mean = imp_stats.get("mean")
                    imp_p_std = imp_stats.get("std")
                    cls_p_mean = cls_stats.get("mean")
                    cls_p_std = cls_stats.get("std")

            except Exception as e:
                print(f"⚠️ Failed {model_type}_{domain.lower()}: {e}")
            finally:
                if model is not None:
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        table_rows.append(
            (
                row_label,
                imp_m_mean,
                imp_m_std,
                cls_m_mean,
                cls_m_std,
                imp_p_mean,
                imp_p_std,
                cls_p_mean,
                cls_p_std,
            )
        )

    _write_table_txt(out_path, table_rows)
    return out_path


def main():
    p = argparse.ArgumentParser(description="ICML baseline Imp/Cls sweep")
    p.add_argument("--run_id", type=str, default=None, help="Output table run id")
    p.add_argument("--max_samples", type=int, default=1000, help="Dataset sampling size per domain")
    p.add_argument("--imp_runs", type=int, default=5, help="Repeat masked-prediction runs")
    p.add_argument("--xgb_runs", type=int, default=5, help="Repeat XGBoost runs")
    p.add_argument("--movie_cls_path", type=Path, default=None, help="Path to WDC_movie_for_cls.jsonl")
    p.add_argument("--movie_mp_path", type=Path, default=None, help="Path to WDC_movie_for_mp.jsonl")
    p.add_argument("--product_cls_path", type=Path, default=None, help="Path to WDC_product_for_cls.jsonl")
    p.add_argument("--product_mp_path", type=Path, default=None, help="Path to WDC_product_for_mp.jsonl")
    args = p.parse_args()

    run_icml_baseline_imp_cls(
        max_samples=args.max_samples,
        imp_runs=args.imp_runs,
        xgb_runs=args.xgb_runs,
        run_id=args.run_id,
        movie_cls_path=args.movie_cls_path,
        movie_mp_path=args.movie_mp_path,
        product_cls_path=args.product_cls_path,
        product_mp_path=args.product_mp_path,
    )


if __name__ == "__main__":
    main()

