#!/usr/bin/env python3
"""
ICML ESA routing entropy ablation sweep.

Evaluates 4 entropy routing variants:
  navi_{domain}_esaRouting{variant} for domain in {Movie, Product}

Reports:
  Imp: masked value prediction accuracy (epoch=5) as mean/std over multiple runs
  Cls: row classification macro-F1 (XGBoost) as mean/std over multiple runs
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from config import config as app_config

from utils.navi_checkpoints import resolve_navi_epoch_dir

from model.navi import NaviForMaskedLM
from dataset.dataset import NaviDataset
from dataset.collator import CollatorForMaskedPrediction

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


DATA_ROOT = "data/"


def _set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_tokenizer():
    use_local = app_config.use_local_files_only()
    from transformers import BertTokenizer

    return BertTokenizer.from_pretrained(
        app_config.get_bert_name(),
        local_files_only=use_local,
    )


def _get_collator(tokenizer):
    return CollatorForMaskedPrediction(tokenizer=tokenizer)


def _get_cls_dataset(domain: str, max_samples: int):
    """
    Returns:
      cls_data: list of dict rows
      target_col: 'genres' (Movie) or 'category' (Product)
    """
    domain_lower = domain.lower()
    cls_path = (
        Path(DATA_ROOT)
        / "cleaned"
        / domain
        / "test"
        / f"WDC_{domain_lower}_for_cls.jsonl"
    )
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


def _get_mp_dataset(domain: str, max_samples: int):
    domain_lower = domain.lower()
    mp_path = (
        Path(DATA_ROOT)
        / "cleaned"
        / domain
        / "test"
        / f"WDC_{domain_lower}_for_mp.jsonl"
    )
    raw = load_mp_data(str(mp_path))

    random.seed(42)
    if len(raw) > max_samples:
        raw = random.sample(raw, max_samples)
    json_data = [(i, row) for i, row in enumerate(raw)]
    return NaviDataset(json_data)


def _variant_map():
    # Table row label -> model folder suffix
    # Note: Default uses the base model folder `navi_{domain}` (not `navi_{domain}_esaRoutingentropy`)
    return [
        ("Default", "base"),
        ("AllLow", "all_low"),
        ("AllHigh", "all_high"),
        ("Random", "random"),
    ]


def _fmt_mean_std(mean, std) -> str:
    if mean is None or std is None:
        return "---"
    try:
        return f"{float(mean):.4f} +/- {float(std):.4f}"
    except (TypeError, ValueError):
        return "---"


def _write_table_txt(out_path: Path, table_rows):
    """
    table_rows: List of
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
        lines.append(
            f"{row_label}|{imp_m}|{cls_m}|{imp_p}|{cls_p}"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✅ ICML entropy ablation table written to: {out_path}")


def load_navi_model(domain: str, suffix: str, checkpoint_epoch: int, device: torch.device):
    domain_lower = domain.lower()
    if suffix == "base":
        base_dir = Path("./models") / f"navi_{domain_lower}"
        model_name = f"navi_{domain_lower}"
    else:
        base_dir = Path("./models") / f"navi_{domain_lower}_esaRouting{suffix}"
        model_name = f"navi_{domain_lower}_esaRouting{suffix}"
    epoch_dir = resolve_navi_epoch_dir(base_dir, checkpoint_epoch)
    if epoch_dir is None:
        raise FileNotFoundError(
            f"Missing checkpoint for {model_name} at epoch_{checkpoint_epoch}"
        )
    model = NaviForMaskedLM(epoch_dir).to(device)
    model.eval()
    return model


def run_repeated_masked_value_prediction(
    dataset,
    model,
    tokenizer,
    collator,
    epoch: int = 5,
    n_runs: int = 5,
    base_seed: int = 42,
):
    """Repeat masked prediction evaluation; masking randomness differs per run."""
    scores = []
    for run in range(n_runs):
        _set_all_seeds(base_seed + run)
        acc = evaluate_masked_prediction(
            dataset,
            model,
            tokenizer,
            collator,
            epoch=epoch,
        )
        scores.append(acc)
    arr = np.asarray(scores, dtype=np.float64)
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "scores": scores}


def run_icml_entropy_ablation(
    max_samples: int = 1000,
    imp_runs: int = 5,
    xgb_runs: int = 5,
    run_id: str | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _set_all_seeds(app_config.SEED if hasattr(app_config, "SEED") else 42)

    checkpoint_epoch = getattr(app_config, "CHECKPOINT_EPOCH", 2)

    tokenizer = _get_tokenizer()
    collator = _get_collator(tokenizer)

    # Load datasets once per domain to keep Imp/Cls consistent.
    domains = ["Movie", "Product"]
    mp_datasets = {d: _get_mp_dataset(d, max_samples=max_samples) for d in domains}
    cls_datasets = {d: _get_cls_dataset(d, max_samples=max_samples) for d in domains}

    rid = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        Path("experiments/icml_entropy_ablation")
        / f"icml_entropy_ablation_{rid}.txt"
    )

    table_rows = []
    for row_label, suffix in _variant_map():
        imp_m_mean = None
        imp_m_std = None
        cls_m_mean = None
        cls_m_std = None
        imp_p_mean = None
        imp_p_std = None
        cls_p_mean = None
        cls_p_std = None

        for domain in domains:
            domain_lower = domain.lower()
            model_name = (
                f"navi_{domain_lower}"
                if suffix == "base"
                else f"navi_{domain_lower}_esaRouting{suffix}"
            )
            print(f"\n[{domain}] {row_label}: loading {model_name}...")

            _set_all_seeds(app_config.SEED if hasattr(app_config, "SEED") else 42)
            model = load_navi_model(domain, suffix, checkpoint_epoch, device)
            try:
                # Imp (value prediction accuracy, epoch=5) => mean/std over runs
                imp_stats = run_repeated_masked_value_prediction(
                    mp_datasets[domain],
                    model,
                    tokenizer,
                    collator,
                    epoch=5,
                    n_runs=imp_runs,
                    base_seed=42,
                )

                # Cls (row classification F1, XGBoost) => mean/std over runs
                cls_data, target_col = cls_datasets[domain]
                cls_stats = run_repeated_classification(
                    dataset=cls_data,
                    target_col=target_col,
                    model=model,
                    model_name=model_name,
                    domain=domain_lower,
                    ml_model="xgboost",
                    n_runs=xgb_runs,
                    embedding_type="cls",
                )

            finally:
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

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
    p = argparse.ArgumentParser(description="ICML ESA routing entropy ablation sweep")
    p.add_argument("--run_id", type=str, default=None, help="Output table run id")
    p.add_argument("--max_samples", type=int, default=1000, help="Dataset sampling size per domain")
    p.add_argument("--imp_runs", type=int, default=5, help="Repeat masked-prediction (Imp) runs to report mean/std")
    p.add_argument("--xgb_runs", type=int, default=5, help="Repeat XGBoost runs to report mean/std")
    args = p.parse_args()

    run_icml_entropy_ablation(
        max_samples=args.max_samples,
        imp_runs=args.imp_runs,
        xgb_runs=args.xgb_runs,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()

