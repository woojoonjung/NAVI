#!/usr/bin/env python3
"""
Create deterministic leakage-free train/val/test splits for CM2 finetuning.

Outputs (per domain):
  baselines/CM2/data/cleaned/{Domain}/strict/
    - WDC_{domain}_for_cls_finetune_train.csv
    - WDC_{domain}_for_cls_finetune_val.csv
    - WDC_{domain}_for_cls_heldout_test.csv
    - split_manifest.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[2]


def _label_distribution(df: pd.DataFrame, target: str) -> Dict[str, int]:
    counts = df[target].value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def _split_once(
    df: pd.DataFrame,
    target: str,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y = df[target]
    try:
        left, right = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=y,
            shuffle=True,
        )
    except ValueError:
        left, right = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
    return left, right


def build_domain_splits(domain: str, target: str, seed: int, test_size: float, val_size: float) -> None:
    domain_cap = domain.capitalize()
    domain_low = domain.lower()
    src = REPO_ROOT / "baselines" / "CM2" / "data" / "cleaned" / domain_cap / "test" / f"WDC_{domain_low}_for_cls.csv"
    if not src.is_file():
        raise FileNotFoundError(f"Missing source CSV: {src}")

    out_dir = REPO_ROOT / "baselines" / "CM2" / "data" / "cleaned" / domain_cap / "strict"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in {src}")
    df = df.dropna(subset=[target]).copy()
    df["_row_uid"] = pd.util.hash_pandas_object(df, index=False).astype(str)

    # 1) Holdout test split from full set.
    train_val_df, test_df = _split_once(df, target=target, test_size=test_size, seed=seed)

    # 2) Validation split from train_val.
    train_df, val_df = _split_once(train_val_df, target=target, test_size=val_size, seed=seed + 1)

    train_uids = set(train_df["_row_uid"].tolist())
    val_uids = set(val_df["_row_uid"].tolist())
    test_uids = set(test_df["_row_uid"].tolist())

    train_out = train_df.drop(columns=["_row_uid"]).reset_index(drop=True)
    val_out = val_df.drop(columns=["_row_uid"]).reset_index(drop=True)
    test_out = test_df.drop(columns=["_row_uid"]).reset_index(drop=True)

    train_path = out_dir / f"WDC_{domain_low}_for_cls_finetune_train.csv"
    val_path = out_dir / f"WDC_{domain_low}_for_cls_finetune_val.csv"
    test_path = out_dir / f"WDC_{domain_low}_for_cls_heldout_test.csv"

    train_out.to_csv(train_path, index=False)
    val_out.to_csv(val_path, index=False)
    test_out.to_csv(test_path, index=False)

    manifest = {
        "domain": domain_cap,
        "source_csv": str(src.relative_to(REPO_ROOT)),
        "seed": seed,
        "target": target,
        "test_size": test_size,
        "val_size": val_size,
        "rows": {
            "total": int(len(df)),
            "finetune_train": int(len(train_out)),
            "finetune_val": int(len(val_out)),
            "heldout_test": int(len(test_out)),
        },
        "label_distribution": {
            "finetune_train": _label_distribution(train_out, target),
            "finetune_val": _label_distribution(val_out, target),
            "heldout_test": _label_distribution(test_out, target),
        },
        "overlap_counts": {
            "train_val": int(len(train_uids & val_uids)),
            "train_test": int(len(train_uids & test_uids)),
            "val_test": int(len(val_uids & test_uids)),
        },
        "outputs": {
            "finetune_train_csv": str(train_path.relative_to(REPO_ROOT)),
            "finetune_val_csv": str(val_path.relative_to(REPO_ROOT)),
            "heldout_test_csv": str(test_path.relative_to(REPO_ROOT)),
        },
    }

    manifest_path = out_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[{domain_cap}] wrote split files to: {out_dir}")
    print(f"[{domain_cap}] manifest: {manifest_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare deterministic WDC splits for strict CM2 finetune eval")
    p.add_argument("--domain", choices=("Movie", "Product", "all"), default="all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--val_size", type=float, default=0.2, help="Validation size from train_val partition")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    domains = ["Movie", "Product"] if args.domain == "all" else [args.domain]
    target_by_domain = {"Movie": "genres", "Product": "category"}

    for d in domains:
        build_domain_splits(
            domain=d,
            target=target_by_domain[d],
            seed=args.seed,
            test_size=args.test_size,
            val_size=args.val_size,
        )


if __name__ == "__main__":
    main()
