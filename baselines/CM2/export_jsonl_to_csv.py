#!/usr/bin/env python3
"""
Export NAVI-style JSON Lines tables (one JSON object per line, flat header->value) to CSV
files for CM2 masked pretraining. Sources may use .jsonl or .json extensions (training data
often uses .json). Output mirrors data/cleaned/ under baselines/CM2/data/.

CM2 load_pretrain_data.check_data_quality rejects tables with any cell having >= 30 words;
we truncate long cells to 29 words.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

_TRAIN_TEST_EXTS = frozenset({".json", ".jsonl"})

# CM2 rejects cells where len(str(text).split()) >= 30
MAX_WORDS_PER_CELL = 29

NULL_LIKE = {"", "none", "null", "nan", "#n/a"}


def _truncate_words(text: Any, max_words: int = MAX_WORDS_PER_CELL) -> str:
    s = str(text).strip()
    words = s.split()
    if len(words) <= max_words:
        return s
    return " ".join(words[:max_words])


def _normalize_cell(v: Any) -> Any:
    if v is None:
        return pd.NA
    if isinstance(v, float) and pd.isna(v):
        return pd.NA
    s = str(v).strip()
    if s.lower() in NULL_LIKE:
        return pd.NA
    return s


def _cell_for_cm2(v: Any) -> Any:
    v = _normalize_cell(v)
    if v is pd.NA or pd.isna(v):
        return v
    return _truncate_words(v)


def jsonl_to_dataframe(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in df.columns:
        df[col] = df[col].map(_cell_for_cm2)
    return df


def export_file(src: Path, dst: Path) -> None:
    df = jsonl_to_dataframe(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False, encoding="utf-8")


def collect_source_files(cleaned: Path, mode: str) -> List[Path]:
    """
    mode:
      jsonl — every *.jsonl under cleaned (WDC test shards, etc.)
      train — cleaned/**/train/*.{json,jsonl}  (parent dir name is \"train\")
      test — cleaned/**/test/*.{json,jsonl}
    """
    if mode == "jsonl":
        return sorted(cleaned.rglob("*.jsonl"))
    want = "train" if mode == "train" else "test"
    out: List[Path] = []
    for p in cleaned.rglob("*"):
        if not p.is_file():
            continue
        if p.parent.name != want:
            continue
        if p.suffix.lower() not in _TRAIN_TEST_EXTS:
            continue
        out.append(p)
    return sorted(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export cleaned JSON Lines to CM2-ready CSV.")
    parser.add_argument(
        "--data_root",
        type=Path,
        default=None,
        help="Root containing cleaned/ (default: repo ../../data from this script).",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Output root (default: baselines/CM2/data next to this script).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="jsonl",
        choices=("jsonl", "train", "test"),
        help="jsonl: all *.jsonl under cleaned; train/test: **/{train,test}/*.{json,jsonl}",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_root = args.data_root or (script_dir.parent.parent / "data")
    out_root = args.out_dir or (script_dir / "data")

    cleaned = data_root / "cleaned"
    if not cleaned.is_dir():
        print(f"Missing cleaned data directory: {cleaned}", file=sys.stderr)
        sys.exit(1)

    jsonl_files = collect_source_files(cleaned, args.mode)
    if not jsonl_files:
        print(f"No source files for mode={args.mode!r} under {cleaned}", file=sys.stderr)
        sys.exit(1)

    for src in jsonl_files:
        rel = src.relative_to(cleaned)
        dst = out_root / "cleaned" / rel.with_suffix(".csv")
        print(f"{src} -> {dst}")
        export_file(src, dst)

    print(f"Done. Wrote {len(jsonl_files)} CSV file(s) under {out_root / 'cleaned'}")


if __name__ == "__main__":
    main()
