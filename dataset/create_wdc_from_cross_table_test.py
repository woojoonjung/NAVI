#!/usr/bin/env python3
"""
Create WDC_{domain}_for_mp.jsonl and WDC_{domain}_for_cls.jsonl from cross-table test files.

This script is standalone (does not run full preprocess pipeline) and is intended for
rebuttal-time experiments that use:
  data/cleaned/<Domain>/cross_table/test
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from preprocess import (
    load_jsonl,
    save_jsonl,
    unify_genre_keys,
    unify_category_keys,
)


def _normalize_domain(domain: str) -> str:
    d = domain.strip().lower()
    if d not in {"movie", "product"}:
        raise ValueError(f"Unsupported domain: {domain}. Use Movie or Product.")
    return d


def _collect_rows_from_test_tables(test_dir: Path) -> List[Dict]:
    files = sorted(list(test_dir.glob("*.json")) + list(test_dir.glob("*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No .json/.jsonl files found in: {test_dir}")

    rows: List[Dict] = []
    for fp in files:
        data = load_jsonl(str(fp))
        rows.extend(data)
    return rows


def _build_cls_rows(domain: str, all_rows: List[Dict]) -> List[Dict]:
    if domain == "movie":
        cls_rows = [unify_genre_keys(row) for row in all_rows]
        cls_rows = [
            row
            for row in cls_rows
            if "genres" in row and row["genres"] and row["genres"] != "None"
        ]
        return cls_rows

    cls_rows = [unify_category_keys(row) for row in all_rows]
    cls_rows = [
        row
        for row in cls_rows
        if "category" in row and row["category"] and row["category"] != "None"
    ]
    return cls_rows


def create_wdc_from_cross_table_test(
    domain: str,
    test_dir: Path,
    output_dir: Path | None = None,
    force: bool = False,
) -> tuple[Path, Path]:
    d = _normalize_domain(domain)
    out_dir = output_dir if output_dir is not None else test_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mp_path = out_dir / f"WDC_{d}_for_mp.jsonl"
    cls_path = out_dir / f"WDC_{d}_for_cls.jsonl"

    if not force and mp_path.exists() and cls_path.exists():
        print(f"Both output files already exist. Use --force to overwrite.")
        print(f"  {mp_path}")
        print(f"  {cls_path}")
        return mp_path, cls_path

    all_rows = _collect_rows_from_test_tables(test_dir)
    cls_rows = _build_cls_rows(d, all_rows)

    save_jsonl(all_rows, str(mp_path))
    save_jsonl(cls_rows, str(cls_path))

    print(f"Domain: {d}")
    print(f"Source test dir: {test_dir}")
    print(f"Tables loaded rows (MP): {len(all_rows)}")
    print(f"Filtered rows (CLS): {len(cls_rows)}")
    print(f"Saved MP:  {mp_path}")
    print(f"Saved CLS: {cls_path}")
    return mp_path, cls_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create WDC_{domain}_for_mp/cls.jsonl from cross-table test files"
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain name: Movie or Product",
    )
    parser.add_argument(
        "--test_dir",
        type=Path,
        default=None,
        help=(
            "Directory containing cross-table test table files "
            "(default: data/cleaned/<Domain>/cross_table/test)"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to write WDC outputs (default: same as --test_dir)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing WDC files if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d = _normalize_domain(args.domain)

    if args.test_dir is None:
        repo_root = Path(__file__).resolve().parents[1]
        test_dir = repo_root / "data" / "cleaned" / d.capitalize() / "cross_table" / "test"
    else:
        test_dir = args.test_dir

    create_wdc_from_cross_table_test(
        domain=d,
        test_dir=test_dir,
        output_dir=args.output_dir,
        force=args.force,
    )


if __name__ == "__main__":
    main()

