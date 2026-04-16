#!/usr/bin/env python3
"""
Create deterministic cross-table splits from cleaned top100-style table files.

This script does NOT modify existing row-based splits. It creates a separate
directory layout:

  data/cleaned/<Domain>/cross_table/train
  data/cleaned/<Domain>/cross_table/validation
  data/cleaned/<Domain>/cross_table/test

Default policies:
  - Movie:   80 train / 10 validation / rest test
  - Product: 80 train / 10 validation / 10 test
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass(frozen=True)
class SplitSpec:
    domain: str
    source_subdir: str
    train_count: int
    val_count: int
    test_count: int | None = None


def _list_table_files(source_dir: Path) -> List[Path]:
    files = sorted(source_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No .json files found under: {source_dir}")
    return files


def _compute_partition(files: Sequence[Path], spec: SplitSpec, seed: int) -> tuple[List[Path], List[Path], List[Path]]:
    file_count = len(files)
    train_count = spec.train_count
    val_count = spec.val_count
    test_count = spec.test_count if spec.test_count is not None else file_count - train_count - val_count

    if train_count < 0 or val_count < 0 or test_count < 0:
        raise ValueError(f"Negative split size for domain={spec.domain}")
    if train_count + val_count + test_count != file_count:
        raise ValueError(
            f"Split mismatch for {spec.domain}: train+val+test={train_count+val_count+test_count}, files={file_count}"
        )

    shuffled = list(files)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    train_files = shuffled[:train_count]
    val_files = shuffled[train_count : train_count + val_count]
    test_files = shuffled[train_count + val_count :]
    return train_files, val_files, test_files


def _copy_split(files: Sequence[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, out_dir / src.name)


def _write_manifest(
    out_root: Path,
    spec: SplitSpec,
    seed: int,
    train_files: Sequence[Path],
    val_files: Sequence[Path],
    test_files: Sequence[Path],
) -> None:
    manifest = {
        "domain": spec.domain,
        "seed": seed,
        "source_subdir": spec.source_subdir,
        "counts": {
            "train": len(train_files),
            "validation": len(val_files),
            "test": len(test_files),
            "total": len(train_files) + len(val_files) + len(test_files),
        },
        "train_files": [p.name for p in train_files],
        "validation_files": [p.name for p in val_files],
        "test_files": [p.name for p in test_files],
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def create_cross_table_split(root: Path, spec: SplitSpec, seed: int, force: bool) -> None:
    source_dir = root / "data" / "cleaned" / spec.source_subdir
    files = _list_table_files(source_dir)

    out_root = root / "data" / "cleaned" / spec.domain / "cross_table"
    train_dir = out_root / "train"
    val_dir = out_root / "validation"
    test_dir = out_root / "test"

    if force and out_root.exists():
        shutil.rmtree(out_root)

    train_files, val_files, test_files = _compute_partition(files=files, spec=spec, seed=seed)
    _copy_split(train_files, train_dir)
    _copy_split(val_files, val_dir)
    _copy_split(test_files, test_dir)
    _write_manifest(out_root, spec, seed, train_files, val_files, test_files)

    print(
        f"[{spec.domain}] source={spec.source_subdir} total={len(files)} "
        f"train={len(train_files)} val={len(val_files)} test={len(test_files)}"
    )
    print(f"[{spec.domain}] output={out_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create cross-table splits for Movie/Product cleaned datasets.")
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root directory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic table shuffling.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing cross_table outputs before writing.",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default="Movie,Product",
        help="Comma-separated domains to generate: Movie, Product, or both.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = {d.strip() for d in args.domains.split(",") if d.strip()}

    split_specs = [
        SplitSpec(domain="Movie", source_subdir="Movie_top100", train_count=80, val_count=10, test_count=None),
        SplitSpec(domain="Product", source_subdir="Product_top100", train_count=80, val_count=10, test_count=10),
    ]

    for spec in split_specs:
        if spec.domain in selected:
            create_cross_table_split(args.root_dir, spec, seed=args.seed, force=args.force)


if __name__ == "__main__":
    main()
