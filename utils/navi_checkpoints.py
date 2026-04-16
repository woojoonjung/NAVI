"""Discover NAVI checkpoints under ./models for a domain and resolve epoch directories."""

from __future__ import annotations

import glob
import os
from pathlib import Path

from utils.paths import project_root


def resolve_navi_epoch_dir(base_dir: str | Path, epoch: int) -> Path | None:
    """Resolve *epoch_{epoch} or epoch_{epoch} under base_dir (longest match wins)."""
    base = Path(base_dir)
    pattern = str(base / f"*epoch_{epoch}")
    matches = glob.glob(pattern)
    if matches:
        matches.sort(key=len, reverse=True)
        return Path(matches[0])
    direct = base / f"epoch_{epoch}"
    if direct.is_dir():
        return direct
    return None


def resolve_navi_checkpoint_path_with_fallback(
    model_name: str,
    base_dir: str | Path,
    epoch: int,
) -> Path | None:
    """
    Resolve training checkpoint directory for a NAVI run folder.

    Tries standard epoch_* layout first, then explicit ablation layout used by train_navi.
    """
    base = Path(base_dir)
    resolved = resolve_navi_epoch_dir(base, epoch)
    if resolved is not None:
        return resolved
    parts = model_name.split("_")
    if len(parts) >= 3:
        ablation = parts[-1]
        if ablation in ("woSSI", "woMSM", "woESA", "woGHA", "woGHC"):
            cand = base / f"{ablation}_HVB_seed42_cleaned_tau0.02_0.14_percentile_epoch_{epoch}"
            if cand.is_dir():
                return cand
    return None


def discover_navi_domain_model_dirs(
    domain: str,
    models_root: str | Path | None = None,
) -> list[tuple[str, Path]]:
    """
    List (model_name, base_dir) for navi_{domain_lower} and each navi_{domain_lower}_*
    under models_root. Sorted by model_name.
    """
    domain_lower = domain.lower()
    root = Path(models_root) if models_root is not None else project_root() / "models"
    if not root.is_absolute():
        root = project_root() / root

    out: list[tuple[str, Path]] = []
    base_name = f"navi_{domain_lower}"
    base_dir = root / base_name
    if base_dir.is_dir():
        out.append((base_name, base_dir.resolve()))

    pattern = str(root / f"{base_name}_*")
    for path_str in sorted(glob.glob(pattern)):
        p = Path(path_str)
        if p.is_dir():
            out.append((p.name, p.resolve()))

    return sorted(out, key=lambda x: x[0])


def navi_ablation_mode_for_model_name(model_name: str) -> str:
    """Architecture ablation flags only; dataset/training ablations load as full."""
    if model_name.endswith("_woGHA"):
        return "woGHA"
    if model_name.endswith("_woGHC"):
        return "woGHC"
    return "full"


def iter_chunks(items: list, chunk_size: int):
    """Yield successive chunks from items (last chunk may be shorter)."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]
