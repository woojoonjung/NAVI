"""Shared summary statistics for unified_semantic (NAVI + CM2): min/max/mean and quartiles."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def summarize_scores(scores: Sequence[float]) -> Dict[str, float]:
    if not scores:
        return {k: float("nan") for k in ("min", "max", "avg", "Q1", "Q2", "Q3")}
    a = np.asarray(scores, dtype=np.float64)
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "avg": float(a.mean()),
        "Q1": float(np.quantile(a, 0.25)),
        "Q2": float(np.quantile(a, 0.50)),
        "Q3": float(np.quantile(a, 0.75)),
    }


def print_unified_semantic_block(model_name: str, split_label: str, summary: Dict[str, float], n: int) -> None:
    print(f"\n{model_name} — {split_label}  (n={n})")
    if n == 0:
        print("  (no samples)  min=max=avg=Q1=Q2=Q3=nan")
        return
    print(
        f"  min={summary['min']:.4f}  max={summary['max']:.4f}  avg={summary['avg']:.4f}  "
        f"Q1={summary['Q1']:.4f}  Q2={summary['Q2']:.4f}  Q3={summary['Q3']:.4f}"
    )
