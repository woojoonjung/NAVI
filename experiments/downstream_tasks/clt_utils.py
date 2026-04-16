"""
Small clustering / geometry helpers shared across experiments (e.g. header clustering).
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np


def centroid_cosine_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    """1 - cosine similarity between two vectors; NaN if either has zero L2 norm."""
    c1 = np.asarray(c1, dtype=np.float64).ravel()
    c2 = np.asarray(c2, dtype=np.float64).ravel()
    n1 = float(np.linalg.norm(c1))
    n2 = float(np.linalg.norm(c2))
    if n1 == 0.0 or n2 == 0.0:
        return float(np.nan)
    cos_sim = float(np.dot(c1, c2) / (n1 * n2))
    cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
    return float(1.0 - cos_sim)


def mean_pairwise_centroid_l2(
    X: np.ndarray,
    y: np.ndarray,
    class_names: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    For rows in X with integer labels y, compute each group's centroid in the original
    embedding space, then the mean of pairwise L2 distances between distinct centroids.

    Args:
        X: (n_samples, dim) embedding matrix.
        y: (n_samples,) integer labels in 0 .. K-1 (contiguous for present groups).
        class_names: optional length-K array; name for label i used in pairwise keys.

    Returns:
        dict with keys:
          mean_l2: float, or np.nan if fewer than two groups with at least one point.
          n_groups: number of distinct labels with at least one sample.
          pairwise: dict mapping stable string keys to L2 distance for each pair
                    (only when class_names is provided and len(unique) >= 2).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    if y.shape[0] != X.shape[0]:
        raise ValueError("y length must match number of rows in X")

    unique = np.unique(y)
    centroids: dict[int, np.ndarray] = {}
    for lab in unique:
        mask = y == lab
        if not np.any(mask):
            continue
        centroids[int(lab)] = X[mask].mean(axis=0)

    n_groups = len(centroids)
    if n_groups < 2:
        return {"mean_l2": float(np.nan), "n_groups": n_groups, "pairwise": {}}

    labs = sorted(centroids.keys())
    dists: list[float] = []
    pairwise: dict[str, float] = {}

    for i, j in combinations(range(len(labs)), 2):
        li, lj = labs[i], labs[j]
        d = float(np.linalg.norm(centroids[li] - centroids[lj]))
        dists.append(d)
        if class_names is not None and len(class_names) > max(li, lj):
            n1 = str(class_names[li])
            n2 = str(class_names[lj])
            key = f"{n1}_vs_{n2}" if n1 <= n2 else f"{n2}_vs_{n1}"
            pairwise[key] = d

    return {
        "mean_l2": float(np.mean(dists)),
        "n_groups": n_groups,
        "pairwise": pairwise,
    }


def mean_pairwise_centroid_cosine_dist(
    X: np.ndarray,
    y: np.ndarray,
    class_names: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Same grouping as mean_pairwise_centroid_l2, but pairwise centroid **cosine distance**
    (1 - cosine similarity) in the original embedding space.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    if y.shape[0] != X.shape[0]:
        raise ValueError("y length must match number of rows in X")

    unique = np.unique(y)
    centroids: dict[int, np.ndarray] = {}
    for lab in unique:
        mask = y == lab
        if not np.any(mask):
            continue
        centroids[int(lab)] = X[mask].mean(axis=0)

    n_groups = len(centroids)
    if n_groups < 2:
        return {"mean_cosine_dist": float(np.nan), "n_groups": n_groups, "pairwise": {}}

    labs = sorted(centroids.keys())
    dists: list[float] = []
    pairwise: dict[str, float] = {}

    for i, j in combinations(range(len(labs)), 2):
        li, lj = labs[i], labs[j]
        d = centroid_cosine_distance(centroids[li], centroids[lj])
        dists.append(d)
        if class_names is not None and len(class_names) > max(li, lj):
            n1 = str(class_names[li])
            n2 = str(class_names[lj])
            key = f"{n1}_vs_{n2}" if n1 <= n2 else f"{n2}_vs_{n1}"
            pairwise[key] = d

    # Mean over finite pairwise values (skip nan pairs)
    arr = np.asarray(dists, dtype=np.float64)
    mean_cd = float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float(np.nan)

    return {
        "mean_cosine_dist": mean_cd,
        "n_groups": n_groups,
        "pairwise": pairwise,
    }
