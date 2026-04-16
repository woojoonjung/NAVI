from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    # This file lives at <root>/utils/paths.py
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return project_root() / "data"

