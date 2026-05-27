#!/usr/bin/env python
"""Generate plain-text paper tables from JSON logs and clustering CSVs.

Tables written to experiments/paper_tables/:
  table1_masked_prediction.txt
  table2_header_clustering.txt
  table3_row_classification.txt

CM2 classification table written to --output_dir:
  table_cm2_cls_{run_id}.txt

Usage:
  python experiments/generate_paper_tables.py --tables 1 2 3 --run_id my_run
  python experiments/generate_paper_tables.py --tables cm2_cls --run_id my_run --output_dir experiments/navi_cm2_results
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

LOGS_DIR   = _REPO_ROOT / "experiments" / "logs"
TABLES_DIR = _REPO_ROOT / "experiments" / "paper_tables"
ARTIFACTS  = _REPO_ROOT / "artifacts" / "lexvar"

DOMAINS    = ["Movie", "Product"]
CLS_MODELS = ["xgboost", "lr", "tabpfn"]


# ── JSON log helpers ─────────────────────────────────────────────────────────

def _find_log(pattern_prefix: str, run_id: str | None) -> Path | None:
    """Return the best-matching log file. run_id match preferred; else latest."""
    candidates = sorted(LOGS_DIR.glob(f"{pattern_prefix}*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None
    if run_id:
        exact = [p for p in candidates if run_id in p.name]
        if exact:
            return exact[-1]
    return candidates[-1]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# ── Table 1 ─────────────────────────────────────────────────────────────────

_TABLE1_MODEL_ORDER = [
    ("bert",         "BERT"),
    ("tapas",        "TAPAS"),
    ("haetae",       "HAETAE"),
    ("navi",         "NAVI"),
    ("navi_woMSM",   "NAVI-woMSM"),
    ("navi_woESA",   "NAVI-woESA"),
    ("navi_woGAE",   "NAVI-woGAE"),
]
_CELL_W = 19


def _fmt(mean, std) -> str:
    if mean is None:
        return "N/A".center(_CELL_W)
    return f"{mean:.4f}±{std:.4f}".center(_CELL_W)


def _agg_key(suffix: str, domain_lower: str) -> str:
    """Map _TABLE1_MODEL_ORDER suffix → actual key in the JSON log results dict."""
    if "_" in suffix:
        head, tail = suffix.split("_", 1)
        return f"{head}_{domain_lower}_{tail}"
    return f"{suffix}_{domain_lower}"


def _build_table1(run_id: str | None) -> str:
    log = _find_log("masked_prediction_table1_", run_id)
    if log is None:
        return "[Table 1] No masked_prediction_table1_*.json found in experiments/logs/\n"
    data = _load_json(log)
    results = data.get("results", {})

    cols = []
    for d in DOMAINS:
        cols.append(f"{d} Header".center(_CELL_W))
        cols.append(f"{d} Value".center(_CELL_W))
    header = f"{'Model':<20} | " + " | ".join(cols)
    sep = "-" * len(header)
    lines = [header, sep]

    for suffix, label in _TABLE1_MODEL_ORDER:
        row_parts: list[str] = []
        for domain in DOMAINS:
            dl = domain.lower()
            entry = results.get(domain, {}).get(_agg_key(suffix, dl))
            if entry is None:
                row_parts += ["N/A".center(_CELL_W), "N/A".center(_CELL_W)]
            else:
                row_parts += [
                    _fmt(entry.get("header_mean"), entry.get("header_std")),
                    _fmt(entry.get("value_mean"),  entry.get("value_std")),
                ]
        lines.append(f"{label:<20} | " + " | ".join(row_parts))

    return "\n".join(lines) + "\n"


# ── Table 2 ─────────────────────────────────────────────────────────────────

_CLUSTER_MODELS = ["bert", "tapas", "haetae", "navi"]
_CLUSTER_COLS   = ["B3_F1", "NMI", "Silhouette", "mean_intra_L2", "mean_intra_cosine_dist"]
_COL2_W = 12


def _build_table2() -> str:
    lines: list[str] = []
    for domain in DOMAINS:
        clean_domain = f"cleaned_{domain}"
        lines.append(f"\n=== {domain} ===")
        header_cols = [c.center(_COL2_W) for c in _CLUSTER_COLS]
        header = f"{'Model':<12} | " + " | ".join(header_cols)
        sep = "-" * len(header)
        lines += [header, sep]
        for model in _CLUSTER_MODELS:
            csv_path = ARTIFACTS / f"results_clustering_{model}_{clean_domain}.csv"
            if not csv_path.exists():
                lines.append(f"{model:<12} | " + " | ".join("N/A".center(_COL2_W) for _ in _CLUSTER_COLS))
                continue
            df = pd.read_csv(csv_path)
            if df.empty:
                lines.append(f"{model:<12} | " + " | ".join("N/A".center(_COL2_W) for _ in _CLUSTER_COLS))
                continue
            row = df.iloc[0]
            cells: list[str] = []
            for col in _CLUSTER_COLS:
                val = row.get(col) if col in row.index else None
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    cells.append("N/A".center(_COL2_W))
                else:
                    cells.append(f"{float(val):.4f}".center(_COL2_W))
            lines.append(f"{model:<12} | " + " | ".join(cells))
    return "\n".join(lines) + "\n"


# ── Table 3 ─────────────────────────────────────────────────────────────────

_TABLE3_LM_ORDER = [
    ("bert",       "BERT"),
    ("tapas",      "TAPAS"),
    ("haetae",     "HAETAE"),
    ("navi",       "NAVI"),
]
_TABLE3_ABL_ORDER = [
    ("navi_woSSI", "NAVI-woSSI"),
    ("navi_woMSM", "NAVI-woMSM"),
    ("navi_woESA", "NAVI-woESA"),
    ("navi_woGHA", "NAVI-woGHA"),
    ("navi_woGHC", "NAVI-woGHC"),
]

_T3_CELL = 18


def _cls_cell(entry: dict | None, ml: str, domain_lower: str) -> str:
    if entry is None:
        return "N/A".center(_T3_CELL)
    result = entry.get(f"{ml}_{domain_lower}")
    if result is None:
        return "N/A".center(_T3_CELL)
    mean = result.get("mean")
    std  = result.get("std")
    if mean is None:
        return "FAIL".center(_T3_CELL)
    return f"{mean:.4f}±{std:.4f}".center(_T3_CELL)


def _build_table3(run_id: str | None) -> str:
    # Load logs for each domain/mode
    lm_data: dict[str, Any] = {}
    abl_data: dict[str, Any] = {}
    for domain in DOMAINS:
        dl = domain.lower()
        log_lm = _find_log(f"row_classification_lm_encoders_{dl}_", run_id)
        if log_lm:
            lm_data[domain] = _load_json(log_lm)
        log_abl = _find_log(f"row_classification_ablations_{dl}_", run_id)
        if log_abl:
            abl_data[domain] = _load_json(log_abl)

    # Column header: one column per (domain, classifier)
    col_headers: list[str] = []
    for domain in DOMAINS:
        for ml in CLS_MODELS:
            col_headers.append(f"{domain[:3]}-{ml}".center(_T3_CELL))
    header = f"{'Model':<20} | " + " | ".join(col_headers)
    sep = "-" * len(header)
    lines = [header, sep]

    def _row(label: str, model_key: str, source: dict) -> str:
        cells: list[str] = []
        for domain in DOMAINS:
            dl = domain.lower()
            domain_data = source.get(domain, {})
            entry = domain_data.get(model_key)
            for ml in CLS_MODELS:
                cells.append(_cls_cell(entry, ml, dl))
        return f"{label:<20} | " + " | ".join(cells)

    lines.append("--- LM Encoders ---")
    for key, label in _TABLE3_LM_ORDER:
        lines.append(_row(label, key, lm_data))

    lines.append("--- Ablations ---")
    for key, label in _TABLE3_ABL_ORDER:
        lines.append(_row(label, key, abl_data))

    return "\n".join(lines) + "\n"


# ── CM2 classification table ─────────────────────────────────────────────────

def _build_cm2_cls(run_id: str | None) -> str:
    col_headers: list[str] = []
    for domain in DOMAINS:
        for ml in CLS_MODELS:
            col_headers.append(f"{domain[:3]}-{ml}".center(_T3_CELL))
    header = f"{'Model':<12} | " + " | ".join(col_headers)
    sep = "-" * len(header)
    lines = ["=== CM2 Row Classification ===", header, sep]

    cells: list[str] = []
    for domain in DOMAINS:
        dl = domain.lower()
        log = _find_log(f"row_classification_cm2_{dl}_", run_id)
        data = _load_json(log) if log else {}
        entry = data.get("cm2")
        for ml in CLS_MODELS:
            cells.append(_cls_cell(entry, ml, dl))

    lines.append(f"{'CM2':<12} | " + " | ".join(cells))
    return "\n".join(lines) + "\n"


# ── entry point ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper tables from experiment logs.")
    parser.add_argument(
        "--tables",
        nargs="+",
        default=["1", "2", "3"],
        help="Which tables to generate: 1 2 3 cm2_cls (or 'all')",
    )
    parser.add_argument("--run_id", type=str, default=None, help="Shared run_id for log matching")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for cm2_cls table (default: experiments/navi_cm2_results/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    tables = args.tables
    if "all" in tables:
        tables = ["1", "2", "3"]

    for table in tables:
        if table == "1":
            content = _build_table1(args.run_id)
            out = TABLES_DIR / "table1_masked_prediction.txt"
            out.write_text(content, encoding="utf-8")
            print(f"✅ Table 1 written to: {out}")

        elif table == "2":
            content = _build_table2()
            out = TABLES_DIR / "table2_header_clustering.txt"
            out.write_text(content, encoding="utf-8")
            print(f"✅ Table 2 written to: {out}")

        elif table == "3":
            content = _build_table3(args.run_id)
            out = TABLES_DIR / "table3_row_classification.txt"
            out.write_text(content, encoding="utf-8")
            print(f"✅ Table 3 written to: {out}")

        elif table == "cm2_cls":
            run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = args.output_dir or (_REPO_ROOT / "experiments" / "navi_cm2_results")
            out_dir.mkdir(parents=True, exist_ok=True)
            content = _build_cm2_cls(args.run_id)
            out = out_dir / f"table_cm2_cls_{run_id}.txt"
            out.write_text(content, encoding="utf-8")
            print(f"✅ CM2 cls table written to: {out}")

        else:
            print(f"⚠️  Unknown table: {table}")


if __name__ == "__main__":
    main()
