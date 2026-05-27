#!/bin/bash
# navi_cm2.sh — NAVI vs CM2 comparison (imputation plot + CM2 row classification)
#
# Usage:
#   conda activate CM2
#   ./navi_cm2.sh [RUN_ID]
#
# Skip stages:
#   SKIP_PLOT=1 ./navi_cm2.sh       # skip imputation plot
#   SKIP_CM2_CLS=1 ./navi_cm2.sh    # skip CM2 row classification

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "CM2" ]]; then
  echo "[ERROR] This pipeline must run from the CM2 conda environment." >&2
  echo "        Run: conda activate CM2" >&2
  exit 2
fi

CM2_PYTHON="${CONDA_PREFIX}/bin/python"
if [[ ! -x "$CM2_PYTHON" ]]; then
  echo "[ERROR] Could not find CM2 python at ${CM2_PYTHON}" >&2
  exit 2
fi

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_TOKEN="${HF_TOKEN:-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}"
export TABPFN_MODEL_VERSION="${TABPFN_MODEL_VERSION:-v2}"
export TABPFN_DISABLE_TELEMETRY="${TABPFN_DISABLE_TELEMETRY:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RUN_ID="${1:-$(date +%Y%m%d_%H%M%S)}"
DOMAINS=("Movie" "Product")
OUT_DIR="experiments/navi_cm2_results"

log() { echo "[$(date +'%H:%M:%S')] $*"; }

mkdir -p "$OUT_DIR"
mkdir -p experiments/logs

echo "============================================================"
echo "  NAVI vs CM2  (run_id=${RUN_ID})"
echo "============================================================"
echo ""

# ── Imputation Plot: NAVI feature-level vs CM2 cosine similarity ──────────────

if [[ "${SKIP_PLOT:-0}" != "1" ]]; then
  log "=== Imputation Plot ==="
  for DOMAIN in "${DOMAINS[@]}"; do
    log "  ${DOMAIN}"
    "$CM2_PYTHON" experiments/navi_cm2_imputation_plot.py \
      --domain "$DOMAIN" \
      --run_id "${RUN_ID}" \
      --output_dir "$OUT_DIR"
  done
fi

# ── CM2 Row Classification (8 runs, XGBoost / LR / TabPFN) ───────────────────

if [[ "${SKIP_CM2_CLS:-0}" != "1" ]]; then
  log "=== CM2 Row Classification ==="
  for DOMAIN in "${DOMAINS[@]}"; do
    log "  ${DOMAIN}"
    "$CM2_PYTHON" experiments/downstream_tasks/row_classification.py \
      --mode cm2 \
      --domain "$DOMAIN" \
      --embedding_type cls \
      --run_id "${RUN_ID}"
  done

  log "=== Generating CM2 classification table ==="
  "$CM2_PYTHON" experiments/generate_paper_tables.py \
    --tables cm2_cls \
    --run_id "${RUN_ID}" \
    --output_dir "$OUT_DIR"
fi

echo ""
echo "============================================================"
echo "  Done.  Results in ${OUT_DIR}/"
echo "============================================================"
