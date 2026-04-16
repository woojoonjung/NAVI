#!/usr/bin/env bash
# Run experiments/downstream_tasks/row_classification.py --mode cm2 for every
# baselines/CM2/models/{movie|product}_finetune/epoch_* checkpoint.
#
# Usage (from repository root):
#   bash experiments/downstream_tasks/run_cm2_finetune_epoch_sweep.sh [movie|product|all]
#
# Optional env:
#   EXTRA_ROW_CLS_ARGS  — extra args passed to row_classification.py (quoted string)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SCOPE="${1:-all}"

run_domain() {
  local domain_key="$1"   # movie | product
  local domain_title="$2" # Movie | Product
  local finetune_dir="$REPO_ROOT/baselines/CM2/models/${domain_key}_finetune"

  if [[ ! -d "$finetune_dir" ]]; then
    echo "Skip: missing directory $finetune_dir" >&2
    return 0
  fi

  shopt -s nullglob
  local epochs=( "$finetune_dir"/epoch_* )
  shopt -u nullglob

  if [[ ${#epochs[@]} -eq 0 ]]; then
    echo "No epoch_* checkpoints under $finetune_dir" >&2
    return 0
  fi

  mapfile -t epochs < <(printf '%s\n' "${epochs[@]}" | sort -V)

  for ckpt in "${epochs[@]}"; do
    [[ -d "$ckpt" ]] || continue
    ep_name="$(basename "$ckpt")"
    run_id="cm2_${domain_key}_${ep_name}"
    echo "========================================"
    echo "Domain: $domain_title  checkpoint: $ckpt"
    echo "========================================"
    # shellcheck disable=SC2086
    python experiments/downstream_tasks/row_classification.py \
      --mode cm2 \
      --domain "$domain_title" \
      --cm2_checkpoint "$ckpt" \
      --run_id "$run_id" \
      ${EXTRA_ROW_CLS_ARGS:-}
  done
}

case "$SCOPE" in
  movie)
    run_domain movie Movie
    ;;
  product)
    run_domain product Product
    ;;
  all)
    run_domain movie Movie
    run_domain product Product
    ;;
  *)
    echo "Usage: $0 [movie|product|all]" >&2
    exit 1
    ;;
esac

echo "Done CM2 finetune epoch sweep (scope=$SCOPE)."
