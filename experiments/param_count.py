#!/usr/bin/env python3
"""
Print parameter counts for BERT, TAPAS, HAETAE, and NAVI checkpoints.

Paths and epoch follow config / experiments/downstream_tasks/row_classification.py.
Run from repository root:

  python experiments/param_count.py
  python experiments/param_count.py --domain product --models_root ./models_b200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


def _find_navi_epoch_dir(base_path: str, epoch: int, resolve) -> str | None:
    model_name = Path(base_path).name
    p = resolve(model_name, base_path, epoch)
    return str(p) if p else None


def count_params(model, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count parameters for BERT / TAPAS / HAETAE / NAVI.")
    parser.add_argument(
        "--domain",
        choices=("movie", "product"),
        default="movie",
        help="Which domain checkpoint to load (architecture is the same per model family).",
    )
    parser.add_argument(
        "--models_root",
        type=str,
        default="./models",
        help="Root containing bert_{domain}, tapas_{domain}, haetae_{domain}, navi_{domain}.",
    )
    parser.add_argument(
        "--use_b200",
        action="store_true",
        help="Use ./models_b200/ layout (same as load_b200_models in row_classification).",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Override CHECKPOINT_EPOCH from config (default: config.CHECKPOINT_EPOCH).",
    )
    parser.add_argument("--trainable-only", action="store_true", help="Also print trainable-only counts.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from transformers import BertConfig, BertForMaskedLM, BertTokenizer, TapasForMaskedLM

    from baselines.haetae.model import HAETAE
    from config import config as app_config
    from model.navi import NaviForMaskedLM
    from utils.navi_checkpoints import resolve_navi_checkpoint_path_with_fallback

    ep = args.epoch if args.epoch is not None else getattr(app_config, "CHECKPOINT_EPOCH", 2)
    root = Path("./models_b200" if args.use_b200 else args.models_root).resolve()
    dom = args.domain

    bert_name = app_config.get_bert_name()
    bert_kw = {"local_files_only": app_config.use_local_files_only()}
    bconfig = BertConfig.from_pretrained(bert_name, **bert_kw)
    tokenizer = BertTokenizer.from_pretrained(bert_name, **bert_kw)

    rows: list[tuple[str, str, int, int | None]] = []

    def add_row(label: str, source: str, model) -> None:
        tot = count_params(model, trainable_only=False)
        tr = count_params(model, trainable_only=True) if args.trainable_only else None
        rows.append((label, source, tot, tr))

    # BERT
    bert_dir = root / f"bert_{dom}" / f"epoch_{ep}"
    if bert_dir.is_dir():
        m = BertForMaskedLM.from_pretrained(str(bert_dir), local_files_only=True)
        m.eval()
        add_row("bert", str(bert_dir), m)
    else:
        print(f"[skip] BERT: not found {bert_dir}", file=sys.stderr)

    # TAPAS
    tap_dir = root / f"tapas_{dom}" / f"epoch_{ep}"
    if tap_dir.is_dir():
        m = TapasForMaskedLM.from_pretrained(str(tap_dir), local_files_only=True)
        m.eval()
        add_row("tapas", str(tap_dir), m)
    else:
        print(f"[skip] TAPAS: not found {tap_dir}", file=sys.stderr)

    # HAETAE
    hae_dir = root / f"haetae_{dom}" / f"epoch_{ep}"
    if hae_dir.is_dir():
        m = HAETAE(bconfig, tokenizer, str(hae_dir))
        m.eval()
        add_row("haetae", str(hae_dir), m)
    else:
        print(f"[skip] HAETAE: not found {hae_dir}", file=sys.stderr)

    # NAVI
    navi_base = root / f"navi_{dom}"
    navi_path = _find_navi_epoch_dir(str(navi_base), ep, resolve_navi_checkpoint_path_with_fallback)
    if navi_path:
        m = NaviForMaskedLM(navi_path)
        m.eval()
        add_row("navi", navi_path, m)
    else:
        print(f"[skip] NAVI: no epoch_{ep} under {navi_base}", file=sys.stderr)

    if not rows:
        print("No models loaded. Check --models_root / --use_b200 / checkpoints.", file=sys.stderr)
        sys.exit(1)

    print(f"models_root={root}  domain={dom}  epoch={ep}")
    if args.trainable_only:
        print(f"{'model':<8}  {'total':>14}  {'trainable':>14}  checkpoint")
        for label, src, tot, tr in rows:
            assert tr is not None
            print(f"{label:<8}  {tot:>14,}  {tr:>14,}  {src}")
    else:
        print(f"{'model':<8}  {'params':>14}  checkpoint")
        for label, src, tot, _ in rows:
            print(f"{label:<8}  {tot:>14,}  {src}")


if __name__ == "__main__":
    main()
