#!/usr/bin/env python3
"""
Evaluate a finetuned CM2Classifier with sklearn F1.

The finetune trainer only saves ``pytorch_model.bin`` (no ``extractor/``). Load the same
pretrain checkpoint used in ``run_finetune.py`` (--cpt), then overlay finetuned weights.

**Checkpoint vs fold:** ``run_finetune.py`` reuses the same ``--output_dir`` for every fold
and overwrites it each time, so on disk you only have weights from the **last** completed
fold (typically fold 5 when ``n_splits=5``). Use ``--fold`` that matches that run (default 5).

Split options:
  * ``fixed`` — evaluate directly on ``--eval_data`` (recommended for strict heldout test).
  * ``cv`` — same StratifiedKFold (n=5, random_state=42) as ``run_finetune.py``; pick
    ``--fold`` (1–5) for that fold's held-out indices.
  * ``random`` — stratified holdout (for a quick sanity check).

Usage (from baselines/CM2 with PYTHONPATH=.):

  python eval_finetune_f1.py \\
    --pretrain ./mask_v1_product_unsup \\
    --finetune ./models/product_finetune \\
    --task_data ./data/cleaned/Product/test/WDC_product_for_cls.csv \\
    --target category --fold 5
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from CM2.dataset_openml import load_single_data_all

import CM2
from CM2 import constants


def parse_args():
    p = argparse.ArgumentParser(description="F1 evaluation for CM2 finetuned classifier")
    p.add_argument(
        "--finetune",
        type=str,
        required=True,
        help="Directory containing finetuned pytorch_model.bin",
    )
    p.add_argument(
        "--pretrain",
        type=str,
        default=None,
        help="Pretrain dir (with extractor/) if finetune has weights only. Must match run_finetune --cpt.",
    )
    p.add_argument("--task_data", type=str, required=True)
    p.add_argument(
        "--eval_data",
        type=str,
        default=None,
        help="Explicit evaluation CSV used when split_mode=fixed.",
    )
    p.add_argument("--target", type=str, required=True)
    p.add_argument(
        "--split_mode",
        type=str,
        choices=("fixed", "cv", "random"),
        default="cv",
        help="fixed: evaluate on eval_data; cv: StratifiedKFold; random: stratified holdout",
    )
    p.add_argument(
        "--fold",
        type=int,
        default=5,
        help="1-based fold when split_mode=cv. Default 5: run_finetune overwrites output_dir each fold, so weights usually match the last fold.",
    )
    p.add_argument("--n_splits", type=int, default=5, help="K for StratifiedKFold")
    p.add_argument("--test_size", type=float, default=0.2, help="Used when split_mode=random")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--table_flag", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--num_layer", type=int, default=3)
    return p.parse_args()


def load_finetuned_model(
    finetune_dir: str,
    pretrain_dir: str | None,
    num_class: int,
    device: str,
    num_layer: int,
) -> CM2.CM2Classifier:
    finetune_dir = os.path.abspath(finetune_dir)
    has_extractor = os.path.isdir(os.path.join(finetune_dir, constants.EXTRACTOR_STATE_DIR))

    if has_extractor:
        base_ckpt = finetune_dir
    else:
        if not pretrain_dir:
            raise ValueError(
                f"{finetune_dir} has no {constants.EXTRACTOR_STATE_DIR}/; "
                "pass --pretrain (same as run_finetune --cpt)."
            )
        base_ckpt = os.path.abspath(pretrain_dir)

    model = CM2.build_classifier(
        checkpoint=base_ckpt,
        device=device,
        num_class=num_class,
        num_layer=num_layer,
        hidden_dropout_prob=0.1,
        vocab_freeze=True,
        use_bert=True,
    )

    if not has_extractor:
        weights_path = os.path.join(finetune_dir, constants.WEIGHTS_NAME)
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Missing {weights_path}")
        state = torch.load(weights_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[load_state_dict] missing keys ({len(missing)}): {missing[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"[load_state_dict] unexpected keys ({len(unexpected)}): {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")

    return model


def main():
    args = parse_args()
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    X, y, cat_cols, num_cols, bin_cols = load_single_data_all(args.task_data, target=args.target)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    num_class = int(len(np.unique(y)))

    if args.split_mode == "fixed":
        if not args.eval_data:
            raise ValueError("--eval_data is required when --split_mode fixed")
        X_test, y_test, _, _, _ = load_single_data_all(args.eval_data, target=args.target)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
    elif args.split_mode == "cv":
        if not 1 <= args.fold <= args.n_splits:
            raise ValueError("--fold must be in [1, n_splits]")
        skf = StratifiedKFold(n_splits=args.n_splits, random_state=42, shuffle=True)
        splits = list(skf.split(X, y))
        _, val_idx = splits[args.fold - 1]
        X_test = X.loc[val_idx]
        y_test = y.loc[val_idx]
    else:
        try:
            _, X_test, _, y_test = train_test_split(
                X,
                y,
                test_size=args.test_size,
                random_state=args.seed,
                stratify=y,
                shuffle=True,
            )
        except ValueError:
            _, X_test, _, y_test = train_test_split(
                X,
                y,
                test_size=args.test_size,
                random_state=args.seed,
                shuffle=True,
            )

    model = load_finetuned_model(
        args.finetune,
        args.pretrain,
        num_class=num_class,
        device=dev,
        num_layer=args.num_layer,
    )
    cat_cols = [cat_cols]
    num_cols = [num_cols]
    bin_cols = [bin_cols]
    model.update({"cat": cat_cols, "num": num_cols, "bin": bin_cols})

    ypred = CM2.predict(model, X_test, table_flag=args.table_flag)

    if ypred.shape[-1] == 1:
        y_pred = (ypred >= 0.5).astype(np.int64).ravel()
    else:
        y_pred = np.argmax(ypred, axis=-1)

    y_true = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)

    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    print(f"finetune: {args.finetune}")
    if args.pretrain:
        print(f"pretrain: {args.pretrain}")
    print(f"task_data: {args.task_data}  target={args.target}  num_class={num_class}")
    if args.split_mode == "fixed":
        print(f"split_mode=fixed eval_data={args.eval_data}")
    elif args.split_mode == "cv":
        print(f"split_mode=cv fold={args.fold}/{args.n_splits}")
    else:
        print(f"split_mode=random test_size={args.test_size}")
    print(f"n_test={len(y_test)}")
    print(f"F1 macro:    {macro:.6f}")
    print(f"F1 weighted: {weighted:.6f}")
    print(f"F1 micro:    {micro:.6f}")
    print("\nclassification_report:\n")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


if __name__ == "__main__":
    CM2.random_seed(42)
    main()
