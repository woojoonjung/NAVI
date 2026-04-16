"""
Helpers to load frozen CM2 (CM2ForMask) checkpoints and extract row-level embeddings
for downstream sklearn classifiers (same protocol as NAVI/BERT CLS in cls_utils).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_cm2_import() -> None:
    cm2_root = repo_root() / "baselines" / "CM2"
    if not cm2_root.is_dir():
        raise FileNotFoundError(f"CM2 baseline not found at {cm2_root}")
    p = str(cm2_root)
    if p not in sys.path:
        sys.path.insert(0, p)


def infer_pretrain_table_num(state_dict: dict) -> int:
    """Infer number of pretraining tables from saved num_projection_head.* keys."""
    idxs = []
    for k in state_dict:
        if k.startswith("num_projection_head.") and len(k.split(".")) > 1:
            part = k.split(".")[1]
            if part.isdigit():
                idxs.append(int(part))
    return max(idxs) + 1 if idxs else 1


def infer_num_classes(state_dict: dict) -> int:
    """Infer classifier output dimension from saved CM2Classifier weights."""
    for key in ("clf.fc.bias", "clf.fc.weight"):
        if key in state_dict:
            tensor = state_dict[key]
            return int(tensor.shape[0])
    raise KeyError("Could not infer num_class from checkpoint state_dict (missing clf.fc.*)")


def load_cm2_mask_model(checkpoint_dir: str | Path, device: torch.device | None = None) -> Any:
    """
    Load CM2ForMask from a directory containing pytorch_model.bin and extractor/.
    """
    ensure_cm2_import()
    import CM2  # noqa: WPS433 — local import after path fix

    checkpoint_dir = Path(checkpoint_dir)
    weights = checkpoint_dir / "pytorch_model.bin"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing {weights}")

    sd = torch.load(weights, map_location="cpu")
    n_tables = infer_pretrain_table_num(sd)
    placeholder = [[] for _ in range(n_tables)]

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CM2.build_mask_features_learner(
        categorical_columns=placeholder,
        numerical_columns=placeholder,
        binary_columns=placeholder,
        pretrain_table_num=n_tables,
        device=str(dev),
        checkpoint=str(checkpoint_dir),
    )
    model.eval()
    return model


def is_cm2_mask_model(model: Any) -> bool:
    # Works for both CM2ForMask and CM2Classifier checkpoints.
    return (
        hasattr(model, "input_encoder")
        and hasattr(model, "cls_token")
        and hasattr(model, "encoder")
    )


def load_cm2_classifier_model(
    checkpoint_dir: str | Path,
    device: torch.device | None = None,
    pretrain_dir: str | Path | None = None,
    num_layer: int = 3,
) -> Any:
    """
    Load CM2Classifier checkpoint for embedding extraction.

    Supports:
    - full checkpoint dirs containing extractor/ + pytorch_model.bin, or
    - weights-only dirs when pretrain_dir is provided.
    """
    ensure_cm2_import()
    import CM2  # noqa: WPS433

    checkpoint_dir = Path(checkpoint_dir)
    weights = checkpoint_dir / "pytorch_model.bin"
    if not weights.is_file():
        raise FileNotFoundError(f"Missing {weights}")

    sd = torch.load(weights, map_location="cpu")
    num_class = infer_num_classes(sd)
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    has_extractor = (checkpoint_dir / "extractor").is_dir()
    base_ckpt = checkpoint_dir if has_extractor else (Path(pretrain_dir) if pretrain_dir else None)
    if base_ckpt is None:
        raise ValueError(
            f"{checkpoint_dir} has no extractor/; pass pretrain_dir for weights-only CM2Classifier load."
        )

    model = CM2.build_classifier(
        checkpoint=str(base_ckpt),
        device=str(dev),
        num_class=num_class,
        num_layer=num_layer,
        hidden_dropout_prob=0.1,
        vocab_freeze=True,
        use_bert=True,
    )
    if not has_extractor:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[CM2 classifier load] missing keys ({len(missing)})")
        if unexpected:
            print(f"[CM2 classifier load] unexpected keys ({len(unexpected)})")
    model.eval()
    return model


def configure_cm2_classifier_schema(model: Any, schema_csv: str | Path, target: str) -> None:
    """
    Initialize CM2 classifier feature-extractor schema from a reference CSV.
    """
    ensure_cm2_import()
    from CM2.dataset_openml import load_single_data_all  # noqa: WPS433

    _, _, cat_cols, num_cols, bin_cols = load_single_data_all(str(schema_csv), target=target)
    model.update({"cat": [cat_cols], "num": [num_cols], "bin": [bin_cols]})


def json_row_to_cm2_dataframe(row_dict: Dict[str, Any]) -> pd.DataFrame:
    """Single-row DataFrame with lowercase string column names (CM2 convention)."""
    lower = {str(k).lower(): v for k, v in row_dict.items()}
    return pd.DataFrame([lower])


def get_cm2_row_embedding(
    model: Any,
    row_dict: Dict[str, Any],
    table_flag: int = 0,
    embedding_type: str = "cls",
) -> np.ndarray:
    """
    Encoder hidden state only (same as other baselines' CLS): feature_extractor →
    cls_token → transformer encoder. Does not apply mask heads or clf logits.
    - cls: first sequence position after encoder (row-level vector).
    - mean: mean-pool over sequence positions after encoder.
    """
    df = json_row_to_cm2_dataframe(row_dict)
    model.eval()
    with torch.no_grad():
        inputs = model.input_encoder.feature_extractor(df, table_flag=table_flag)
        outputs, _ = model.input_encoder.feature_processor(**inputs)
        outputs = model.cls_token(**outputs)
        encoder_output = model.encoder(**outputs)
        seq = encoder_output[0].float()
        if embedding_type == "mean":
            emb = seq.mean(dim=0).cpu().numpy()
        else:
            emb = seq[0, :].cpu().numpy()
    return emb


def default_cm2_movie_ckpt() -> str:
    return os.environ.get(
        "CM2_MOVIE_CHECKPOINT",
        str(repo_root() / "baselines" / "CM2" / "mask_v1_movie_unsup"),
    )


def default_cm2_product_ckpt() -> str:
    return os.environ.get(
        "CM2_PRODUCT_CHECKPOINT",
        str(repo_root() / "baselines" / "CM2" / "mask_v1_product_unsup"),
    )


def load_cm2_models_for_domain(
    domain: str,
    device: torch.device | None = None,
    movie_ckpt: str | None = None,
    product_ckpt: str | None = None,
) -> dict:
    """Return { 'cm2_movie': model } or { 'cm2_product': model } for the given domain."""
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = domain.lower()
    out: dict = {}
    if domain == "movie":
        ckpt = movie_ckpt or default_cm2_movie_ckpt()
        if Path(ckpt).is_dir() and (Path(ckpt) / "pytorch_model.bin").is_file():
            out["cm2_movie"] = load_cm2_mask_model(ckpt, dev)
            print(f"✓ Loaded CM2 Movie from: {ckpt}")
        else:
            print(f"⚠️  CM2 Movie checkpoint not found or incomplete: {ckpt}")
    elif domain == "product":
        ckpt = product_ckpt or default_cm2_product_ckpt()
        if Path(ckpt).is_dir() and (Path(ckpt) / "pytorch_model.bin").is_file():
            out["cm2_product"] = load_cm2_mask_model(ckpt, dev)
            print(f"✓ Loaded CM2 Product from: {ckpt}")
        else:
            print(f"⚠️  CM2 Product checkpoint not found or incomplete: {ckpt}")
    else:
        raise ValueError(f"Unknown domain: {domain}")
    return out
