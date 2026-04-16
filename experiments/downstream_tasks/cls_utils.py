"""
Utility functions for downstream task experiments (row classification).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
from io import StringIO
import torch
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score


def _lm_model_classes():
    """Lazy import so sklearn-only paths (e.g. CM2 row emb + XGB) avoid loading NAVI/transformers."""
    from model.navi import NaviForMaskedLM
    from baselines.haetae.model import HAETAE
    from transformers import TapasForMaskedLM, BertForMaskedLM

    return NaviForMaskedLM, HAETAE, TapasForMaskedLM, BertForMaskedLM


def load_data(path, path_is="jsonl"):
    """Load data from JSONL or CSV file."""
    data = []

    if path_is == "csv":
        if path is not None:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            table = pd.read_csv(StringIO(''.join(lines)))
            if "class" in table.columns:
                table.drop(columns=["class"], inplace=True)
            for _, row in table.iterrows():
                row_json = row.to_dict() 
                data.append(row_json)

    elif path_is == "jsonl":
        if path is not None:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))

    return data


# WDC Product classification: predefined "numeric" columns for text-numeric ablations.
# Union with any present feature column whose name ends with "price" (e.g. offers.price;
# excludes offers.pricecurrency).
PRODUCT_NUMERIC_EXPLICIT = frozenset(
    {
        "offers.lowprice",
        "offers.highprice",
        "aggregaterating.reviewcount",
        "aggregaterating.ratingvalue",
        "aggregaterating.bestrating",
        "aggregaterating.worstrating",
    }
)


def product_union_keys(rows: List[dict]) -> set:
    keys: set = set()
    for r in rows:
        keys.update(r.keys())
    return keys


def product_numeric_feature_columns(all_keys: Iterable[str], target_col: str) -> frozenset:
    """Explicit numeric keys plus any feature column ending with 'price'."""
    feature_keys = frozenset(k for k in all_keys if k != target_col)
    by_suffix = {k for k in feature_keys if k.endswith("price")}
    return frozenset((PRODUCT_NUMERIC_EXPLICIT & feature_keys) | by_suffix)


def product_column_ablation_info(rows: List[dict], target_col: str) -> Tuple[Dict[str, Any], frozenset, frozenset]:
    """
    Returns (log_info, numeric_set, half_kept_text_set) for Product ablation.
    log_info has keys: numeric_keys, text_keys, half_text_kept (sorted lists).
    """
    all_keys = product_union_keys(rows)
    numeric = product_numeric_feature_columns(all_keys, target_col)
    feature_keys = frozenset(k for k in all_keys if k != target_col)
    text_keys = feature_keys - numeric
    text_sorted = sorted(text_keys)
    n_keep = max(1, len(text_sorted) // 2)
    half_kept = frozenset(text_sorted[:n_keep])
    log_info = {
        "numeric_keys": sorted(numeric),
        "text_keys": text_sorted,
        "half_text_kept": sorted(half_kept),
    }
    return log_info, numeric, half_kept


def apply_product_column_ablation(
    rows: List[dict],
    target_col: str,
    mode: str,
) -> Tuple[List[dict], Dict[str, Any]]:
    """
    Product test-set column ablation. mode: 'allcol' | 'text_only' | 'half_text'.

    - allcol: no change.
    - text_only: drop all predefined numeric columns (text-only features + target).
    - half_text: keep predefined numeric columns + first half of sorted text columns.
    """
    if mode not in ("allcol", "text_only", "half_text"):
        raise ValueError("mode must be 'allcol', 'text_only', or 'half_text'")

    log_info, numeric, half_kept = product_column_ablation_info(rows, target_col)

    if mode == "allcol":
        return rows, log_info

    if mode == "text_only":
        if not log_info["text_keys"]:
            raise ValueError(
                "Product column ablation text_only: no text columns (text_keys empty)."
            )
        allowed = frozenset([target_col]) | frozenset(log_info["text_keys"])
        out = [{k: v for k, v in row.items() if k in allowed} for row in rows]
        return out, log_info

    # half_text
    allowed = frozenset([target_col]) | numeric | half_kept
    out = [{k: v for k, v in row.items() if k in allowed} for row in rows]
    return out, log_info


def get_meanpooled_embedding(dataset, idx, model):
    """Get mean-pooled embedding from model."""
    NaviForMaskedLM, HAETAE, TapasForMaskedLM, BertForMaskedLM = _lm_model_classes()
    if hasattr(dataset, '__getitem__'):
        if hasattr(dataset[idx], 'copy'):
            data_item = dataset[idx].copy()
        else:
            data_item = dict(dataset[idx])
    else:
        data_item = dataset[idx].copy()
    
    if isinstance(model, NaviForMaskedLM):
        allowed_keys = ['input_ids', 'attention_mask', 'position_ids', 'segment_ids', 'header_strings']
        if 'header_positions' in data_item and isinstance(data_item['header_positions'], dict):
            data_item['header_positions'] = [data_item['header_positions']]
        if 'header_strings' in data_item and isinstance(data_item['header_strings'], list):
            if all(isinstance(k, str) for k in data_item['header_strings']):
                data_item['header_strings'] = [data_item['header_strings']]
    elif isinstance(model, HAETAE):
        allowed_keys = ['input_ids', 'attention_mask', 'key_positions']
        if 'key_positions' in data_item and isinstance(data_item['key_positions'], dict):
            data_item['key_positions'] = [data_item['key_positions']]
    elif isinstance(model, TapasForMaskedLM):
        allowed_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        if 'token_type_ids' in data_item and isinstance(data_item['token_type_ids'], dict):
            data_item['token_type_ids'] = [data_item['token_type_ids']]
    else:
        allowed_keys = ['input_ids', 'attention_mask']
        
    device = next(model.parameters()).device

    inputs = {
        k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in data_item.items()
        if k in allowed_keys
    }

    with torch.no_grad():
        if isinstance(model, NaviForMaskedLM):
            outputs = model(**inputs)
            last_hidden = outputs[0]
        elif isinstance(model, HAETAE):
            outputs = model(**inputs)
            last_hidden = outputs["hidden_states"]
        elif isinstance(model, BertForMaskedLM):
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
        elif isinstance(model, TapasForMaskedLM):
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
    attention_mask = inputs['attention_mask'].unsqueeze(-1)
    masked_hidden = last_hidden * attention_mask
    sum_hidden = masked_hidden.sum(dim=1)
    valid_token_counts = attention_mask.sum(dim=1)
    mean_pooled = (sum_hidden / valid_token_counts).squeeze().to("cpu").numpy()

    return mean_pooled


def get_cls_embedding(dataset, idx, model):
    """Get CLS token embedding from model."""
    NaviForMaskedLM, HAETAE, TapasForMaskedLM, BertForMaskedLM = _lm_model_classes()
    if hasattr(dataset, '__getitem__'):
        if hasattr(dataset[idx], 'copy'):
            data_item = dataset[idx].copy()
        else:
            data_item = dict(dataset[idx])
    else:
        data_item = dataset[idx].copy()
    
    if isinstance(model, NaviForMaskedLM):
        allowed_keys = ['input_ids', 'attention_mask', 'position_ids', 'segment_ids', 'header_strings']
        if 'header_positions' in data_item and isinstance(data_item['header_positions'], dict):
            data_item['header_positions'] = [data_item['header_positions']]
        if 'header_strings' in data_item and isinstance(data_item['header_strings'], list):
            if all(isinstance(k, str) for k in data_item['header_strings']):
                data_item['header_strings'] = [data_item['header_strings']]
    elif isinstance(model, HAETAE):
        allowed_keys = ['input_ids', 'attention_mask', 'key_positions']
        if 'key_positions' in data_item and isinstance(data_item['key_positions'], dict):
            data_item['key_positions'] = [data_item['key_positions']]
    elif isinstance(model, TapasForMaskedLM):
        allowed_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        if 'token_type_ids' in data_item and isinstance(data_item['token_type_ids'], dict):
            data_item['token_type_ids'] = [data_item['token_type_ids']]
    else:
        allowed_keys = ['input_ids', 'attention_mask']
        
    device = next(model.parameters()).device

    inputs = {
        k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in data_item.items()
        if k in allowed_keys
    }

    with torch.no_grad():
        if isinstance(model, NaviForMaskedLM):
            outputs = model(**inputs)
            cls_embedding = outputs[0][:, 0, :].squeeze().to("cpu").numpy()
        elif isinstance(model, HAETAE):
            outputs = model(**inputs)
            cls_embedding = outputs["hidden_states"][:, 0, :].squeeze().to("cpu").numpy()
        elif isinstance(model, BertForMaskedLM):
            outputs = model(**inputs, output_hidden_states=True)
            cls_embedding = outputs.hidden_states[-1][:, 0, :].squeeze().to("cpu").numpy()
        elif isinstance(model, TapasForMaskedLM):
            outputs = model(**inputs, output_hidden_states=True)
            cls_embedding = outputs.hidden_states[-1][:, 0, :].squeeze().to("cpu").numpy()
    return cls_embedding


def extract_raw_features(dataset, target_col):
    """Extract raw features from JSON dataset for end-to-end baselines."""
    df = pd.DataFrame(dataset)
    y = df[target_col].values
    feature_df = df.drop(columns=[target_col])
    
    processed_features = []
    feature_names = []
    
    for col in feature_df.columns:
        col_data = feature_df[col]
        numeric_values = []
        is_numeric = True
        
        for val in col_data:
            if pd.isna(val) or val == '' or val is None or val == 'None':
                numeric_values.append(np.nan)
                continue
            
            try:
                val_str = str(val).strip()
                try:
                    float_val = float(val_str)
                    numeric_values.append(float_val)
                except ValueError:
                    is_numeric = False
                    break
            except (ValueError, TypeError):
                is_numeric = False
                break
        
        if is_numeric and len([v for v in numeric_values if not pd.isna(v)]) > 0:
            processed_features.append(numeric_values)
            feature_names.append(f"{col}_numeric")
        else:
            unique_ratio = col_data.nunique() / len(col_data) if len(col_data) > 0 else 1.0
            
            if unique_ratio < 0.5 and col_data.nunique() < 100:
                le = LabelEncoder()
                col_filled = col_data.fillna('__MISSING__')
                encoded = le.fit_transform(col_filled.astype(str))
                processed_features.append(encoded.tolist())
                feature_names.append(f"{col}_categorical")
            else:
                text_features = []
                for val in col_data:
                    if pd.isna(val) or val == '' or val is None:
                        text_features.append([0, 0])
                    else:
                        val_str = str(val)
                        text_features.append([len(val_str), len(val_str.split())])
                
                text_lengths = [f[0] for f in text_features]
                text_word_counts = [f[1] for f in text_features]
                processed_features.append(text_lengths)
                feature_names.append(f"{col}_text_length")
                processed_features.append(text_word_counts)
                feature_names.append(f"{col}_text_word_count")
    
    if processed_features:
        X = np.array(processed_features).T
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        X = np.zeros((len(dataset), 1))
    
    return X, y


def extract_features_tablevectorizer(
    dataset,
    target_col,
    high_cardinality_encoder="text",
    string_encoder_random_state=42,
):
    """Extract features using TableVectorizer from skrub.

    high_cardinality_encoder:
        - 'text': TableVectorizer with skrub TextEncoder (pretrained LM) for high-cardinality strings.
        - 'string': TableVectorizer with skrub StringEncoder (TF-IDF + truncated SVD).
    """
    from skrub import TableVectorizer

    if high_cardinality_encoder not in ("text", "string"):
        raise ValueError(
            "high_cardinality_encoder must be 'text' or 'string', "
            f"got {high_cardinality_encoder!r}"
        )

    df = pd.DataFrame(dataset)
    y = df[target_col].values
    feature_df = df.drop(columns=[target_col])

    if high_cardinality_encoder == "text":
        try:
            from skrub import TextEncoder

            vectorizer = TableVectorizer(high_cardinality=TextEncoder())
            X = vectorizer.fit_transform(feature_df)
        except (ImportError, ValueError, Exception) as e:
            print(
                f"⚠️  Warning: TextEncoder failed ({type(e).__name__}: {e}), using default TableVectorizer"
            )
            vectorizer = TableVectorizer()
            X = vectorizer.fit_transform(feature_df)
    else:
        try:
            from skrub import StringEncoder

            vectorizer = TableVectorizer(
                high_cardinality=StringEncoder(random_state=string_encoder_random_state)
            )
            X = vectorizer.fit_transform(feature_df)
        except (ImportError, ValueError, Exception) as e:
            print(
                f"⚠️  Warning: StringEncoder failed ({type(e).__name__}: {e}), using default TableVectorizer"
            )
            vectorizer = TableVectorizer()
            X = vectorizer.fit_transform(feature_df)

    if hasattr(X, "toarray"):
        X = X.toarray()
    else:
        X = np.array(X)

    return X, y


def extract_features_textencoder(dataset, target_col):
    """Extract features using TextEncoder from skrub."""
    from skrub import TextEncoder
    
    linearized_texts = []
    y = []
    
    for row_dict in dataset:
        feature_dict = {k: v for k, v in row_dict.items() if k != target_col}
        text_parts = []
        for key, value in feature_dict.items():
            if value is not None and value != '':
                text_parts.append(f"{key} is {value}")
        linearized_text = " ".join(text_parts)
        linearized_texts.append(linearized_text)
        y.append(row_dict[target_col])
    
    text_encoder = TextEncoder()
    text_df = pd.DataFrame({'text': linearized_texts})
    X = text_encoder.fit_transform(text_df)
    
    if hasattr(X, 'toarray'):
        X = X.toarray()
    else:
        X = np.array(X)
    
    y = np.array(y)
    return X, y


def extract_concatenated_navi_features(dataset, target_col, model, dataset_X):
    """Extract concatenated features: NAVI embeddings + numerical attributes."""
    from sklearn.preprocessing import StandardScaler
    
    navi_embeddings = []
    labels = []
    numerical_features_list = []
    
    df = pd.DataFrame(dataset)
    feature_df = df.drop(columns=[target_col])
    
    numerical_cols = []
    for col in feature_df.columns:
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            numerical_cols.append(col)
        else:
            sample_values = feature_df[col].dropna().head(100)
            if len(sample_values) == 0:
                continue
            
            is_numeric = True
            for val in sample_values:
                if pd.isna(val) or val == '' or val is None or val == 'None':
                    continue
                val_str = str(val).strip()
                try:
                    float(val_str)
                except (ValueError, TypeError):
                    is_numeric = False
                    break
            
            if is_numeric:
                numerical_cols.append(col)
    
    for i, row in enumerate(dataset):
        navi_emb = get_cls_embedding(dataset_X, i, model)
        navi_embeddings.append(navi_emb)
        
        numerical_vals = []
        for col in numerical_cols:
            val = row.get(col)
            if pd.isna(val) or val is None or val == '' or val == 'None':
                numerical_vals.append(0.0)
            else:
                try:
                    val_str = str(val).strip()
                    numerical_vals.append(float(val_str))
                except (ValueError, TypeError):
                    numerical_vals.append(0.0)
        numerical_features_list.append(numerical_vals)
        labels.append(row[target_col])
    
    navi_X = np.stack(navi_embeddings)
    
    if numerical_cols and len(numerical_cols) > 0:
        numerical_X = np.array(numerical_features_list)
        if numerical_X.shape[1] > 0:
            scaler = StandardScaler()
            numerical_X = scaler.fit_transform(numerical_X)
            X = np.hstack([navi_X, numerical_X])
        else:
            X = navi_X
    else:
        X = navi_X
    
    y = np.array(labels)
    return X, y


def run_classification_with_features(X, y, model_type="xgboost", test_size=0.2):
    """Generic function to run classification on features."""
    if model_type == "tabpfn":
        return run_row_classification_tabpfn(X, y, test_size=test_size)
    elif model_type in ["xgboost", "catboost", "lr"]:
        return run_row_classification(X, y, model_type=model_type, test_size=test_size)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def run_row_classification(X, y, model_type="xgboost", test_size=0.2):
    """Run row classification with specified model type."""
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X)
    
    has_nan = np.isnan(X).any()
    
    if has_nan:
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        X = imputer.fit_transform(X)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=None
        )
    except ValueError:
        # e.g. singleton classes — stratify impossible; match sklearn fallback
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=None
        )

    if model_type == "lr":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if model_type == "xgboost":
        clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', tree_method='hist', random_state=None)
    elif model_type == "catboost":
        clf = CatBoostClassifier(random_state=None, verbose=False, iterations=100)
    elif model_type == "lr":
        clf = LogisticRegression(max_iter=1000, random_state=None)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\n✅ Macro-F1 ({model_type.upper()} row classification): {f1:.4f}")
    return f1


def run_row_classification_tabpfn(X, y, test_size=0.2):
    """Run classification using TabPFN."""
    def _ensure_conda_runtime_for_llvmlite():
        import os
        import ctypes

        prefix = os.environ.get("CONDA_PREFIX")
        if not prefix:
            return

        libdir = os.path.join(prefix, "lib")
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        ld_parts = [p for p in ld.split(":") if p]
        if libdir not in ld_parts:
            os.environ["LD_LIBRARY_PATH"] = f"{libdir}:{ld}" if ld else libdir

        candidates = [
            os.path.join(libdir, "libstdc++.so.6"),
            os.path.join(prefix, "libstdc++.so.6"),
            os.path.join(libdir, "libgcc_s.so.1"),
            os.path.join(prefix, "libgcc_s.so.1"),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass

    _ensure_conda_runtime_for_llvmlite()

    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        raise ImportError("TabPFN is not installed. Please install it with: pip install tabpfn")
    except OSError as e:
        raise RuntimeError(
            "TabPFN import failed due to a native dependency issue (likely numba/llvmlite + libstdc++).\n"
            "If you're using conda, try re-activating the env so LD_LIBRARY_PATH points to $CONDA_PREFIX/lib.\n"
            f"Original error: {e}"
        )
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X = X.astype(np.float32)
    y_encoded = y_encoded.astype(np.int64)
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=None
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=None
        )
    
    try:
        import torch
        tabpfn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        clf = TabPFNClassifier(device=tabpfn_device)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"\n✅ Macro-F1 (TabPFN row classification): {f1:.4f}")
        return f1
    except (RuntimeError, Exception) as e:
        error_msg = str(e)
        if "Authentication error" in error_msg or "gated" in error_msg.lower():
            raise RuntimeError(
                "TabPFN requires HuggingFace authentication for gated models.\n"
                "Options:\n"
                "1. Run: hf auth login\n"
                "2. Set environment variable: export HF_TOKEN=your_token_here\n"
                "   (Get token from https://huggingface.co/settings/tokens)\n"
                "3. Visit https://huggingface.co/Prior-Labs/tabpfn_2_5 to accept terms first.\n"
                f"Original error: {error_msg}"
            )
        else:
            raise RuntimeError(f"TabPFN initialization failed: {error_msg}")
