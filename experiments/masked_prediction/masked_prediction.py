# Device

import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available and will be used.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU.")

# Imports

import sys
import os
import argparse
import gc
import glob
import json
import random
from datetime import datetime
from pathlib import Path

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import warnings
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

try:
    from config import config as app_config
    _checkpoint_epoch = getattr(app_config, "CHECKPOINT_EPOCH", 2)
except ImportError:
    app_config = None
    _checkpoint_epoch = 2

DATA_ROOT = "data/"

from experiments.masked_prediction.mp_utils import (
    evaluate_masked_prediction,
    evaluate_masked_prediction_cell_level,
    load_data,
)
from dataset.dataset import NaviDataset, BertDataset, HaetaeDataset, TapasDataset
from dataset.collator import CollatorForMaskedPrediction

from model.navi import NaviForMaskedLM
from baselines.haetae.model import HAETAE
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer, TapasForMaskedLM, BertConfig

from utils.navi_checkpoints import (
    discover_navi_domain_model_dirs,
    resolve_navi_checkpoint_path_with_fallback,
    resolve_navi_epoch_dir,
    navi_ablation_mode_for_model_name,
    iter_chunks,
)


# -- helper functions -- #

def load_baseline_models(tokenizer):
    """Load baseline models (BERT, HAETAE, TAPAS)"""
    models = {}
    
    # Config for HAETAE (local or online per RUN_ONLINE)
    if app_config:
        _bert_name = app_config.BERT_NAME
        _bert_kw = {"local_files_only": True}
    else:
        _bert_name = "bert-base-uncased"
        _bert_kw = {}
    bconfig = BertConfig.from_pretrained(_bert_name, **_bert_kw)

    ep = _checkpoint_epoch
    # Baseline models - using trained versions
    models['bert_movie'] = BertForMaskedLM.from_pretrained(f'./models/bert_movie/epoch_{ep}', local_files_only=True)
    models['bert_movie'] = models['bert_movie'].to(device)
    models['bert_movie'].eval()

    models['bert_product'] = BertForMaskedLM.from_pretrained(f'./models/bert_product/epoch_{ep}', local_files_only=True)
    models['bert_product'] = models['bert_product'].to(device)
    models['bert_product'].eval()

    models['haetae_product'] = HAETAE(bconfig, tokenizer, f"./models/haetae_product/epoch_{ep}")
    models['haetae_product'] = models['haetae_product'].to(device)
    models['haetae_product'].eval()

    models['haetae_movie'] = HAETAE(bconfig, tokenizer, f"./models/haetae_movie/epoch_{ep}")
    models['haetae_movie'] = models['haetae_movie'].to(device)
    models['haetae_movie'].eval()

    models['tapas_movie'] = TapasForMaskedLM.from_pretrained(f'./models/tapas_movie/epoch_{ep}', local_files_only=True)
    models['tapas_movie'] = models['tapas_movie'].to(device)
    models['tapas_movie'].eval()

    models['tapas_product'] = TapasForMaskedLM.from_pretrained(f'./models/tapas_product/epoch_{ep}', local_files_only=True)
    models['tapas_product'] = models['tapas_product'].to(device)
    models['tapas_product'].eval()

    # NAVI models - use find_epoch2_path to locate the actual epoch directory
    navi_movie_base = './models/navi_movie'
    navi_movie_path = find_epoch2_path(navi_movie_base)
    if navi_movie_path:
        models['navi_movie'] = NaviForMaskedLM(navi_movie_path)
        models['navi_movie'] = models['navi_movie'].to(device)
        models['navi_movie'].eval()
    else:
        print(f"⚠️  NAVI Movie model not found: {navi_movie_base}/*epoch_{_checkpoint_epoch}")
    
    navi_product_base = './models/navi_product'
    navi_product_path = find_epoch2_path(navi_product_base)
    if navi_product_path:
        models['navi_product'] = NaviForMaskedLM(navi_product_path)
        models['navi_product'] = models['navi_product'].to(device)
        models['navi_product'].eval()
    else:
        print(f"⚠️  NAVI Product model not found: {navi_product_base}/*epoch_{_checkpoint_epoch}")
    
    return models


def load_navi_model_for_domain(domain: str):
    """
    Load only NaviForMaskedLM for ./models/navi_{domain} (epoch from find_epoch2_path / CHECKPOINT_EPOCH).
    Returns None if no checkpoint found.
    """
    domain_lower = domain.lower()
    base = f"./models/navi_{domain_lower}"
    path = find_epoch2_path(base)
    if not path:
        print(f"⚠️  NAVI model not found: {base}/*epoch_{_checkpoint_epoch}")
        return None
    print(f"✓ Loading NAVI from {path}")
    model = NaviForMaskedLM(path).to(device).eval()
    return model


def run_cell_level_masked_prediction_split(
    dataset,
    model,
    tokenizer,
    collator,
    epoch: int,
    split_label: str,
    thresholds,
):
    """Per-field cell-level τ sweep for one split (header epoch=1 or value epoch=5)."""
    print(f"{split_label} (cell-level):")
    return evaluate_masked_prediction_cell_level(
        dataset, model, tokenizer, collator, epoch=epoch, thresholds=thresholds,
    )


def run_navi_masked_prediction_eval(
    dataset,
    model,
    tokenizer,
    collator,
    cell_level_thresholds=(0.5, 0.6, 0.8),
    include_cell_level: bool = False,
    section_title: str = "NAVI",
):
    """
    Token-level masked prediction for NAVI (header + value epochs) and optional cell-level splits.
    """
    print(f"\n{section_title}")
    print("Header:")
    evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch=1)
    if include_cell_level:
        run_cell_level_masked_prediction_split(
            dataset, model, tokenizer, collator, epoch=1, split_label="Header", thresholds=cell_level_thresholds,
        )
    print("Value:")
    evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch=5)
    if include_cell_level:
        run_cell_level_masked_prediction_split(
            dataset, model, tokenizer, collator, epoch=5, split_label="Value", thresholds=cell_level_thresholds,
        )


def _navi_ablation_mode_for_load(ablation):
    """Model only has woGHA/woGHC; woSSI, woMSM, woESA are dataset/training-only -> load as full."""
    return ablation if ablation in ('woGHA', 'woGHC') else 'full'

def load_ablation_models(tokenizer):
    """Load ablation models (woSSI: dataset-only, woGHA/woGHC: model ablations, woMSM/woESA: training-only)."""
    models = {}
    
    domains = ['movie', 'product']
    ablation_values = ['woSSI', 'woMSM', 'woESA', 'woGHA', 'woGHC']

    for domain in domains:
        for ablation in ablation_values:
            base_path = f'./models/navi_{domain}_{ablation}'
            epoch_path = find_epoch2_path(base_path)
            if not epoch_path:
                explicit_path = f'./models/navi_{domain}_{ablation}/{ablation}_HVB_seed42_cleaned_tau0.02_0.14_percentile_epoch_{_checkpoint_epoch}'
                epoch_path = explicit_path if os.path.exists(explicit_path) else None
            if epoch_path:
                model_name = f'navi_{domain}_{ablation}'
                model_ablation = _navi_ablation_mode_for_load(ablation)
                models[model_name] = NaviForMaskedLM(epoch_path, ablation_mode=model_ablation)
                models[model_name] = models[model_name].to(device)
                models[model_name].eval()
            else:
                print(f"⚠️  Model not found: {base_path}/*epoch_{_checkpoint_epoch}")
    
    return models

def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def _save_mp_hyperparam_json(results, domain, run_id=None):
    """Write masked-prediction hyperparam sweep results for table export."""
    rid = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path("experiments/logs") / f"masked_prediction_hyperparam_sensitivity_{domain.lower()}_{rid}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    variants_out = {
        name: {
            "header_accuracy": float(m["header_accuracy"]),
            "value_accuracy": float(m["value_accuracy"]),
        }
        for name, m in results.items()
    }
    payload = {
        "experiment": "masked_prediction_hyperparam_sensitivity",
        "domain": domain,
        "run_id": rid,
        "checkpoint_epoch": _checkpoint_epoch,
        "variants": variants_out,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n✅ Results saved to {out_path}")

def find_epoch2_path(base_path):
    """Find checkpoint epoch directory using glob pattern (e.g. epoch_4)."""
    p = resolve_navi_epoch_dir(base_path, _checkpoint_epoch)
    return str(p) if p else None

def load_b200_models(tokenizer):
    """Load baseline models from models_b200/ directory (BERT, HAETAE, TAPAS, NAVI)"""
    models = {}
    
    # Config for HAETAE (local or online per RUN_ONLINE)
    if app_config:
        _bert_name = app_config.BERT_NAME
        _bert_kw = {"local_files_only": True}
    else:
        _bert_name = "bert-base-uncased"
        _bert_kw = {}
    bconfig = BertConfig.from_pretrained(_bert_name, **_bert_kw)

    ep = _checkpoint_epoch
    # Baseline models - using trained versions from models_b200/
    models['bert_movie'] = BertForMaskedLM.from_pretrained(f'./models_b200/bert_movie/epoch_{ep}', local_files_only=True)
    models['bert_movie'] = models['bert_movie'].to(device)
    models['bert_movie'].eval()

    models['bert_product'] = BertForMaskedLM.from_pretrained(f'./models_b200/bert_product/epoch_{ep}', local_files_only=True)
    models['bert_product'] = models['bert_product'].to(device)
    models['bert_product'].eval()

    models['haetae_product'] = HAETAE(bconfig, tokenizer, f"./models_b200/haetae_product/epoch_{ep}")
    models['haetae_product'] = models['haetae_product'].to(device)
    models['haetae_product'].eval()

    models['haetae_movie'] = HAETAE(bconfig, tokenizer, f"./models_b200/haetae_movie/epoch_{ep}")
    models['haetae_movie'] = models['haetae_movie'].to(device)
    models['haetae_movie'].eval()

    models['tapas_movie'] = TapasForMaskedLM.from_pretrained(f'./models_b200/tapas_movie/epoch_{ep}', local_files_only=True)
    models['tapas_movie'] = models['tapas_movie'].to(device)
    models['tapas_movie'].eval()

    models['tapas_product'] = TapasForMaskedLM.from_pretrained(f'./models_b200/tapas_product/epoch_{ep}', local_files_only=True)
    models['tapas_product'] = models['tapas_product'].to(device)
    models['tapas_product'].eval()

    # NAVI models - use find_epoch2_path to locate the actual epoch directory
    navi_movie_base = './models_b200/navi_movie'
    navi_movie_path = find_epoch2_path(navi_movie_base)
    if navi_movie_path:
        models['navi_movie'] = NaviForMaskedLM(navi_movie_path)
        models['navi_movie'] = models['navi_movie'].to(device)
        models['navi_movie'].eval()
    else:
        print(f"⚠️  NAVI Movie model not found: {navi_movie_base}/*epoch_{_checkpoint_epoch}")
    
    navi_product_base = './models_b200/navi_product'
    navi_product_path = find_epoch2_path(navi_product_base)
    if navi_product_path:
        models['navi_product'] = NaviForMaskedLM(navi_product_path)
        models['navi_product'] = models['navi_product'].to(device)
        models['navi_product'].eval()
    else:
        print(f"⚠️  NAVI Product model not found: {navi_product_base}/*epoch_{_checkpoint_epoch}")
    
    return models

def load_prototyping_models(tokenizer, domain):
    """Load prototyping models: default navi_{domain} plus woGHA and woGHC ablations."""
    models = {}
    domain_lower = domain.lower()

    # 1. Default navi_{domain}
    base_path = f'./models/navi_{domain_lower}'
    epoch_path = find_epoch2_path(base_path)
    if epoch_path:
        model_name = f'navi_{domain_lower}'
        models[model_name] = NaviForMaskedLM(epoch_path)
        models[model_name] = models[model_name].to(device)
        models[model_name].eval()
        print(f"✓ Loaded {model_name} from: {epoch_path}")
    else:
        print(f"⚠️  Model not found: {base_path}/*epoch_{_checkpoint_epoch}")

    # 2. woGHA and woGHC (model ablations; pass ablation_mode for correct architecture)
    for ablation in ['woGHA', 'woGHC']:
        base_path = f'./models/navi_{domain_lower}_{ablation}'
        epoch_path = find_epoch2_path(base_path)
        if epoch_path:
            model_name = f'navi_{domain_lower}_{ablation}'
            models[model_name] = NaviForMaskedLM(epoch_path, ablation_mode=ablation)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {epoch_path}")
        else:
            # Try explicit path used by load_ablation_models
            explicit_path = f'./models/navi_{domain_lower}_{ablation}/{ablation}_HVB_seed42_cleaned_tau0.02_0.14_percentile_epoch_{_checkpoint_epoch}'
            if os.path.exists(explicit_path):
                model_name = f'navi_{domain_lower}_{ablation}'
                models[model_name] = NaviForMaskedLM(explicit_path, ablation_mode=ablation)
                models[model_name] = models[model_name].to(device)
                models[model_name].eval()
                print(f"✓ Loaded {model_name} from: {explicit_path}")
            else:
                print(f"⚠️  Model not found: {base_path}/*epoch_{_checkpoint_epoch} or {explicit_path}")

    return models

def load_ablation_variants_models(tokenizer, domain):
    """Load ablation variant models for a specific domain"""
    models = {}
    domain_lower = domain.lower()
    
    # 1. Default model
    base_path = f'./models/navi_{domain_lower}'
    model_path = find_epoch2_path(base_path)
    if model_path:
        model_name = f'navi_{domain_lower}'
        models[model_name] = NaviForMaskedLM(model_path)
        models[model_name] = models[model_name].to(device)
        models[model_name].eval()
        print(f"✓ Loaded {model_name} from: {model_path}")
    else:
        print(f"⚠️  Model not found: {base_path}/*epoch_{_checkpoint_epoch}")
    
    # 2-4. Entropy threshold variants
    ethresh_values = ['10_90', '40_60', '50_50']
    for ethresh in ethresh_values:
        base_path = f'./models/navi_{domain_lower}_ethresh{ethresh}'
        model_path = find_epoch2_path(base_path)
        if model_path:
            model_name = f'navi_{domain_lower}_ethresh{ethresh}'
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {model_path}")
        else:
            print(f"⚠️  Model not found: {base_path}/*epoch_{_checkpoint_epoch}")
    
    # 5-6. Temperature variants
    tau_values = ['0p07_0p07', '0p13_0p13']
    for tau in tau_values:
        base_path = f'./models/navi_{domain_lower}_tau{tau}'
        model_path = find_epoch2_path(base_path)
        if model_path:
            model_name = f'navi_{domain_lower}_tau{tau}'
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {model_path}")
        else:
            print(f"⚠️  Model not found: {base_path}/*epoch_{_checkpoint_epoch}")
    
    # 7-8. Gradient accumulation variants (Neg batch size)
    ga_values = ['1', '4']  # ga1 = batch size 32, ga4 = batch size 128
    for ga in ga_values:
        base_path = f'./models/navi_{domain_lower}_ga{ga}'
        model_path = find_epoch2_path(base_path)
        if model_path:
            model_name = f'navi_{domain_lower}_ga{ga}'
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {model_path}")
        else:
            print(f"⚠️  Model not found: {base_path}/*epoch_{_checkpoint_epoch}")
    
    return models

def group_data_by_table(data):
    """
    Group data by table based on row_id patterns.
    Each table contains 10 consecutive rows with row_id 450-459.
    
    Args:
        data: List of dictionaries with row_id field
        
    Returns:
        List of tuples (table_idx, table_rows) where table_rows is a list of 10 rows
    """
    grouped_tables = []
    current_table = []
    table_idx = 0
    
    for row in data:
        row_id = int(row['row_id'])
        
        # Check if this is the start of a new table (row_id 450)
        if row_id == 450 and current_table:
            # Save the previous table
            if len(current_table) == 10:  # Ensure we have a complete table
                grouped_tables.append((table_idx, current_table))
                table_idx += 1
            current_table = []
        
        current_table.append(row)
    
    # Don't forget the last table
    if current_table and len(current_table) == 10:
        grouped_tables.append((table_idx, current_table))
    
    print(f"Grouped {len(data)} rows into {len(grouped_tables)} tables")
    return grouped_tables

def _parse_cell_level_thresholds(s: str):
    """Comma-separated floats, e.g. '0.5,0.6,0.8'."""
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    if not parts:
        return (0.5, 0.6, 0.8)
    return tuple(float(p) for p in parts)


def evaluate_baselines(
    bert_dataset,
    haetae_dataset,
    tapas_dataset,
    navi_dataset,
    models,
    tokenizers,
    collators,
    domain,
    cell_level_eval: bool = False,
    cell_level_thresholds=(0.5, 0.6, 0.8),
):
    """Evaluate baseline models (BERT, HAETAE, TAPAS, NAVI). Cell-level (τ sweep) runs for NAVI only."""
    print(f"\n{domain} - Baselines")
    
    # BERT
    print("\nBERT")
    bert_key = f'bert_{domain.lower()}'
    print("Header:")
    evaluate_masked_prediction(bert_dataset, models[bert_key], tokenizers[0], collators[0], epoch=1)
    print("Value:")
    evaluate_masked_prediction(bert_dataset, models[bert_key], tokenizers[0], collators[0], epoch=5)
    
    # TAPAS
    print("\nTAPAS")
    tapas_key = f'tapas_{domain.lower()}'
    print("Header:")
    evaluate_masked_prediction(tapas_dataset, models[tapas_key], tokenizers[1], collators[1], epoch=1)
    print("Value:")
    evaluate_masked_prediction(tapas_dataset, models[tapas_key], tokenizers[1], collators[1], epoch=5)
    
    # HAETAE
    print("\nHAETAE")
    haetae_key = f'haetae_{domain.lower()}'
    print("Header:")
    evaluate_masked_prediction(haetae_dataset, models[haetae_key], tokenizers[0], collators[0], epoch=1)
    print("Value:")
    evaluate_masked_prediction(haetae_dataset, models[haetae_key], tokenizers[0], collators[0], epoch=5)
    
    # NAVI
    navi_key = f'navi_{domain.lower()}'
    run_navi_masked_prediction_eval(
        navi_dataset,
        models[navi_key],
        tokenizers[0],
        collators[0],
        cell_level_thresholds=cell_level_thresholds,
        include_cell_level=cell_level_eval,
        section_title="NAVI",
    )

def evaluate_ablations(
    dataset,
    models,
    tokenizer,
    collator,
    domain,
    cell_level_eval: bool = False,
    cell_level_thresholds=(0.5, 0.6, 0.8),
):
    """Evaluate NAVI ablation checkpoints. Cell-level (τ sweep) runs for each loaded NAVI model."""
    print(f"\n{domain} - Ablations")
    
    # Filter models for this domain (include exact navi_{domain} and names containing _{domain}_)
    domain_lower = domain.lower()
    domain_models = {k: v for k, v in models.items() if k == f'navi_{domain_lower}' or f"_{domain_lower}_" in k}
    
    for model_name, model in domain_models.items():
        run_navi_masked_prediction_eval(
            dataset,
            model,
            tokenizer,
            collator,
            cell_level_thresholds=cell_level_thresholds,
            include_cell_level=cell_level_eval,
            section_title=model_name,
        )

def evaluate_navi_hyperparam_sweep(dataset, tokenizer, collator, domain, max_loaded_variants=4):
    """Evaluate all `./models/navi_{domain}` and `./models/navi_{domain}_*` checkpoints for this domain."""
    print(f"\n{domain} - NAVI hyperparameter sweep (max_loaded_variants={max_loaded_variants})")
    discovered = discover_navi_domain_model_dirs(domain)
    results = {}
    pending = []
    for model_name, base_dir in discovered:
        epoch_path = resolve_navi_checkpoint_path_with_fallback(model_name, base_dir, _checkpoint_epoch)
        if epoch_path is None:
            print(f"⚠️  Skipping {model_name}: no checkpoint at epoch_{_checkpoint_epoch}")
            continue
        pending.append((model_name, epoch_path))

    for wave in iter_chunks(pending, max_loaded_variants):
        wave_models = {}
        for model_name, epoch_path in wave:
            mode = navi_ablation_mode_for_model_name(model_name)
            print(f"✓ Loading {model_name} from {epoch_path}")
            wave_models[model_name] = NaviForMaskedLM(str(epoch_path), ablation_mode=mode).to(device).eval()
        for model_name, model in wave_models.items():
            print(f"\n{model_name}")
            print("Header:")
            header_acc = evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch=1)
            print("Value:")
            value_acc = evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch=5)
            results[model_name] = {
                'header_accuracy': header_acc,
                'value_accuracy': value_acc,
            }
        del wave_models
        clear_memory()
    return results


def evaluate_hyperparams(dataset, tokenizer, collator, domain, max_loaded_variants=4):
    """Same sweep as hyperparam_sensitivity; kept for backward compatibility."""
    return evaluate_navi_hyperparam_sweep(dataset, tokenizer, collator, domain, max_loaded_variants)


def evaluate_hyperparam_sensitivity(dataset, tokenizer, collator, domain, max_loaded_variants=4):
    """Same sweep as hyperparams; kept for backward compatibility."""
    return evaluate_navi_hyperparam_sweep(dataset, tokenizer, collator, domain, max_loaded_variants)

def _icml_entropy_ablation_variant_map():
    """
    Map table row labels -> model folder suffix.

    Model folders follow: navi_{domain}_esaRouting{suffix}
    """
    return [
        ("Default", "entropy"),
        ("AllLow", "all_low"),
        ("AllHigh", "all_high"),
        ("Random", "random"),
    ]


def _fmt_cell_or_na(v) -> str:
    if v is None:
        return "---"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


def _write_icml_entropy_ablation_table(out_path, table_rows):
    """
    table_rows: List[Tuple[row_label, imp_movie, cls_movie, imp_product, cls_product]]
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header_1 = "            |    Movie   | Product"
    header_2 = "            | Imp | Cls | Imp | Cls"
    bar = "-" * 34

    lines = [header_1, header_2, bar]
    for (row_label, imp_m, cls_m, imp_p, cls_p) in table_rows:
        # Match the requested "Default|" style.
        lines.append(
            f"{row_label}|{_fmt_cell_or_na(imp_m)}|{_fmt_cell_or_na(cls_m)}|"
            f"{_fmt_cell_or_na(imp_p)}|{_fmt_cell_or_na(cls_p)}"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✅ ICML entropy ablation table written to: {out_path}")


def evaluate_icml_entropy_ablation(tokenizer, collator, max_samples=1000, run_id=None):
    """
    ICML entropy routing ablation:
    - Imp: masked value prediction accuracy (epoch=5)
    - Cls: row-classification F1 (xgboost, cls embedding, single split)
    """
    import numpy as np
    from datetime import datetime as _dt

    # Import lazily to avoid heavy import unless this mode is used.
    from experiments.downstream_tasks.row_classification import (
        run_cls_classification,
        preprocess_wdc_movie,
        preprocess_wdc_product,
        stratified_sample,
        verify_target_columns_present,
    )

    rid = run_id or _dt.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path("experiments/icml_entropy_ablation") / f"icml_entropy_ablation_{rid}.txt"

    domains = ["Movie", "Product"]
    variant_map = _icml_entropy_ablation_variant_map()

    # Load & preprocess classification + masked-prediction data once per domain.
    domain_mp_datasets = {}
    domain_cls_datasets = {}

    for domain in domains:
        domain_lower = domain.lower()

        # --- Masked prediction dataset (Imp) ---
        mp_path = Path(DATA_ROOT) / "cleaned" / domain / "test" / f"WDC_{domain_lower}_for_mp.jsonl"
        raw_json = load_data(str(mp_path))
        print(f"\n[{domain}] Loaded {len(raw_json)} masked-prediction rows from: {mp_path}")

        random.seed(42)
        if len(raw_json) > max_samples:
            raw_json = random.sample(raw_json, max_samples)
            print(f"[{domain}] Sampled {max_samples} rows (seed=42) for Imp")
        else:
            print(f"[{domain}] Using all {len(raw_json)} rows for Imp")

        json_data = [(i, row) for i, row in enumerate(raw_json)]
        navi_dataset = NaviDataset(json_data)
        domain_mp_datasets[domain] = navi_dataset

        # --- Classification dataset (Cls) ---
        cls_path = Path(DATA_ROOT) / "cleaned" / domain / "test" / f"WDC_{domain_lower}_for_cls.jsonl"
        raw_cls = load_data(str(cls_path))
        print(f"[{domain}] Loaded {len(raw_cls)} classification rows from: {cls_path}")

        if domain == "Product":
            raw_cls = verify_target_columns_present(raw_cls, target_columns=["category"])
            raw_cls = preprocess_wdc_product(raw_cls)
            raw_cls = stratified_sample(raw_cls, "category", sample_size=max_samples)
            target_col = "category"
        else:
            raw_cls = verify_target_columns_present(raw_cls, target_columns=["genres"])
            raw_cls = preprocess_wdc_movie(raw_cls)
            raw_cls = stratified_sample(raw_cls, "genres", sample_size=max_samples)
            target_col = "genres"

        print(f"[{domain}] Classification dataset size after sampling: {len(raw_cls)}")
        domain_cls_datasets[domain] = (raw_cls, target_col)

    # Sweep models and fill table.
    table_rows = []
    for row_label, suffix in variant_map:
        imp_movie = cls_movie = imp_product = cls_product = None

        for domain in domains:
            domain_lower = domain.lower()
            base_dir = Path("./models") / f"navi_{domain_lower}_esaRouting{suffix}"
            epoch_path = find_epoch2_path(str(base_dir))

            if not epoch_path:
                print(f"⚠️  Missing model for {domain} {row_label}: {base_dir}/*epoch_{_checkpoint_epoch}")
                continue

            model_name = f"navi_{domain_lower}_esaRouting{suffix}"
            print(f"\n[{domain}] Loading {model_name} from: {epoch_path}")
            model = NaviForMaskedLM(epoch_path).to(device).eval()

            try:
                # Imp (value prediction) = epoch=5
                imp_acc = evaluate_masked_prediction(
                    domain_mp_datasets[domain],
                    model,
                    tokenizer,
                    collator,
                    epoch=5,
                )

                # Cls (row classification) = xgboost on cls embeddings
                cls_data, target_col = domain_cls_datasets[domain]

                # Deterministic single split for the table.
                random.seed(42)
                np.random.seed(42)
                f1 = run_cls_classification(
                    dataset=cls_data,
                    target_col=target_col,
                    model=model,
                    model_name=model_name,
                    domain=domain_lower,
                    ml_model="xgboost",
                    embedding_type="cls",
                )
            except Exception as e:
                print(f"⚠️  Evaluation failed for {model_name}: {e}")
                imp_acc, f1 = None, None
            finally:
                del model
                clear_memory()

            if domain == "Movie":
                imp_movie, cls_movie = imp_acc, f1
            else:
                imp_product, cls_product = imp_acc, f1

        table_rows.append((row_label, imp_movie, cls_movie, imp_product, cls_product))

    _write_icml_entropy_ablation_table(out_path, table_rows)
    return out_path

def load_tau_align_ethresh_models(tokenizer, domain):
    """Load tau/align/ethresh variant models for a specific domain"""
    models = {}
    domain_lower = domain.lower()
    
    # Define all model variants based on log files
    # Format: (tau, align, ga) where ga can be None or a string like '1'
    variants = [
        ('0p01_0p1', '0p1', None),
        ('0p01_0p14', '0p1', None),
        ('0p01_0p14', '0p1', '1'),
        ('0p01_0p14', '0p05', None),
        ('0p01_0p14', '0p05', '1'),
        ('0p02_0p14', '0p1', None),
        ('0p02_0p14', '0p05', None),
        ('0p02_0p14', '0p05', '1'),
        ('0p02_0p14', '1p0', None),
        ('0p03_0p25', '0p1', None),
        ('0p05_0p1', '0p1', None),
    ]
    
    ethresh = '10_90'
    
    for tau, align, ga in variants:
        # Build base path
        if ga is not None:
            base_path = f'./models/navi_{domain_lower}_tau{tau}_align{align}_ethresh{ethresh}_ga{ga}'
            model_name = f'navi_{domain_lower}_tau{tau}_align{align}_ethresh{ethresh}_ga{ga}'
        else:
            base_path = f'./models/navi_{domain_lower}_tau{tau}_align{align}_ethresh{ethresh}'
            model_name = f'navi_{domain_lower}_tau{tau}_align{align}_ethresh{ethresh}'
        
        model_path = find_epoch2_path(base_path)
        if model_path:
            models[model_name] = NaviForMaskedLM(model_path)
            models[model_name] = models[model_name].to(device)
            models[model_name].eval()
            print(f"✓ Loaded {model_name} from: {model_path}")
        else:
            print(f"⚠️  Model not found: {base_path}/*epoch_{_checkpoint_epoch}")
    
    return models

def evaluate_ablation_variants(dataset, tokenizer, collator, domain):
    """Evaluate ablation variants (entropy threshold, temperature, gradient accumulation)"""
    print(f"\n{domain} - Ablation Variants")
    
    results = {}
    
    # Load all ablation variant models
    models = load_ablation_variants_models(tokenizer, domain)
    
    # Evaluate each model (Value prediction only, epoch=5)
    for model_name, model in models.items():
        print(f"\n{model_name}")
        print("Value:")
        value_acc = evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch=5)
        results[model_name] = {
            'value_accuracy': value_acc
        }
    
    # Clear memory
    del models
    clear_memory()
    
    return results

def evaluate_tau_align_ethresh(dataset, tokenizer, collator, domain):
    """Evaluate tau/align/ethresh variant models"""
    print(f"\n{domain} - Tau/Align/Ethresh Variants")
    
    results = {}
    
    # Load tau/align/ethresh models
    models = load_tau_align_ethresh_models(tokenizer, domain)
    
    # Evaluate each model (Header and Value prediction)
    for model_name, model in models.items():
        print(f"\n{model_name}")
        print("Header:")
        header_acc = evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch=1)
        print("Value:")
        value_acc = evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch=5)
        results[model_name] = {
            'header_accuracy': header_acc,
            'value_accuracy': value_acc
        }
    
    # Clear memory
    del models
    clear_memory()
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate masked prediction models')
    parser.add_argument(
        '--model',
        choices=[
            'baselines',
            'ablations',
            'hyperparams',
            'ablation_variants',
            'tau_align_ethresh',
            'hyperparam_sensitivity',
            'prototyping',
            'b200',
            'icml_entropy_ablation',
            'cm2',
            'navi_cell_level',
            'unified_semantic',
        ],
        required=True,
        help=(
            "Evaluation mode. navi_cell_level: load only NAVI for this domain and run token + cell-level "
            "(no BERT/TAPAS/HAETAE). cm2: CM2 tabular masking. unified_semantic: NAVI + CM2 field/cell "
            "scores with matching min/max/avg/Q1/Q2/Q3 (see docs). b200: checkpoints under models_b200/."
        ),
    )
    parser.add_argument('--domain', choices=['Movie', 'Product'], 
                       required=True, help='Domain to evaluate on')
    parser.add_argument('--max_loaded_variants', type=int, default=4, metavar='K',
                       help='How many NAVI checkpoints to keep on GPU at once during hyperparam sweeps (default 1; use 3–4 only if VRAM allows)')
    parser.add_argument('--run_id', type=str, default=None,
                       help='Optional id for output JSON filename (e.g. shared with row_classification + table export)')
    parser.add_argument('--cm2_checkpoint', type=str, default=None,
                       help='For --model cm2 or unified_semantic: directory with pytorch_model.bin (default: CM2_* env or baselines/CM2/mask_v1_{domain}_unsup)')
    parser.add_argument('--cm2_table_flag', type=int, default=0,
                       help='For --model cm2 or unified_semantic: table_flag for multi-table CM2 checkpoints')
    parser.add_argument('--cm2_num_tol', type=float, default=0.05,
                       help='For --model cm2: |pred_norm-target| < tol counts as correct for numerical masks; '
                       'for unified_semantic CM2 num: scales clamp(1-|err|/tol,0,1)')
    parser.add_argument('--cm2_cat_cos', type=float, default=0.70,
                       help='For --model cm2: cosine(pred,target) > this counts as correct for categorical masks (0.95 is often too strict vs training 1−cos loss)')
    parser.add_argument(
        '--cm2_mode',
        type=str,
        default='structured',
        choices=['structured', 'naive_tokens'],
        help=(
            "For --model cm2: 'structured' = CM2's native num/cat mask heads; "
            "'naive_tokens' = mask one tokenizer token per cell / column-name, predict id via embedding softmax (not encoder)"
        ),
    )
    parser.add_argument(
        '--cm2_naive_include_headers',
        action='store_true',
        help="For --cm2_mode naive_tokens: also score column-name tokens (num_col / col_cat header strings)",
    )
    parser.add_argument(
        '--cell_level_eval',
        action='store_true',
        help=(
            "Additionally print per-field cell-level accuracy (additive; token-level metrics unchanged). "
            "For --model baselines and b200: NAVI only. For ablations/prototyping: all loaded NAVI checkpoints. "
            "Each field (column) is one vote; only MLM targets (labels != -100) in that field count; "
            "success at τ if correct_masked/total_masked_in_field >= τ. See --cell_level_thresholds."
        ),
    )
    parser.add_argument(
        '--cell_level_thresholds',
        type=str,
        default='0.5,0.6,0.8',
        help="Comma-separated τ values for --cell_level_eval or --model navi_cell_level (default: 0.5,0.6,0.8).",
    )
    
    args = parser.parse_args()
    cell_level_thresholds = _parse_cell_level_thresholds(args.cell_level_thresholds)
    
    print(f"Evaluating {args.model} models on {args.domain} domain")
    
    # Data loading - only load data for the specified domain
    domain_lower = args.domain.lower()
    data_path = Path(DATA_ROOT) / "cleaned" / args.domain / "test" / f"WDC_{domain_lower}_for_mp.jsonl"
    print(f"Loading data from: {data_path}")
    raw_json = load_data(str(data_path))
    print(f"Loaded {len(raw_json)} rows")
    
    # Sample 1000 rows with fixed random seed for reproducibility
    random.seed(42)
    if len(raw_json) > 1000:
        raw_json = random.sample(raw_json, 1000)
        print(f"Sampled 1000 rows (seed=42)")
    else:
        print(f"Using all {len(raw_json)} rows (less than 1000 available)")

    # Create tuples with sequential table IDs (0, 1, 2, ...) for NaviDataset
    # NaviDataset can handle tuple format (table_id, row)
    json_data = [(i, row) for i, row in enumerate(raw_json)]

    # Extract flat list of rows for other datasets
    json_data_flat = [row for _, row in json_data]

    # Prepare datasets
    navi_dataset = NaviDataset(json_data)  # NaviDataset can handle tuple format

    if args.model == 'navi_cell_level':
        if app_config:
            AHB_tokenizer = BertTokenizer.from_pretrained(
                app_config.BERT_NAME, local_files_only=app_config.use_local_files_only()
            )
        else:
            AHB_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        AHB_collator = CollatorForMaskedPrediction(tokenizer=AHB_tokenizer)
        navi_model = load_navi_model_for_domain(args.domain)
        if navi_model is None:
            sys.exit(1)
        run_navi_masked_prediction_eval(
            navi_dataset,
            navi_model,
            AHB_tokenizer,
            AHB_collator,
            cell_level_thresholds=cell_level_thresholds,
            include_cell_level=True,
            section_title="NAVI",
        )
        del navi_model
        clear_memory()
        return

    if args.model == "unified_semantic":
        from experiments.downstream_tasks.cm2_utils import (
            default_cm2_movie_ckpt,
            default_cm2_product_ckpt,
            load_cm2_mask_model,
        )
        from experiments.masked_prediction.cm2_mp_utils import evaluate_cm2_unified_semantic_scores
        from experiments.masked_prediction.unified_semantic_navi import evaluate_navi_unified_semantic

        if app_config:
            AHB_tokenizer = BertTokenizer.from_pretrained(
                app_config.BERT_NAME, local_files_only=app_config.use_local_files_only()
            )
        else:
            AHB_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        AHB_collator = CollatorForMaskedPrediction(
            tokenizer=AHB_tokenizer,
            mask_full_field=True,
            mask_replace_prob=1.0,
            random_replace_prob=0.0,
        )

        print(f"\n=== Unified semantic evaluation ({args.domain}) ===")
        navi_model = load_navi_model_for_domain(args.domain)
        if navi_model is None:
            sys.exit(1)
        evaluate_navi_unified_semantic(navi_dataset, navi_model, AHB_collator)
        del navi_model
        clear_memory()

        ckpt = args.cm2_checkpoint
        if not ckpt:
            ckpt = default_cm2_movie_ckpt() if args.domain == "Movie" else default_cm2_product_ckpt()
        print(f"Loading CM2 checkpoint: {ckpt}")
        cm2_model = load_cm2_mask_model(ckpt, device)
        evaluate_cm2_unified_semantic_scores(
            json_data_flat,
            cm2_model,
            table_flag=args.cm2_table_flag,
            mlm_probability=0.35,
            base_seed=42,
            num_abs_tol=args.cm2_num_tol,
        )
        del cm2_model
        clear_memory()
        return

    bert_dataset = BertDataset(json_data_flat, mode="masked_prediction")
    haetae_dataset = HaetaeDataset(json_data_flat)
    tapas_dataset = TapasDataset(json_data_flat, mode="masked_prediction")

    # Tokenizers (local or online per RUN_ONLINE)
    if app_config:
        AHB_tokenizer = BertTokenizer.from_pretrained(app_config.BERT_NAME, local_files_only=app_config.use_local_files_only())
        tapas_tokenizer = AutoTokenizer.from_pretrained(app_config.TAPAS_NAME, local_files_only=app_config.use_local_files_only())
    else:
        AHB_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tapas_tokenizer = AutoTokenizer.from_pretrained("google/tapas-base-masklm")
    tokenizers = [AHB_tokenizer, tapas_tokenizer]

    # Collators
    AHB_collator = CollatorForMaskedPrediction(tokenizer=tokenizers[0])
    tapas_collator = CollatorForMaskedPrediction(tokenizer=tokenizers[1])
    collators = [AHB_collator, tapas_collator]

    # Load models based on type
    if args.model == 'baselines':
        models = load_baseline_models(tokenizers[0])
        evaluate_baselines(
            bert_dataset, haetae_dataset, tapas_dataset, navi_dataset, models, tokenizers, collators, args.domain,
            cell_level_eval=args.cell_level_eval,
            cell_level_thresholds=cell_level_thresholds,
        )
        
    elif args.model == 'ablations':
        models = load_ablation_models(tokenizers[0])
        evaluate_ablations(
            navi_dataset, models, tokenizers[0], collators[0], args.domain,
            cell_level_eval=args.cell_level_eval,
            cell_level_thresholds=cell_level_thresholds,
        )

    elif args.model == 'prototyping':
        models = load_prototyping_models(tokenizers[0], args.domain)
        evaluate_ablations(
            navi_dataset, models, tokenizers[0], collators[0], args.domain,
            cell_level_eval=args.cell_level_eval,
            cell_level_thresholds=cell_level_thresholds,
        )

    elif args.model == 'b200':
        models = load_b200_models(tokenizers[0])
        evaluate_baselines(
            bert_dataset, haetae_dataset, tapas_dataset, navi_dataset, models, tokenizers, collators, args.domain,
            cell_level_eval=args.cell_level_eval,
            cell_level_thresholds=cell_level_thresholds,
        )
        
    elif args.model == 'hyperparams':
        results = evaluate_hyperparams(
            navi_dataset, tokenizers[0], collators[0], args.domain,
            max_loaded_variants=args.max_loaded_variants,
        )
        _save_mp_hyperparam_json(results, args.domain, args.run_id)
        print(f"\n=== Summary for {args.domain} Hyperparameter Evaluation ===")
        for model_name, metrics in results.items():
            print(f"{model_name}: Header={metrics['header_accuracy']:.4f}, Value={metrics['value_accuracy']:.4f}")
    
    elif args.model == 'ablation_variants':
        results = evaluate_ablation_variants(navi_dataset, tokenizers[0], collators[0], args.domain)
        
        # Print summary of results
        print(f"\n=== Summary for {args.domain} Ablation Variants Evaluation ===")
        print("\nValue Prediction Accuracy (epoch=5):")
        for model_name, metrics in results.items():
            print(f"{model_name}: {metrics['value_accuracy']:.4f}")
    
    elif args.model == 'tau_align_ethresh':
        results = evaluate_tau_align_ethresh(navi_dataset, tokenizers[0], collators[0], args.domain)
        
        # Print summary of results
        print(f"\n=== Summary for {args.domain} Tau/Align/Ethresh Variants Evaluation ===")
        for model_name, metrics in results.items():
            print(f"{model_name}: Header={metrics['header_accuracy']:.4f}, Value={metrics['value_accuracy']:.4f}")
    
    elif args.model == 'hyperparam_sensitivity':
        results = evaluate_hyperparam_sensitivity(
            navi_dataset, tokenizers[0], collators[0], args.domain,
            max_loaded_variants=args.max_loaded_variants,
        )
        _save_mp_hyperparam_json(results, args.domain, args.run_id)
        print(f"\n=== Summary for {args.domain} Hyperparameter Sensitivity Analysis ===")
        print("\nHeader Prediction Accuracy (epoch=1):")
        for model_name, metrics in results.items():
            print(f"{model_name}: {metrics['header_accuracy']:.4f}")
        print("\nValue Prediction Accuracy (epoch=5):")
        for model_name, metrics in results.items():
            print(f"{model_name}: {metrics['value_accuracy']:.4f}")

    elif args.model == 'icml_entropy_ablation':
        # Note: this mode always evaluates both Movie + Product.
        from experiments.icml_entropy_ablation import run_icml_entropy_ablation

        run_icml_entropy_ablation(
            max_samples=1000,
            imp_runs=5,
            xgb_runs=5,
            run_id=args.run_id,
        )

    elif args.model == 'cm2':
        from experiments.downstream_tasks.cm2_utils import (
            default_cm2_movie_ckpt,
            default_cm2_product_ckpt,
            load_cm2_mask_model,
        )

        ckpt = args.cm2_checkpoint
        if not ckpt:
            ckpt = default_cm2_movie_ckpt() if args.domain == "Movie" else default_cm2_product_ckpt()
        print(f"Loading CM2 checkpoint: {ckpt}")
        cm2_model = load_cm2_mask_model(ckpt, device)

        if args.cm2_mode == 'naive_tokens':
            from experiments.masked_prediction.cm2_naive_token_mlm import (
                evaluate_cm2_naive_token_mlm,
                print_cm2_naive_token_mlm_disclaimer,
            )

            print_cm2_naive_token_mlm_disclaimer()
            evaluate_cm2_naive_token_mlm(
                json_data_flat,
                cm2_model,
                table_flag=args.cm2_table_flag,
                base_seed=42,
                include_column_name_tokens=args.cm2_naive_include_headers,
            )
        else:
            from experiments.masked_prediction.cm2_mp_utils import evaluate_cm2_like_baselines

            evaluate_cm2_like_baselines(
                json_data_flat,
                cm2_model,
                table_flag=args.cm2_table_flag,
                num_abs_tol=args.cm2_num_tol,
                cat_cos_min=args.cm2_cat_cos,
            )
        del cm2_model
        clear_memory()

if __name__ == "__main__":
    main()