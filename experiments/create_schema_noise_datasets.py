#!/usr/bin/env python3
"""
Create variant datasets with schema perturbations for schema noise robustness experiments.
"""

import json
import os
import argparse
from pathlib import Path
from experiments.experiment_utils import load_data
from experiments.schema_perturbations import (
    apply_synonym_replacement,
    apply_header_typos,
    apply_column_reordering
)


def load_synonym_map(synonym_map_path: str) -> dict:
    """Load synonym map from JSON file."""
    with open(synonym_map_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_data(data: list, output_path: str):
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f"✓ Saved {len(data)} rows to {output_path}")


def create_variant_datasets(domain: str, synonym_map_path: str, random_seed: int = 42):
    """
    Create variant datasets for a given domain.
    
    Args:
        domain: Domain name ('Movie' or 'Product')
        synonym_map_path: Path to synonym map JSON file
        random_seed: Random seed for reproducibility
    """
    domain_lower = domain.lower()
    
    # Load synonym map
    synonym_map = load_synonym_map(synonym_map_path)
    
    # Define input paths
    base_dir = f"data/cleaned/{domain}/test"
    cls_input_path = f"{base_dir}/WDC_{domain_lower}_for_cls.jsonl"
    mp_input_path = f"{base_dir}/WDC_{domain_lower}_for_mp.jsonl"
    
    # Check if input files exist
    if not os.path.exists(cls_input_path):
        print(f"⚠️  Warning: {cls_input_path} not found, skipping classification dataset")
        cls_data = None
    else:
        print(f"\n📂 Loading classification data from: {cls_input_path}")
        cls_data = load_data(cls_input_path)
        print(f"   Loaded {len(cls_data)} rows")
    
    if not os.path.exists(mp_input_path):
        print(f"⚠️  Warning: {mp_input_path} not found, skipping masked prediction dataset")
        mp_data = None
    else:
        print(f"\n📂 Loading masked prediction data from: {mp_input_path}")
        mp_data = load_data(mp_input_path)
        print(f"   Loaded {len(mp_data)} rows")
    
    # Process classification dataset
    if cls_data:
        print(f"\n{'='*60}")
        print(f"Processing {domain} Classification Dataset")
        print(f"{'='*60}")
        
        # 1. Synonym replacement
        print("\n1. Applying synonym replacement...")
        cls_synonym = apply_synonym_replacement(
            cls_data, synonym_map, domain_lower, 
            sample_ratio=0.5, random_seed=random_seed
        )
        save_data(cls_synonym, f"{base_dir}/WDC_{domain_lower}_for_cls_synonym.jsonl")
        
        # 2. Header typos
        print("\n2. Applying header typos...")
        cls_typo = apply_header_typos(
            cls_data, domain_lower, 
            sample_ratio=0.5, random_seed=random_seed
        )
        save_data(cls_typo, f"{base_dir}/WDC_{domain_lower}_for_cls_typo.jsonl")
        
        # 3. Column reordering
        print("\n3. Applying column reordering...")
        cls_reordered = apply_column_reordering(cls_data, random_seed=random_seed)
        save_data(cls_reordered, f"{base_dir}/WDC_{domain_lower}_for_cls_reordered.jsonl")
    
    # Process masked prediction dataset
    if mp_data:
        print(f"\n{'='*60}")
        print(f"Processing {domain} Masked Prediction Dataset")
        print(f"{'='*60}")
        
        # 1. Synonym replacement
        print("\n1. Applying synonym replacement...")
        mp_synonym = apply_synonym_replacement(
            mp_data, synonym_map, domain_lower, 
            sample_ratio=0.5, random_seed=random_seed
        )
        save_data(mp_synonym, f"{base_dir}/WDC_{domain_lower}_for_mp_synonym.jsonl")
        
        # 2. Header typos
        print("\n2. Applying header typos...")
        mp_typo = apply_header_typos(
            mp_data, domain_lower, 
            sample_ratio=0.5, random_seed=random_seed
        )
        save_data(mp_typo, f"{base_dir}/WDC_{domain_lower}_for_mp_typo.jsonl")
        
        # 3. Column reordering
        print("\n3. Applying column reordering...")
        mp_reordered = apply_column_reordering(mp_data, random_seed=random_seed)
        save_data(mp_reordered, f"{base_dir}/WDC_{domain_lower}_for_mp_reordered.jsonl")
    
    print(f"\n✅ Completed creating variant datasets for {domain}")


def main():
    parser = argparse.ArgumentParser(
        description='Create variant datasets with schema perturbations'
    )
    parser.add_argument(
        '--domain', 
        choices=['Movie', 'Product', 'Both'],
        default='Both',
        help='Domain to process (default: Both)'
    )
    parser.add_argument(
        '--synonym_map',
        type=str,
        default='artifacts/schema_noise/synonym_map.json',
        help='Path to synonym map JSON file'
    )
    parser.add_argument(
        '--random_seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    if args.domain == 'Both':
        domains = ['Movie', 'Product']
    else:
        domains = [args.domain]
    
    for domain in domains:
        print(f"\n{'='*80}")
        print(f"Creating Variant Datasets for {domain}")
        print(f"{'='*80}")
        create_variant_datasets(domain, args.synonym_map, args.random_seed)
    
    print(f"\n{'='*80}")
    print("All variant datasets created successfully!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

