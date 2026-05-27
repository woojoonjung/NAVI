#!/usr/bin/env python3
"""
Unified Preprocessing Pipeline for NAVI Dataset

This script follows the exact 5-step workflow:
1. Resize product dataset
2. Flatten both movie and product datasets
3. Create heldout datasets for masked prediction and classification
4. Remove heldout rows from training datasets
5. Clean the remaining training datasets

Usage:
    python dataset/preprocess.py

Raw inputs default to data/raw/Movie_top100 and data/raw/Product_top100.
"""

import json
import os
import glob
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from transformers import BertTokenizer
import unicodedata
import random

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL (JSON Lines) data from a file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data: List[Dict[str, Any]], output_path: str) -> None:
    """Save data to JSONL file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def preprocess_text(text):
    """Clean and preprocess text."""
    if isinstance(text, str):
        text = text.lower().strip()
    return str(text)

def flatten_json(json_obj, parent_key='', sep='.'):
    """Recursively flattens a nested JSON object."""
    flattened = {}
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            full_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, (dict, list)):
                flattened.update(flatten_json(value, full_key, sep))
            else:
                flattened[full_key] = preprocess_text(value)
    elif isinstance(json_obj, list):
        for idx, item in enumerate(json_obj):
            full_key = f"{parent_key}{sep}{idx}" if parent_key else str(idx)
            if isinstance(item, (dict, list)):
                flattened.update(flatten_json(item, full_key, sep))
            else:
                flattened[full_key] = preprocess_text(json_obj)
    else:
        flattened[parent_key] = preprocess_text(json_obj)
    return flattened

def preprocess_jsonl_data(jsonl_data):
    """Preprocesses JSONL data by flattening nested structures."""
    return [flatten_json(row) for row in jsonl_data]

# ============================================================================
# STEP 1: RESIZE PRODUCT DATASET
# ============================================================================

def resize_product_dataset(input_dir: str, max_rows: int = 5000) -> None:
    """Resize product dataset to specified number of rows."""
    print("="*80)
    print("📏 STEP 1: Resizing Product Dataset")
    print("="*80)
    print(f"Resizing Product dataset to {max_rows} rows...")
    
    jsonl_files = glob.glob(os.path.join(input_dir, "*.jsonl")) + glob.glob(os.path.join(input_dir, "*.json"))
    
    for file_path in sorted(jsonl_files):
        filename = os.path.basename(file_path)
        original_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        try:
            # Read and filter the file
            filtered_rows = []
            original_count = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    original_count += 1
                    if len(filtered_rows) < max_rows:
                        filtered_rows.append(line.strip())
            
            # Write filtered data back to original file
            with open(file_path, 'w', encoding='utf-8') as f:
                for row in filtered_rows:
                    f.write(row + '\n')
            
            new_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  {filename}: {original_count} → {len(filtered_rows)} rows | {original_size_mb:.1f}MB → {new_size_mb:.1f}MB")
            
        except Exception as e:
            print(f"  ERROR processing {filename}: {str(e)}")

# ============================================================================
# STEP 2: FLATTEN BOTH MOVIE AND PRODUCT DATASETS
# ============================================================================

def flatten_datasets(movie_dir: str, product_dir: str) -> None:
    """Flatten both movie and product datasets."""
    print("\n" + "="*80)
    print("🔄 STEP 2: Flattening Movie and Product Datasets")
    print("="*80)
    
    # Flatten Movie dataset
    print("Flattening Movie dataset...")
    movie_files = glob.glob(os.path.join(movie_dir, "*.jsonl")) + glob.glob(os.path.join(movie_dir, "*.json"))
    
    for file_path in sorted(movie_files):
        filename = os.path.basename(file_path)
        print(f"  Processing {filename}...")
        
        try:
            # Load raw data
            raw_data = load_jsonl(file_path)
            print(f"    Loaded {len(raw_data)} rows")
            
            # Flatten data
            flattened_data = preprocess_jsonl_data(raw_data)
            print(f"    Flattened to {len(flattened_data)} rows")
            
            # Save flattened data back to original file
            save_jsonl(flattened_data, file_path)
            print(f"    ✅ Flattened data saved to {filename}")
            
        except Exception as e:
            print(f"    ❌ Error processing {filename}: {str(e)}")
    
    # Flatten Product dataset
    print("\nFlattening Product dataset...")
    product_files = glob.glob(os.path.join(product_dir, "*.jsonl")) + glob.glob(os.path.join(product_dir, "*.json"))
    
    for file_path in sorted(product_files):
        filename = os.path.basename(file_path)
        print(f"  Processing {filename}...")
        
        try:
            # Load raw data
            raw_data = load_jsonl(file_path)
            print(f"    Loaded {len(raw_data)} rows")
            
            # Flatten data
            flattened_data = preprocess_jsonl_data(raw_data)
            print(f"    Flattened to {len(flattened_data)} rows")
            
            # Save flattened data back to original file
            save_jsonl(flattened_data, file_path)
            print(f"    ✅ Flattened data saved to {filename}")
            
        except Exception as e:
            print(f"    ❌ Error processing {filename}: {str(e)}")

# ============================================================================
# NEW STEP 1 & 2 IMPLEMENTATIONS (STRATIFIED RESIZE + DIR-BASED FLATTEN)
# ============================================================================

def resize_product_dataset_stratified(
    raw_product_dir: str,
    target_total_rows: int = 480_817,
    seed: int = 42,
    flattened_product_dir: str = None,
) -> None:
    """
    Resize Product tables so that the total number of rows across all tables
    approximately matches `target_total_rows`, using a global sampling ratio.

    The function:
    - Scans all tables under `raw_product_dir` and counts rows per file.
    - Computes a global ratio = target_total_rows / current_total_rows (capped at 1.0).
    - For each table, samples floor(row_count * ratio) rows (at least 1 if row_count > 0).
    - Rewrites each table file in-place in `raw_product_dir` with the sampled rows.
    """
    print("=" * 80)
    print("📏 STEP 1: Stratified Resizing of Product Dataset")
    print("=" * 80)
    
    # Skip if flattened directory exists (indicates resize was already done)
    if flattened_product_dir and os.path.exists(flattened_product_dir):
        flattened_files = glob.glob(os.path.join(flattened_product_dir, "*.jsonl")) + glob.glob(
            os.path.join(flattened_product_dir, "*.json")
        )
        if flattened_files:
            print(f"⏭️  Skipping resizing: Flattened Product files already exist in {flattened_product_dir}")
            return

    jsonl_files = glob.glob(os.path.join(raw_product_dir, "*.jsonl")) + glob.glob(
        os.path.join(raw_product_dir, "*.json")
    )
    jsonl_files = sorted(jsonl_files)

    if not jsonl_files:
        print(f"❌ No Product files found in {raw_product_dir}")
        return

    # Count rows per file
    file_row_counts: List[Tuple[str, int]] = []
    total_rows = 0
    print("Counting rows per Product table...")
    for file_path in jsonl_files:
        filename = os.path.basename(file_path)
        row_count = 0
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for _ in f:
                    row_count += 1
        except Exception as e:
            print(f"  ❌ Error counting rows in {filename}: {e}")
            continue

        file_row_counts.append((file_path, row_count))
        total_rows += row_count
        print(f"  {filename}: {row_count} rows")

    if total_rows == 0:
        print("❌ Total Product rows = 0. Skipping resizing.")
        return

    ratio = target_total_rows / total_rows
    if ratio >= 1.0:
        print(
            f"ℹ️ Current total Product rows ({total_rows}) "
            f"is less than or equal to target ({target_total_rows}). "
            "Skipping downsampling (ratio >= 1.0)."
        )
        return

    print(f"\nGlobal sampling ratio: {ratio:.6f}")
    rng = random.Random(seed)

    # Sample rows per file and rewrite in-place
    total_sampled = 0
    for file_path, row_count in file_row_counts:
        if row_count == 0:
            continue

        filename = os.path.basename(file_path)
        original_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        sample_size = int(row_count * ratio)
        if sample_size <= 0:
            sample_size = 1
        if sample_size > row_count:
            sample_size = row_count

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                rows = [line.rstrip("\n") for line in f]

            # Deterministic per-file RNG
            file_rng = random.Random(rng.randint(0, 2**31 - 1))
            sampled_rows = file_rng.sample(rows, sample_size)

            with open(file_path, "w", encoding="utf-8") as f:
                for row in sampled_rows:
                    f.write(row + "\n")

            new_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            total_sampled += sample_size
            print(
                f"  {filename}: {row_count} → {sample_size} rows | "
                f"{original_size_mb:.1f}MB → {new_size_mb:.1f}MB"
            )
        except Exception as e:
            print(f"  ❌ Error sampling {filename}: {e}")

    print(
        f"\n✅ Stratified resizing complete. "
        f"Target total rows: {target_total_rows}, "
        f"sampled total rows: {total_sampled}"
    )


def flatten_datasets_with_dirs(
    raw_movie_dir: str,
    raw_product_dir: str,
    out_movie_dir: str,
    out_product_dir: str,
) -> None:
    """Flatten both Movie and Product datasets from raw → flattened directories."""
    print("\n" + "=" * 80)
    print("🔄 STEP 2: Flattening Movie and Product Datasets")
    print("=" * 80)

    # Check if flattened files already exist
    movie_files_exist = False
    product_files_exist = False
    
    if os.path.exists(out_movie_dir):
        existing_movie_files = glob.glob(os.path.join(out_movie_dir, "*.jsonl")) + glob.glob(
            os.path.join(out_movie_dir, "*.json")
        )
        if existing_movie_files:
            movie_files_exist = True
    
    if os.path.exists(out_product_dir):
        existing_product_files = glob.glob(os.path.join(out_product_dir, "*.jsonl")) + glob.glob(
            os.path.join(out_product_dir, "*.json")
        )
        if existing_product_files:
            product_files_exist = True
    
    if movie_files_exist and product_files_exist:
        print(f"⏭️  Skipping flattening: Flattened files already exist in {out_movie_dir} and {out_product_dir}")
        return

    # Flatten Movie dataset
    if not movie_files_exist:
        print("Flattening Movie dataset...")
    else:
        print(f"⏭️  Skipping Movie flattening: Files already exist in {out_movie_dir}")
    
    movie_files = glob.glob(os.path.join(raw_movie_dir, "*.jsonl")) + glob.glob(
        os.path.join(raw_movie_dir, "*.json")
    )

    for file_path in sorted(movie_files):
        filename = os.path.basename(file_path)
        out_path = os.path.join(out_movie_dir, filename)
        
        # Skip if this file already exists
        if os.path.exists(out_path):
            continue
            
        print(f"  Processing {filename}...")

        try:
            raw_data = load_jsonl(file_path)
            print(f"    Loaded {len(raw_data)} rows")

            flattened_data = preprocess_jsonl_data(raw_data)
            print(f"    Flattened to {len(flattened_data)} rows")

            save_jsonl(flattened_data, out_path)
            print(f"    ✅ Flattened data saved to {out_path}")

        except Exception as e:
            print(f"    ❌ Error processing {filename}: {str(e)}")

    # Flatten Product dataset
    if not product_files_exist:
        print("\nFlattening Product dataset...")
    else:
        print(f"\n⏭️  Skipping Product flattening: Files already exist in {out_product_dir}")
    product_files = glob.glob(os.path.join(raw_product_dir, "*.jsonl")) + glob.glob(
        os.path.join(raw_product_dir, "*.json")
    )

    for file_path in sorted(product_files):
        filename = os.path.basename(file_path)
        out_path = os.path.join(out_product_dir, filename)
        
        # Skip if this file already exists
        if os.path.exists(out_path):
            continue
            
        print(f"  Processing {filename}...")

        try:
            raw_data = load_jsonl(file_path)
            print(f"    Loaded {len(raw_data)} rows")

            flattened_data = preprocess_jsonl_data(raw_data)
            print(f"    Flattened to {len(flattened_data)} rows")

            save_jsonl(flattened_data, out_path)
            print(f"    ✅ Flattened data saved to {out_path}")

        except Exception as e:
            print(f"    ❌ Error processing {filename}: {str(e)}")

# ============================================================================
# STEP 3: CREATE HELDOUT DATASETS
# ============================================================================

def unify_genre_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """Unify all genre-related keys to 'genres' field."""
    unified_data = {}
    genre_values = []
    
    for key, value in data.items():
        if key == "genre":
            if value and value != "None" and value != "nan":
                genre_values.append(str(value))
        elif key.startswith("genre.") or key.startswith("url.genre."):
            if value and value != "None" and value != "nan":
                genre_values.append(str(value))
        else:
            unified_data[key] = value
    
    if genre_values:
        seen = set()
        unique_genres = []
        for genre in genre_values:
            if genre not in seen:
                seen.add(genre)
                unique_genres.append(genre)
        unified_data["genres"] = ", ".join(unique_genres)
    else:
        unified_data["genres"] = ""
    
    return unified_data

def extract_category_from_url_category(url_category: str) -> str:
    """Extract category from url.category."""
    if not url_category or url_category == "None" or url_category == "nan":
        return ""
    
    parts = [part.strip() for part in url_category.split(">")]
    if len(parts) >= 2:
        return parts[1]
    else:
        return ""

def unify_category_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """Unify category-related keys to 'category' field."""
    unified_data = {}
    category_value = ""
    
    for key, value in data.items():
        if key == "category":
            category_value = str(value)
        elif key == "url.category":
            extracted_category = extract_category_from_url_category(value)
            if extracted_category:
                category_value = extracted_category
        else:
            unified_data[key] = value
    
    unified_data["category"] = category_value
    return unified_data

def extract_heldout_data(input_directory: str, domain: str, start_row_id: int = 450, end_row_id: int = 459) -> List[Dict[str, Any]]:
    """Extract rows with row_id between start_row_id and end_row_id from each JSONL file."""
    
    # Set file pattern based on domain
    if domain == 'movie':
        file_pattern = "Movie_*"
    elif domain == 'product':
        file_pattern = "Product_*"
    else:
        raise ValueError(f"Unsupported domain: {domain}. Must be 'movie' or 'product'")
    
    # Get all JSON/JSONL files in the directory
    json_files = glob.glob(os.path.join(input_directory, file_pattern + ".json"))
    jsonl_files = glob.glob(os.path.join(input_directory, file_pattern + ".jsonl"))
    all_files = json_files + jsonl_files
    
    print(f"  Found {len(all_files)} {domain} files")
    print(f"  Extracting rows with row_id {start_row_id} to {end_row_id}")
    
    extracted_rows = []
    
    for file_path in sorted(all_files):
        filename = os.path.basename(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as infile:
                for line_num, line in enumerate(infile, 1):
                    try:
                        row = json.loads(line.strip())
                        row_id = int(row.get('row_id', 0))
                        
                        # Check if row_id is in the target range
                        if start_row_id <= row_id <= end_row_id:
                            extracted_rows.append(row)
                            
                    except json.JSONDecodeError as e:
                        print(f"    Warning: Invalid JSON on line {line_num} in {filename}: {e}")
                        continue
                    except (ValueError, TypeError) as e:
                        print(f"    Warning: Invalid row_id on line {line_num} in {filename}: {e}")
                        continue
            
        except Exception as e:
            print(f"    ERROR processing {filename}: {str(e)}")
    
    print(f"  Extracted {len(extracted_rows)} rows")
    return extracted_rows

def create_heldout_datasets(movie_dir: str, product_dir: str) -> None:
    """Create heldout datasets for masked prediction and classification."""
    print("\n" + "="*80)
    print("📤 STEP 3: Creating Heldout Datasets")
    print("="*80)
    
    # Create Movie heldout datasets
    print("Creating Movie heldout datasets...")
    
    # MP dataset: rows 450-459 (10 rows per table for masked prediction)
    movie_mp_data = extract_heldout_data(movie_dir, 'movie', 450, 459)
    movie_mp_output = "data/WDC_movie_for_mp.jsonl"
    save_jsonl(movie_mp_data, movie_mp_output)
    print(f"  ✅ Movie MP dataset saved: {len(movie_mp_data)} rows")
    
    # CLS dataset: rows 0-449 (450 rows per table for classification)
    movie_cls_data = extract_heldout_data(movie_dir, 'movie', 0, 449)
    movie_cls_data = [unify_genre_keys(row) for row in movie_cls_data]
    movie_cls_data = [row for row in movie_cls_data if "genres" in row.keys() and row["genres"] and row["genres"] != "None"]
    movie_cls_output = "data/WDC_movie_for_cls.jsonl"
    save_jsonl(movie_cls_data, movie_cls_output)
    print(f"  ✅ Movie CLS dataset saved: {len(movie_cls_data)} rows")
    
    # Create Product heldout datasets
    print("\nCreating Product heldout datasets...")
    
    # MP dataset: rows 450-459 (10 rows per table for masked prediction)
    product_mp_data = extract_heldout_data(product_dir, 'product', 450, 459)
    product_mp_output = "data/WDC_product_for_mp.jsonl"
    save_jsonl(product_mp_data, product_mp_output)
    print(f"  ✅ Product MP dataset saved: {len(product_mp_data)} rows")
    
    # CLS dataset: rows 0-449 (450 rows per table for classification)
    product_cls_data = extract_heldout_data(product_dir, 'product', 0, 449)
    product_cls_data = [unify_category_keys(row) for row in product_cls_data]
    product_cls_data = [row for row in product_cls_data if "category" in row.keys() and row["category"] and row["category"] != "None"]
    product_cls_output = "data/WDC_product_for_cls.jsonl"
    save_jsonl(product_cls_data, product_cls_output)
    print(f"  ✅ Product CLS dataset saved: {len(product_cls_data)} rows")

# ============================================================================
# STEP 4: REMOVE HELDOUT ROWS FROM TRAINING DATASETS
# ============================================================================

def remove_heldout_rows_from_datasets(movie_dir: str, product_dir: str, min_row_id: int = 460) -> None:
    """Remove rows with row_id <= min_row_id from training datasets."""
    print("\n" + "="*80)
    print("🗑️  STEP 4: Removing Heldout Rows from Training Datasets")
    print("="*80)
    
    # Remove from Movie dataset
    print("Removing heldout rows from Movie dataset...")
    movie_files = glob.glob(os.path.join(movie_dir, "*.jsonl")) + glob.glob(os.path.join(movie_dir, "*.json"))
    
    for file_path in sorted(movie_files):
        filename = os.path.basename(file_path)
        original_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        try:
            filtered_rows = []
            original_count = 0
            removed_count = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    original_count += 1
                    try:
                        row = json.loads(line.strip())
                        row_id = int(row.get('row_id', 0))
                        
                        if row_id > min_row_id:
                            filtered_rows.append(line.strip())
                        else:
                            removed_count += 1
                            
                    except json.JSONDecodeError as e:
                        print(f"    Warning: Invalid JSON on line {line_num} in {filename}: {e}")
                        filtered_rows.append(line.strip())
                    except (ValueError, TypeError) as e:
                        print(f"    Warning: Invalid row_id on line {line_num} in {filename}: {e}")
                        filtered_rows.append(line.strip())
            
            # Write filtered data back to original file
            with open(file_path, 'w', encoding='utf-8') as f:
                for row in filtered_rows:
                    f.write(row + '\n')
            
            new_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  {filename}: {original_count} → {len(filtered_rows)} rows | {removed_count} removed | {original_size_mb:.1f}MB → {new_size_mb:.1f}MB")
            
        except Exception as e:
            print(f"  ERROR processing {filename}: {str(e)}")
    
    # Remove from Product dataset
    print("\nRemoving heldout rows from Product dataset...")
    product_files = glob.glob(os.path.join(product_dir, "*.jsonl")) + glob.glob(os.path.join(product_dir, "*.json"))
    
    for file_path in sorted(product_files):
        filename = os.path.basename(file_path)
        original_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        try:
            filtered_rows = []
            original_count = 0
            removed_count = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    original_count += 1
                    try:
                        row = json.loads(line.strip())
                        row_id = int(row.get('row_id', 0))
                        
                        if row_id > min_row_id:
                            filtered_rows.append(line.strip())
                        else:
                            removed_count += 1
                            
                    except json.JSONDecodeError as e:
                        print(f"    Warning: Invalid JSON on line {line_num} in {filename}: {e}")
                        filtered_rows.append(line.strip())
                    except (ValueError, TypeError) as e:
                        print(f"    Warning: Invalid row_id on line {line_num} in {filename}: {e}")
                        filtered_rows.append(line.strip())
            
            # Write filtered data back to original file
            with open(file_path, 'w', encoding='utf-8') as f:
                for row in filtered_rows:
                    f.write(row + '\n')
            
            new_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  {filename}: {original_count} → {len(filtered_rows)} rows | {removed_count} removed | {original_size_mb:.1f}MB → {new_size_mb:.1f}MB")
            
        except Exception as e:
            print(f"  ERROR processing {filename}: {str(e)}")

# ============================================================================
# STEP 5: CLEAN TRAINING DATASETS
# ============================================================================

def detect_language_simple(text, min_confidence=0.8):
    """Simple language detection based on character analysis."""
    if not isinstance(text, str) or not text.strip():
        return 'unknown'
    
    text_clean = re.sub(r'\s+', '', text)
    if len(text_clean) < 3:
        return 'unknown'
    
    # Count different character types
    ascii_letters = 0
    ascii_digits = 0
    cyrillic_chars = 0
    cjk_chars = 0
    latin_extended = 0
    
    for char in text_clean:
        if char.isascii() and char.isalpha():
            ascii_letters += 1
        elif char.isascii() and char.isdigit():
            ascii_digits += 1
        elif '\u0400' <= char <= '\u04FF':  # Cyrillic
            cyrillic_chars += 1
        elif '\u4e00' <= char <= '\u9fff' or '\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff':  # CJK
            cjk_chars += 1
        elif '\u0100' <= char <= '\u017f' or '\u0180' <= char <= '\u024f':  # Latin Extended
            latin_extended += 1
    
    total_chars = len(text_clean)
    ascii_ratio = (ascii_letters + ascii_digits) / total_chars
    cyrillic_ratio = cyrillic_chars / total_chars
    cjk_ratio = cjk_chars / total_chars
    latin_extended_ratio = latin_extended / total_chars
    
    # Determine language based on character ratios
    if ascii_ratio >= min_confidence:
        return 'en'
    elif cyrillic_ratio >= min_confidence:
        return 'non-en'
    elif cjk_ratio >= min_confidence:
        return 'non-en'
    elif latin_extended_ratio >= 0.3:
        return 'non-en'
    elif ascii_ratio >= 0.5:
        return 'en'
    else:
        return 'non-en'

def is_english_table(table_dict, min_english_ratio=0.7, min_text_fields=2):
    """Check if a table contains primarily English content."""
    text_fields = []
    
    for field_name, field_value in table_dict.items():
        if isinstance(field_value, str) and field_value.strip():
            if (len(field_value.strip()) > 10 and 
                not field_value.startswith('http') and 
                not field_value.replace('.', '').replace(',', '').replace('-', '').replace(':', '').isdigit() and
                not re.match(r'^\d{4}-\d{2}-\d{2}', field_value.strip())):
                text_fields.append(field_value)
    
    if len(text_fields) < min_text_fields:
        return True
    
    english_fields = 0
    for field in text_fields:
        lang = detect_language_simple(field)
        if lang == 'en':
            english_fields += 1
    
    english_ratio = english_fields / len(text_fields)
    return english_ratio >= min_english_ratio

def is_bert_tokenizable(text, tokenizer, unk_threshold=0.2):
    """Check if text can be properly tokenized by BERT."""
    if not isinstance(text, str) or not text.strip():
        return True
    
    tokens = tokenizer.tokenize(text)
    if not tokens:
        return True
    
    unk_count = sum(1 for token in tokens if token == tokenizer.unk_token)
    unk_ratio = unk_count / len(tokens)
    
    return unk_ratio <= unk_threshold

def is_bert_compatible_table(table_dict, tokenizer, unk_threshold=0.3, min_text_fields=1):
    """Check if a table contains text that can be properly tokenized by BERT."""
    text_fields = []
    
    for field_name, field_value in table_dict.items():
        if isinstance(field_value, str) and field_value.strip():
            if (len(field_value.strip()) > 5 and 
                not field_value.startswith('http') and 
                not field_value.replace('.', '').replace(',', '').replace('-', '').isdigit()):
                text_fields.append(field_value)
    
    if len(text_fields) < min_text_fields:
        return True
    
    tokenizable_fields = sum(1 for field in text_fields if is_bert_tokenizable(field, tokenizer, unk_threshold))
    tokenizable_ratio = tokenizable_fields / len(text_fields)
    
    return tokenizable_ratio >= 0.5

def handle_indexed_fields(table_dict, max_indexed_fields=3):
    """Handle indexed fields by sampling representative ones."""
    processed_table = {}
    field_groups = {}
    
    for field_name, field_value in table_dict.items():
        if '.' in field_name and any(char.isdigit() for char in field_name):
            field_type = field_name.split('.')[0]
            
            has_index = False
            for part in field_name.split('.')[1:]:
                if part.isdigit():
                    has_index = True
                    break
            
            if has_index:
                if field_type not in field_groups:
                    field_groups[field_type] = []
                field_groups[field_type].append((field_name, field_value))
            else:
                processed_table[field_name] = field_value
        else:
            processed_table[field_name] = field_value
    
    # Sample indexed fields
    for field_type, fields in field_groups.items():
        if len(fields) > max_indexed_fields:
            sampled = fields[:max_indexed_fields]
            for field_name, field_value in sampled:
                processed_table[field_name] = field_value
        else:
            for field_name, field_value in fields:
                processed_table[field_name] = field_value
    
    return processed_table

def truncate_long_fields(table_dict, max_tokens_per_field=20, tokenizer=None):
    """Truncate fields that exceed token limit based on actual tokenization."""
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    processed_table = {}
    
    for field_name, field_value in table_dict.items():
        estimated_tokens = estimate_token_count(field_name, field_value, tokenizer)
        
        if estimated_tokens > max_tokens_per_field and isinstance(field_value, str):
            truncated_value = truncate_value_to_token_limit(
                field_name, field_value, tokenizer, max_tokens_per_field
            )
            processed_table[field_name] = truncated_value
        else:
            processed_table[field_name] = field_value
    
    return processed_table

def truncate_value_to_token_limit(field_name, field_value, tokenizer, max_tokens_per_field):
    """Truncate field value to fit within token limit."""
    name_tokens = tokenizer.tokenize(field_name)
    separator_tokens = tokenizer.tokenize(" : ")
    reserved_tokens = len(name_tokens) + len(separator_tokens)
    
    available_tokens = max_tokens_per_field - reserved_tokens
    
    if available_tokens <= 0:
        return "..."
    
    value_tokens = tokenizer.tokenize(str(field_value))
    
    if len(value_tokens) <= available_tokens:
        return field_value
    else:
        truncated_tokens = value_tokens[:available_tokens]
        truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
        return truncated_text + "..."

def estimate_token_count(field_name, field_value, tokenizer):
    """Estimate token count for a field."""
    text = f"{field_name} : {field_value}"
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def select_fields_for_tokenization(table_dict, tokenizer, max_tokens=400):
    """Select fields that will fit within token limit."""
    selected_fields = {}
    current_tokens = 0
    
    field_items = list(table_dict.items())
    field_items.sort(key=lambda x: len(str(x[1])))
    
    for field_name, field_value in field_items:
        estimated_tokens = estimate_token_count(field_name, field_value, tokenizer)
        
        if current_tokens + estimated_tokens <= max_tokens:
            selected_fields[field_name] = field_value
            current_tokens += estimated_tokens
        else:
            break
    
    return selected_fields

def clean_table_data(json_data, tokenizer_name="bert-base-uncased", 
                     max_tokens=512, max_indexed_fields=3, max_tokens_per_field=20,
                     skip_non_english=True, skip_non_bert=True, 
                     english_ratio_threshold=0.7, unk_threshold=0.3, min_text_fields=2):
    """Clean table data with language and BERT vocabulary validation."""
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    processed_data = []
    skipped_non_english = 0
    skipped_non_bert = 0
    
    print("🔄 Cleaning table data with language and BERT vocabulary validation...")
    
    for table_id, table_dict in json_data:
        # Skip non-English tables if enabled
        if skip_non_english and not is_english_table(table_dict, english_ratio_threshold, min_text_fields):
            skipped_non_english += 1
            if skipped_non_english % 100 == 0:
                print(f"   Skipped {skipped_non_english} non-English tables...")
            continue
        
        # Skip tables that can't be properly tokenized by BERT
        if skip_non_bert and not is_bert_compatible_table(table_dict, tokenizer, unk_threshold, min_text_fields):
            skipped_non_bert += 1
            if skipped_non_bert % 100 == 0:
                print(f"   Skipped {skipped_non_bert} non-BERT-compatible tables...")
            continue
        
        # Handle indexed fields
        processed_table = handle_indexed_fields(table_dict, max_indexed_fields)
        
        # Truncate long text fields
        processed_table = truncate_long_fields(processed_table, max_tokens_per_field, tokenizer)
        
        # Select fields that fit in token limit
        processed_table = select_fields_for_tokenization(processed_table, tokenizer, max_tokens)
        
        processed_data.append((table_id, processed_table))
        
        if len(processed_data) % 1000 == 0:
            print(f"   Processed {len(processed_data)} instances...")
    
    print(f"✅ Cleaning complete. Processed {len(processed_data)} instances.")
    if skip_non_english:
        print(f"🚫 Skipped {skipped_non_english} non-English tables.")
    if skip_non_bert:
        print(f"🚫 Skipped {skipped_non_bert} non-BERT-compatible tables.")
    return processed_data


def clean_datasets_with_dirs(
    flat_movie_dir: str,
    flat_product_dir: str,
    out_movie_dir: str,
    out_product_dir: str,
) -> None:
    """
    Clean flattened Movie and Product datasets into dedicated cleaned directories.

    Reads from `flat_*_dir` (flattened tables) and writes cleaned tables to
    `out_*_dir`, preserving filenames.
    """
    print("\n" + "=" * 80)
    print("🧹 STEP 3: Cleaning Flattened Datasets")
    print("=" * 80)

    # Check if cleaned files already exist
    movie_files_exist = False
    product_files_exist = False
    
    if os.path.exists(out_movie_dir):
        existing_movie_files = glob.glob(os.path.join(out_movie_dir, "*.jsonl")) + glob.glob(
            os.path.join(out_movie_dir, "*.json")
        )
        if existing_movie_files:
            movie_files_exist = True
    
    if os.path.exists(out_product_dir):
        existing_product_files = glob.glob(os.path.join(out_product_dir, "*.jsonl")) + glob.glob(
            os.path.join(out_product_dir, "*.json")
        )
        if existing_product_files:
            product_files_exist = True
    
    if movie_files_exist and product_files_exist:
        print(f"⏭️  Skipping cleaning: Cleaned files already exist in {out_movie_dir} and {out_product_dir}")
        return

    # Clean Movie dataset
    if not movie_files_exist:
        print("Cleaning Movie dataset...")
    else:
        print(f"⏭️  Skipping Movie cleaning: Files already exist in {out_movie_dir}")
    
    movie_files = glob.glob(os.path.join(flat_movie_dir, "*.jsonl")) + glob.glob(
        os.path.join(flat_movie_dir, "*.json")
    )

    for file_path in sorted(movie_files):
        filename = os.path.basename(file_path)
        out_path = os.path.join(out_movie_dir, filename)
        
        # Skip if this file already exists
        if os.path.exists(out_path):
            continue
            
        print(f"  Processing {filename}...")

        try:
            data = load_jsonl(file_path)
            print(f"    Loaded {len(data)} rows")

            if len(data) == 0:
                print("    ⚠️  No data found, skipping...")
                continue

            table_data = [(i, row) for i, row in enumerate(data)]

            print("    Applying comprehensive table cleaning...")
            cleaned_table_data = clean_table_data(
                table_data,
                tokenizer_name="bert-base-uncased",
                max_tokens=512,
                max_indexed_fields=3,
                max_tokens_per_field=20,
                skip_non_english=True,
                skip_non_bert=True,
                english_ratio_threshold=0.7,
                unk_threshold=0.3,
                min_text_fields=2,
            )

            cleaned_data = [row for _, row in cleaned_table_data]
            print(f"    Final cleaned data: {len(cleaned_data)} rows")

            save_jsonl(cleaned_data, out_path)
            print(f"    ✅ Cleaned data saved to {out_path}")

        except Exception as e:
            print(f"    ❌ Error processing {filename}: {str(e)}")
            continue

    # Clean Product dataset
    if not product_files_exist:
        print("\nCleaning Product dataset...")
    else:
        print(f"\n⏭️  Skipping Product cleaning: Files already exist in {out_product_dir}")
    product_files = glob.glob(os.path.join(flat_product_dir, "*.jsonl")) + glob.glob(
        os.path.join(flat_product_dir, "*.json")
    )

    for file_path in sorted(product_files):
        filename = os.path.basename(file_path)
        out_path = os.path.join(out_product_dir, filename)
        
        # Skip if this file already exists
        if os.path.exists(out_path):
            continue
            
        print(f"  Processing {filename}...")

        try:
            data = load_jsonl(file_path)
            print(f"    Loaded {len(data)} rows")

            if len(data) == 0:
                print("    ⚠️  No data found, skipping...")
                continue

            table_data = [(i, row) for i, row in enumerate(data)]

            print("    Applying comprehensive table cleaning...")
            cleaned_table_data = clean_table_data(
                table_data,
                tokenizer_name="bert-base-uncased",
                max_tokens=512,
                max_indexed_fields=3,
                max_tokens_per_field=20,
                skip_non_english=True,
                skip_non_bert=True,
                english_ratio_threshold=0.7,
                unk_threshold=0.3,
                min_text_fields=2,
            )

            cleaned_data = [row for _, row in cleaned_table_data]
            print(f"    Final cleaned data: {len(cleaned_data)} rows")

            save_jsonl(cleaned_data, out_path)
            print(f"    ✅ Cleaned data saved to {out_path}")

        except Exception as e:
            print(f"    ❌ Error processing {filename}: {str(e)}")
            continue


def split_cleaned_tables_to_splits(
    clean_movie_dir: str,
    clean_product_dir: str,
    movie_train_dir: str,
    movie_val_dir: str,
    product_train_dir: str,
    product_val_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Split cleaned Movie and Product tables into train/validation/test per table.

    - Train: `train_ratio` (default 0.8)
    - Validation: `val_ratio` (default 0.1)
    - Test: remaining portion (default 0.1)

    Per-table train/validation splits are written to the given directories.
    Test rows are accumulated and returned for heldout WDC file creation.
    """
    print("\n" + "=" * 80)
    print("✂️  STEP 4: Splitting Cleaned Tables into Train/Validation/Test")
    print("=" * 80)

    # Check if train/val splits already exist
    movie_splits_exist = False
    product_splits_exist = False
    
    if os.path.exists(movie_train_dir) and os.path.exists(movie_val_dir):
        movie_train_files = glob.glob(os.path.join(movie_train_dir, "*.jsonl")) + glob.glob(
            os.path.join(movie_train_dir, "*.json")
        )
        movie_val_files = glob.glob(os.path.join(movie_val_dir, "*.jsonl")) + glob.glob(
            os.path.join(movie_val_dir, "*.json")
        )
        if movie_train_files and movie_val_files:
            movie_splits_exist = True
    
    if os.path.exists(product_train_dir) and os.path.exists(product_val_dir):
        product_train_files = glob.glob(os.path.join(product_train_dir, "*.jsonl")) + glob.glob(
            os.path.join(product_train_dir, "*.json")
        )
        product_val_files = glob.glob(os.path.join(product_val_dir, "*.jsonl")) + glob.glob(
            os.path.join(product_val_dir, "*.json")
        )
        if product_train_files and product_val_files:
            product_splits_exist = True
    
    # Check if WDC files exist - if they do, we can skip splitting entirely
    movie_test_dir = "data/cleaned/Movie/test"
    product_test_dir = "data/cleaned/Product/test"
    wdc_files_exist = (
        os.path.exists(os.path.join(movie_test_dir, "WDC_movie_for_mp.jsonl")) and
        os.path.exists(os.path.join(movie_test_dir, "WDC_movie_for_cls.jsonl")) and
        os.path.exists(os.path.join(product_test_dir, "WDC_product_for_mp.jsonl")) and
        os.path.exists(os.path.join(product_test_dir, "WDC_product_for_cls.jsonl"))
    )
    
    if wdc_files_exist and movie_splits_exist and product_splits_exist:
        print(f"⏭️  Skipping splitting: Train/validation splits and WDC test files already exist")
        print(f"    Movie: {movie_train_dir} and {movie_val_dir}")
        print(f"    Product: {product_train_dir} and {product_val_dir}")
        return [], []

    rng = random.Random(seed)

    def _split_domain(clean_dir: str, train_dir: str, val_dir: str, label: str, skip_if_exists: bool = False):
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        files = glob.glob(os.path.join(clean_dir, "*.jsonl")) + glob.glob(
            os.path.join(clean_dir, "*.json")
        )
        files = sorted(files)

        all_test_rows = []
        skipped_count = 0
        processed_count = 0
        
        if skip_if_exists:
            print(f"\nChecking {label} tables (some splits may already exist)...")
        else:
            print(f"\nSplitting {label} tables...")
        
        for file_path in files:
            filename = os.path.basename(file_path)
            out_train_path = os.path.join(train_dir, filename)
            out_val_path = os.path.join(val_dir, filename)
            
            # Skip if train and val files already exist
            if os.path.exists(out_train_path) and os.path.exists(out_val_path):
                skipped_count += 1
                continue
            
            processed_count += 1
                
            rows = load_jsonl(file_path)
            n = len(rows)
            print(f"  {filename}: {n} rows before split")

            if n == 0:
                print("    ⚠️  No rows, skipping.")
                continue

            indices = list(range(n))
            rng.shuffle(indices)

            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            if n_train + n_val > n:
                n_val = max(0, n - n_train)
            n_test = n - n_train - n_val

            train_idx = indices[:n_train]
            val_idx = indices[n_train : n_train + n_val]
            test_idx = indices[n_train + n_val :]

            train_rows = [rows[i] for i in train_idx]
            val_rows = [rows[i] for i in val_idx]
            test_rows = [rows[i] for i in test_idx]

            save_jsonl(train_rows, out_train_path)
            save_jsonl(val_rows, out_val_path)

            all_test_rows.extend(test_rows)

            print(
                f"    Split -> train: {len(train_rows)}, "
                f"validation: {len(val_rows)}, test: {len(test_rows)}"
            )

        if skip_if_exists and processed_count == 0:
            print(f"\n⏭️  All {label} tables already split. Skipped {skipped_count} files.")
        else:
            print(
                f"\n✅ Finished splitting {label} tables. "
                f"Processed: {processed_count}, Skipped: {skipped_count}, "
                f"Accumulated test rows: {len(all_test_rows)}"
            )
        return all_test_rows

    movie_test_rows = _split_domain(
        clean_movie_dir, movie_train_dir, movie_val_dir, label="Movie", skip_if_exists=movie_splits_exist
    )
    product_test_rows = _split_domain(
        clean_product_dir, product_train_dir, product_val_dir, label="Product", skip_if_exists=product_splits_exist
    )

    return movie_test_rows, product_test_rows


def create_heldout_datasets_from_test_rows(
    movie_test_rows,
    product_test_rows,
    out_dir: str,
) -> None:
    """
    Create heldout WDC datasets for MP and CLS from aggregated test rows.

    Outputs (in domain-specific test directories):
      - data/cleaned/Movie/test/WDC_movie_for_mp.jsonl
      - data/cleaned/Movie/test/WDC_movie_for_cls.jsonl
      - data/cleaned/Product/test/WDC_product_for_mp.jsonl
      - data/cleaned/Product/test/WDC_product_for_cls.jsonl
    """
    print("\n" + "=" * 80)
    print("📤 STEP 5: Creating Heldout Datasets from Test Splits")
    print("=" * 80)

    # Create domain-specific test directories
    movie_test_dir = "data/cleaned/Movie/test"
    product_test_dir = "data/cleaned/Product/test"
    os.makedirs(movie_test_dir, exist_ok=True)
    os.makedirs(product_test_dir, exist_ok=True)

    # Check if WDC files already exist and are non-empty
    movie_mp_output = os.path.join(movie_test_dir, "WDC_movie_for_mp.jsonl")
    movie_cls_output = os.path.join(movie_test_dir, "WDC_movie_for_cls.jsonl")
    product_mp_output = os.path.join(product_test_dir, "WDC_product_for_mp.jsonl")
    product_cls_output = os.path.join(product_test_dir, "WDC_product_for_cls.jsonl")
    
    all_wdc_exist = (
        os.path.exists(movie_mp_output) and os.path.getsize(movie_mp_output) > 0 and
        os.path.exists(movie_cls_output) and os.path.getsize(movie_cls_output) > 0 and
        os.path.exists(product_mp_output) and os.path.getsize(product_mp_output) > 0 and
        os.path.exists(product_cls_output) and os.path.getsize(product_cls_output) > 0
    )
    
    if all_wdc_exist:
        print(f"⏭️  Skipping heldout creation: All WDC test files already exist")
        print(f"    Movie test files in: {movie_test_dir}")
        print(f"    Product test files in: {product_test_dir}")
        return
    
    # If test rows are empty but we're being called, we may need to skip
    if not movie_test_rows and not product_test_rows:
        print("⚠️  Warning: No test rows provided. Skipping heldout creation.")
        return

    # Movie MP: all Movie test rows
    if not os.path.exists(movie_mp_output) or os.path.getsize(movie_mp_output) == 0:
        save_jsonl(movie_test_rows, movie_mp_output)
        print(f"  ✅ Movie MP dataset saved: {len(movie_test_rows)} rows -> {movie_mp_output}")
    else:
        print(f"  ⏭️  Skipping Movie MP: File already exists -> {movie_mp_output}")

    # Movie CLS: Movie test rows with valid genres
    if not os.path.exists(movie_cls_output) or os.path.getsize(movie_cls_output) == 0:
        movie_cls_data = [unify_genre_keys(row) for row in movie_test_rows]
        movie_cls_data = [
            row for row in movie_cls_data if "genres" in row and row["genres"] and row["genres"] != "None"
        ]
        save_jsonl(movie_cls_data, movie_cls_output)
        print(f"  ✅ Movie CLS dataset saved: {len(movie_cls_data)} rows -> {movie_cls_output}")
    else:
        print(f"  ⏭️  Skipping Movie CLS: File already exists -> {movie_cls_output}")

    # Product MP: all Product test rows
    if not os.path.exists(product_mp_output) or os.path.getsize(product_mp_output) == 0:
        save_jsonl(product_test_rows, product_mp_output)
        print(
            f"  ✅ Product MP dataset saved: {len(product_test_rows)} rows -> {product_mp_output}"
        )
    else:
        print(f"  ⏭️  Skipping Product MP: File already exists -> {product_mp_output}")

    # Product CLS: Product test rows with valid category
    if not os.path.exists(product_cls_output) or os.path.getsize(product_cls_output) == 0:
        product_cls_data = [unify_category_keys(row) for row in product_test_rows]
        product_cls_data = [
            row
            for row in product_cls_data
            if "category" in row and row["category"] and row["category"] != "None"
        ]
        save_jsonl(product_cls_data, product_cls_output)
        print(
            f"  ✅ Product CLS dataset saved: {len(product_cls_data)} rows -> {product_cls_output}"
        )
    else:
        print(f"  ⏭️  Skipping Product CLS: File already exists -> {product_cls_output}")


def clean_training_datasets(movie_dir: str, product_dir: str) -> None:
    """Clean training datasets in both Movie and Product directories."""
    print("\n" + "="*80)
    print("🧹 STEP 5: Cleaning Training Datasets")
    print("="*80)
    
    # Clean Movie dataset
    print("Cleaning Movie dataset...")
    movie_files = glob.glob(os.path.join(movie_dir, "*.jsonl")) + glob.glob(os.path.join(movie_dir, "*.json"))
    
    for file_path in sorted(movie_files):
        filename = os.path.basename(file_path)
        print(f"  Processing {filename}...")
        
        try:
            # Load data (already flattened from step 2)
            data = load_jsonl(file_path)
            print(f"    Loaded {len(data)} rows")
            
            if len(data) == 0:
                print(f"    ⚠️  No data found, skipping...")
                continue
            
            # Convert to table format
            table_data = [(i, row) for i, row in enumerate(data)]
            
            # Apply comprehensive table cleaning
            print(f"    Applying comprehensive table cleaning...")
            cleaned_table_data = clean_table_data(
                table_data, 
                tokenizer_name="bert-base-uncased",
                max_tokens=512,
                max_indexed_fields=3,
                max_tokens_per_field=20,
                skip_non_english=True,
                skip_non_bert=True,
                english_ratio_threshold=0.7,
                unk_threshold=0.3,
                min_text_fields=2
            )
            
            # Convert back to list format
            cleaned_data = [row for table_id, row in cleaned_table_data]
            print(f"    Final cleaned data: {len(cleaned_data)} rows")
            
            # Save cleaned data back to original file
            save_jsonl(cleaned_data, file_path)
            print(f"    ✅ Cleaned data saved to {filename}")
            
        except Exception as e:
            print(f"    ❌ Error processing {filename}: {str(e)}")
            continue
    
    # Clean Product dataset
    print("\nCleaning Product dataset...")
    product_files = glob.glob(os.path.join(product_dir, "*.jsonl")) + glob.glob(os.path.join(product_dir, "*.json"))
    
    for file_path in sorted(product_files):
        filename = os.path.basename(file_path)
        print(f"  Processing {filename}...")
        
        try:
            # Load data (already flattened from step 2)
            data = load_jsonl(file_path)
            print(f"    Loaded {len(data)} rows")
            
            if len(data) == 0:
                print(f"    ⚠️  No data found, skipping...")
                continue
            
            # Convert to table format
            table_data = [(i, row) for i, row in enumerate(data)]
            
            # Apply comprehensive table cleaning
            print(f"    Applying comprehensive table cleaning...")
            cleaned_table_data = clean_table_data(
                table_data, 
                tokenizer_name="bert-base-uncased",
                max_tokens=512,
                max_indexed_fields=3,
                max_tokens_per_field=20,
                skip_non_english=True,
                skip_non_bert=True,
                english_ratio_threshold=0.7,
                unk_threshold=0.3,
                min_text_fields=2
            )
            
            # Convert back to list format
            cleaned_data = [row for table_id, row in cleaned_table_data]
            print(f"    Final cleaned data: {len(cleaned_data)} rows")
            
            # Save cleaned data back to original file
            save_jsonl(cleaned_data, file_path)
            print(f"    ✅ Cleaned data saved to {filename}")
            
        except Exception as e:
            print(f"    ❌ Error processing {filename}: {str(e)}")
            continue

# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified preprocessing pipeline for NAVI dataset"
    )
    _repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--raw_movie_dir",
        type=str,
        default=str(_repo_root / "data" / "raw" / "Movie_top100"),
        help="Raw Movie dataset directory (default: data/raw/Movie_top100)",
    )
    parser.add_argument(
        "--raw_product_dir",
        type=str,
        default=str(_repo_root / "data" / "raw" / "Product_top100"),
        help="Raw Product dataset directory (default: data/raw/Product_top100)",
    )
    parser.add_argument(
        "--flattened_movie_dir",
        type=str,
        default="data/flattened/Movie_top100",
        help="Output directory for flattened Movie tables",
    )
    parser.add_argument(
        "--flattened_product_dir",
        type=str,
        default="data/flattened/Product_top100",
        help="Output directory for flattened Product tables",
    )
    parser.add_argument(
        "--cleaned_movie_dir",
        type=str,
        default="data/cleaned/Movie_top100",
        help="Output directory for cleaned Movie tables",
    )
    parser.add_argument(
        "--cleaned_product_dir",
        type=str,
        default="data/cleaned/Product_top100",
        help="Output directory for cleaned Product tables",
    )
    parser.add_argument(
        "--target_product_rows",
        type=int,
        default=480_817,
        help="Target total number of rows across all Product tables after resizing",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split ratio for each cleaned table",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio for each cleaned table",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling and splitting",
    )

    args = parser.parse_args()
    
    print("🚀 UNIFIED PREPROCESSING PIPELINE")
    print("=" * 80)
    print(f"Raw Movie directory: {args.raw_movie_dir}")
    print(f"Raw Product directory: {args.raw_product_dir}")
    print(f"Flattened Movie directory: {args.flattened_movie_dir}")
    print(f"Flattened Product directory: {args.flattened_product_dir}")
    print(f"Cleaned Movie directory: {args.cleaned_movie_dir}")
    print(f"Cleaned Product directory: {args.cleaned_product_dir}")
    print(f"Target Product total rows: {args.target_product_rows}")
    print(f"Train ratio: {args.train_ratio}")
    print(f"Validation ratio: {args.val_ratio}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)

    # Step 1: Stratified resize for Product raw tables
    resize_product_dataset_stratified(
        args.raw_product_dir, args.target_product_rows, args.seed, args.flattened_product_dir
    )

    # Step 2: Flatten both Movie and Product datasets (raw → flattened)
    flatten_datasets_with_dirs(
        args.raw_movie_dir,
        args.raw_product_dir,
        args.flattened_movie_dir,
        args.flattened_product_dir,
    )

    # Step 3: Clean flattened datasets (flattened → cleaned)
    clean_datasets_with_dirs(
        args.flattened_movie_dir,
        args.flattened_product_dir,
        args.cleaned_movie_dir,
        args.cleaned_product_dir,
    )

    # Step 4: Split cleaned tables into train/validation/test
    movie_train_dir = "data/cleaned/Movie/train"
    movie_val_dir = "data/cleaned/Movie/validation"
    product_train_dir = "data/cleaned/Product/train"
    product_val_dir = "data/cleaned/Product/validation"

    movie_test_rows, product_test_rows = split_cleaned_tables_to_splits(
        args.cleaned_movie_dir,
        args.cleaned_product_dir,
        movie_train_dir,
        movie_val_dir,
        product_train_dir,
        product_val_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    # Step 5: Create heldout WDC datasets from test splits
    # Note: test_out_dir parameter is kept for backward compatibility but files are written to domain-specific directories
    test_out_dir = "data/cleaned/test"  # Kept for reference, but not used directly
    create_heldout_datasets_from_test_rows(
        movie_test_rows,
        product_test_rows,
        test_out_dir,
    )

    print("\n" + "=" * 80)
    print("🎉 PREPROCESSING PIPELINE COMPLETE!")
    print("=" * 80)
    print("📁 Output artifacts:")
    print(f"  - {args.flattened_movie_dir}/ (Flattened Movie tables)")
    print(f"  - {args.flattened_product_dir}/ (Flattened Product tables)")
    print(f"  - {args.cleaned_movie_dir}/ (Cleaned Movie tables)")
    print(f"  - {args.cleaned_product_dir}/ (Cleaned Product tables)")
    print("  - data/cleaned/Movie/train/ (Movie train split)")
    print("  - data/cleaned/Movie/validation/ (Movie validation split)")
    print("  - data/cleaned/Product/train/ (Product train split)")
    print("  - data/cleaned/Product/validation/ (Product validation split)")
    print("  - data/cleaned/Movie/test/WDC_movie_for_mp.jsonl (Movie heldout for MP)")
    print("  - data/cleaned/Movie/test/WDC_movie_for_cls.jsonl (Movie heldout for CLS)")
    print("  - data/cleaned/Product/test/WDC_product_for_mp.jsonl (Product heldout for MP)")
    print("  - data/cleaned/Product/test/WDC_product_for_cls.jsonl (Product heldout for CLS)")

if __name__ == "__main__":
    main()