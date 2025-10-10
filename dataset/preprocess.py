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
    python preprocess.py
"""

import json
import os
import glob
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List
from transformers import BertTokenizer
import unicodedata

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
    movie_heldout_data = extract_heldout_data(movie_dir, 'movie', 450, 459)
    
    # Save Movie MP dataset (raw flattened data)
    movie_mp_output = "data/WDC_movie_for_mp.jsonl"
    save_jsonl(movie_heldout_data, movie_mp_output)
    print(f"  ✅ Movie MP dataset saved: {len(movie_heldout_data)} rows")
    
    # Create Movie CLS dataset (with genre unification)
    movie_cls_data = [unify_genre_keys(row) for row in movie_heldout_data]
    movie_cls_data = [row for row in movie_cls_data if "genres" in row.keys() and row["genres"] and row["genres"] != "None"]
    movie_cls_output = "data/WDC_movie_for_cls.jsonl"
    save_jsonl(movie_cls_data, movie_cls_output)
    print(f"  ✅ Movie CLS dataset saved: {len(movie_cls_data)} rows")
    
    # Create Product heldout datasets
    print("\nCreating Product heldout datasets...")
    product_heldout_data = extract_heldout_data(product_dir, 'product', 450, 459)
    
    # Save Product MP dataset (raw flattened data)
    product_mp_output = "data/WDC_product_for_mp.jsonl"
    save_jsonl(product_heldout_data, product_mp_output)
    print(f"  ✅ Product MP dataset saved: {len(product_heldout_data)} rows")
    
    # Create Product CLS dataset (with category unification)
    product_cls_data = [unify_category_keys(row) for row in product_heldout_data]
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
    parser = argparse.ArgumentParser(description='Unified preprocessing pipeline for NAVI dataset')
    parser.add_argument('--movie_dir', type=str, default='data/Movie_top100',
                        help='Movie dataset directory')
    parser.add_argument('--product_dir', type=str, default='data/Product_top100',
                        help='Product dataset directory')
    parser.add_argument('--product_max_rows', type=int, default=5000,
                        help='Maximum rows for product dataset resizing')
    parser.add_argument('--heldout_start', type=int, default=450,
                        help='Starting row_id for heldout dataset (inclusive)')
    parser.add_argument('--heldout_end', type=int, default=459,
                        help='Ending row_id for heldout dataset (inclusive)')
    
    args = parser.parse_args()
    
    print("🚀 UNIFIED PREPROCESSING PIPELINE")
    print("="*80)
    print(f"Movie directory: {args.movie_dir}")
    print(f"Product directory: {args.product_dir}")
    print(f"Product max rows: {args.product_max_rows}")
    print(f"Heldout range: {args.heldout_start}-{args.heldout_end}")
    print("="*80)
    
    # Step 1: Resize product dataset
    resize_product_dataset(args.product_dir, args.product_max_rows)
    
    # Step 2: Flatten both movie and product datasets
    flatten_datasets(args.movie_dir, args.product_dir)
    
    # Step 3: Create heldout datasets for masked prediction and classification
    create_heldout_datasets(args.movie_dir, args.product_dir)
    
    # Step 4: Remove heldout rows from training datasets
    remove_heldout_rows_from_datasets(args.movie_dir, args.product_dir, args.heldout_end)
    
    print("\n" + "="*80)
    print("🎉 PREPROCESSING PIPELINE COMPLETE!")
    print("="*80)
    print("📁 Output files created:")
    print("  - data/WDC_movie_for_mp.jsonl (Movie heldout for masked prediction)")
    print("  - data/WDC_movie_for_cls.jsonl (Movie heldout for classification)")
    print("  - data/WDC_product_for_mp.jsonl (Product heldout for masked prediction)")
    print("  - data/WDC_product_for_cls.jsonl (Product heldout for classification)")
    print(f"  - {args.movie_dir}/ (Cleaned Movie training datasets)")
    print(f"  - {args.product_dir}/ (Cleaned Product training datasets)")
    print("\n💡 All datasets are flattened. Training datasets are cleaned.")
    print("   Heldout datasets are kept raw for fair evaluation.")

if __name__ == "__main__":
    main()
