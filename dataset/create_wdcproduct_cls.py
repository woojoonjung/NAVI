import json
import os
import re
import argparse
from typing import Dict, Any, List

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL (JSON Lines) data from a file.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def extract_category_from_url_category(url_category: str) -> str:
    """
    Extract category from url.category by taking the word after first ">" before second ">".
    
    Example: "shop > zip pouches > still life > hats" -> "zip pouches"
    """
    if not url_category or url_category == "None" or url_category == "nan":
        return ""
    
    # Split by ">" and strip whitespace
    parts = [part.strip() for part in url_category.split(">")]
    
    # If we have at least 2 parts (meaning there's content after first ">")
    if len(parts) >= 2:
        return parts[1]  # Return the second part (after first ">")
    else:
        return ""

def unify_category_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unify category-related keys to 'category' field.
    
    Handles the following patterns:
    - "category": "single_category" → "category": "single_category"
    - "url.category": "shop > zip pouches > still life > hats" → "category": "zip pouches"
    - Excludes "offers.category" from unification
    """
    unified_data = {}
    category_value = ""
    
    for key, value in data.items():
        # Check if this is a category-related key (but exclude offers.category)
        if key == "category":
            category_value = str(value)
        elif key == "url.category":
            # URL category case - extract the specific part
            extracted_category = extract_category_from_url_category(value)
            if extracted_category:
                category_value = extracted_category
        else:
            # Not a category key, keep as is
            unified_data[key] = value
    
    # Set the unified category field
    unified_data["category"] = category_value
    
    return unified_data

def preprocess_jsonl_data(jsonl_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Preprocesses JSONL data by unifying category keys.
    """
    # First, unify category keys
    unified_data = [unify_category_keys(row) for row in jsonl_data]
    
    # Then, filter to keep only rows that have "category" key and it's not "None"
    final_data = [row for row in unified_data if "category" in row.keys() and row["category"] and row["category"] != "None"]
    
    print(f"Original rows: {len(jsonl_data)}")
    print(f"Rows with valid category: {len(final_data)}")
    print(f"Removed {len(jsonl_data) - len(final_data)} rows without valid category")
    
    return final_data

def save_preprocessed_data(json_data: List[Dict[str, Any]], output_path: str) -> None:
    """
    Saves preprocessed JSON data to a JSONL file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in json_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def process_single_file(input_file: str, output_file: str) -> None:
    """
    Process a single JSONL file to unify category keys.
    """
    print(f"Processing {input_file}...")
    raw_data = load_jsonl(input_file)
    cleaned_data = preprocess_jsonl_data(raw_data)
    save_preprocessed_data(cleaned_data, output_file)
    print(f"Preprocessed data saved to {output_file}")

def process_directory(input_dir: str, output_dir: str = None) -> None:
    """
    Process all JSONL files in a directory to unify category keys.
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSONL files in the input directory
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    
    for jsonl_file in jsonl_files:
        input_file = os.path.join(input_dir, jsonl_file)
        # Create output filename with "_category_unified" suffix
        base_name = os.path.splitext(jsonl_file)[0]
        output_file = os.path.join(output_dir, f"{base_name}_category_unified.jsonl")
        
        process_single_file(input_file, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unify category keys in JSONL files to 'category'")
    parser.add_argument("--input", type=str, default=None, 
                       help="Input JSONL file or directory containing JSONL files")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file or directory (optional, defaults to input location)")
    parser.add_argument("--mode", type=str, choices=["file", "dir"], default="file",
                       help="Mode: 'file' for single file, 'dir' for directory")
    
    args = parser.parse_args()

    # ========== Setup paths ==========
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "..", "data", "Cls_datasets", "WDC_product_45000.jsonl")
    output_path = os.path.join(base_dir, "..", "data", "WDC_product_for_cls.jsonl")
    if args.mode == "file":
        process_single_file(input_path, output_path)
    else:
        process_directory(args.input, args.output)