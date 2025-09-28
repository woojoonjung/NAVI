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

def unify_genre_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unify all genre-related keys to 'genres' field.
    
    Handles the following patterns:
    - "genre": "single_genre" → "genres": "single_genre"
    - "genre.0": "genre1", "genre.1": "genre2", ... → "genres": "genre1, genre2, ..."
    - "url.genre.0": "genre1", "url.genre.1": "genre2", ... → "genres": "genre1, genre2, ..."
    """
    unified_data = {}
    genre_values = []
    
    for key, value in data.items():
        # Check if this is a genre-related key
        if key == "genre":
            # Single genre case
            if value and value != "None" and value != "nan":
                genre_values.append(str(value))
        elif key.startswith("genre.") or key.startswith("url.genre."):
            # Multiple genres case (genre.0, genre.1, etc.)
            if value and value != "None" and value != "nan":
                genre_values.append(str(value))
        else:
            # Not a genre key, keep as is
            unified_data[key] = value
    
    # Create unified genres field
    if genre_values:
        # Remove duplicates while preserving order
        seen = set()
        unique_genres = []
        for genre in genre_values:
            if genre not in seen:
                seen.add(genre)
                unique_genres.append(genre)
        
        unified_data["genres"] = ", ".join(unique_genres)
    else:
        # If no genres found, set to empty string or None
        unified_data["genres"] = ""
    
    return unified_data

def preprocess_jsonl_data(jsonl_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Preprocesses JSONL data by unifying genre keys.
    """
    # First, unify genre keys
    unified_data = [unify_genre_keys(row) for row in jsonl_data]
    
    # Then, filter to keep only rows that have "genres" key
    final_data = [row for row in unified_data if "genres" in row.keys() and row["genres"] and row["genres"] != "None"]
    
    print(f"Original rows: {len(jsonl_data)}")
    print(f"Rows with genres: {len(final_data)}")
    print(f"Removed {len(jsonl_data) - len(final_data)} rows without genres")
    
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
    Process a single JSONL file to unify genre keys.
    """
    print(f"Processing {input_file}...")
    raw_data = load_jsonl(input_file)
    cleaned_data = preprocess_jsonl_data(raw_data)
    save_preprocessed_data(cleaned_data, output_file)
    print(f"Preprocessed data saved to {output_file}")

def process_directory(input_dir: str, output_dir: str = None) -> None:
    """
    Process all JSONL files in a directory to unify genre keys.
    """
    if output_dir is None:
        output_dir = input_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSONL files in the input directory
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    
    for jsonl_file in jsonl_files:
        input_file = os.path.join(input_dir, jsonl_file)
        # Create output filename with "_genre_unified" suffix
        base_name = os.path.splitext(jsonl_file)[0]
        output_file = os.path.join(output_dir, f"{base_name}_genre_unified.jsonl")
        
        process_single_file(input_file, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unify genre keys in JSONL files to 'genres'")
    parser.add_argument("--input", type=str, default=None, 
                       help="Input JSONL file or directory containing JSONL files")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file or directory (optional, defaults to input location)")
    parser.add_argument("--mode", type=str, choices=["file", "dir"], default="file",
                       help="Mode: 'file' for single file, 'dir' for directory")
    
    args = parser.parse_args()

    # ========== Setup paths ==========
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(base_dir, "..", "data", "Cls_datasets", "WDC_movie_45000.jsonl")
    output_path = os.path.join(base_dir, "..", "data", "WDC_movie_for_cls.jsonl")
    if args.mode == "file":
        process_single_file(input_path, output_path)
    else:
        process_directory(args.input, args.output)