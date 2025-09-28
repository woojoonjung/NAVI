#!/usr/bin/env python3
"""
Script to create quarter-sized versions of Movie or Product JSONL files.
Usage: python create_quarter_data.py [Movie|Product]
"""

import os
import sys
import json
import argparse
from pathlib import Path


def process_jsonl_file(input_file_path, output_file_path):
    """
    Read a JSONL file and write the first quarter of its lines to output file.
    
    Args:
        input_file_path (str): Path to input JSONL file
        output_file_path (str): Path to output JSONL file
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
        
        # Calculate quarter size (at least 1 line)
        quarter_size = max(1, len(lines) // 4)
        quarter_lines = lines[-quarter_size:]
        
        # Write quarter data to output file
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            outfile.writelines(quarter_lines)
        
        print(f"Processed {input_file_path.name}: {len(lines)} -> {len(quarter_lines)} lines")
        return True
        
    except Exception as e:
        print(f"Error processing {input_file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Create quarter-sized versions of Movie or Product JSONL files')
    parser.add_argument('data_type', choices=['Movie', 'Product', 'Movie_wo_desc'], 
                       help='Type of data to process: Movie or Product')
    
    args = parser.parse_args()
    data_type = args.data_type
    
    # Define paths
    workspace_root = Path(__file__).parent.parent
    source_dir = workspace_root / 'data' / f'{data_type}_top100_cleaned'
    output_dir = workspace_root / 'data' / f'Quarter_{data_type}_top100_cleaned'
    
    # Check if source directory exists
    if not source_dir.exists():
        print(f"Error: Source directory '{source_dir}' does not exist!")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_dir}")
    
    # Get all JSONL files from source directory
    jsonl_files = list(source_dir.glob('*.jsonl'))
    
    if not jsonl_files:
        print(f"No JSONL files found in {source_dir}")
        sys.exit(1)
    
    print(f"Found {len(jsonl_files)} JSONL files to process...")
    
    # Process each JSONL file
    successful_count = 0
    failed_count = 0
    
    for jsonl_file in sorted(jsonl_files):
        # Skip hidden files and directories
        if jsonl_file.name.startswith('.'):
            continue
            
        output_file = output_dir / jsonl_file.name
        
        if process_jsonl_file(jsonl_file, output_file):
            successful_count += 1
        else:
            failed_count += 1
    
    # Print summary
    print(f"\n=== Processing Summary ===")
    print(f"Successfully processed: {successful_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Output directory: {output_dir}")
    
    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()