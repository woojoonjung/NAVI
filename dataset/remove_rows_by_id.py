import json
import os
import glob
from pathlib import Path

def filter_jsonl_files(directory_path, min_row_id=460):
    """
    Filter all JSONL files in the directory to keep only rows with row_id > min_row_id.
    Directly overwrites the original files without creating backups.
    """
    # Get all .jsonl files in the directory
    jsonl_files = glob.glob(os.path.join(directory_path, "*.jsonl")) + glob.glob(os.path.join(directory_path, "*.json"))
    
    print(f"Found {len(jsonl_files)} JSONL files in {directory_path}")
    print(f"Filtering to keep only rows with row_id > {min_row_id}")
    print("-" * 80)
    
    total_files_processed = 0
    total_rows_removed = 0
    
    for file_path in sorted(jsonl_files):
        filename = os.path.basename(file_path)
        original_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        try:
            # Read and filter the file
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
                        print(f"Warning: Invalid JSON on line {line_num} in {filename}: {e}")
                        # Keep invalid lines as they might be important
                        filtered_rows.append(line.strip())
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Invalid row_id on line {line_num} in {filename}: {e}")
                        # Keep rows with invalid row_id
                        filtered_rows.append(line.strip())
            
            # Write filtered data back to original file
            with open(file_path, 'w', encoding='utf-8') as f:
                for row in filtered_rows:
                    f.write(row + '\n')
            
            # Get new file size
            new_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            print(f"{filename:<60} | {original_count:>6} → {len(filtered_rows):>6} rows | {removed_count:>6} removed | {original_size_mb:>6.1f}MB → {new_size_mb:>6.1f}MB")
            
            total_files_processed += 1
            total_rows_removed += removed_count
            
        except Exception as e:
            print(f"ERROR processing {filename}: {str(e)}")
    
    print("-" * 80)
    print(f"Processing complete!")
    print(f"Files processed: {total_files_processed}")
    print(f"Total rows removed: {total_rows_removed:,}")

# Run the filtering
directory_path = "data/Product_top100"
filter_jsonl_files(directory_path, min_row_id=459)