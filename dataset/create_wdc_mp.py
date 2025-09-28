import json
import os
import glob
from pathlib import Path

def extract_and_combine_movie_data(input_directory, output_file, start_row_id=0, end_row_id=449):
    """
    Extract rows with row_id between start_row_id and end_row_id from each JSONL file
    and combine them into a single output file.
    
    Args:
        input_directory (str): Path to directory containing Movie JSONL files
        output_file (str): Path to output combined JSONL file
        start_row_id (int): Starting row_id (inclusive)
        end_row_id (int): Ending row_id (inclusive)
    """
    
    # Get all Movie JSONL files in the directory
    jsonl_files = glob.glob(os.path.join(input_directory, "Product_*.jsonl"))
    
    print(f"Found {len(jsonl_files)} Movie JSONL files in {input_directory}")
    print(f"Extracting rows with row_id {start_row_id} to {end_row_id} from each file")
    print(f"Output file: {output_file}")
    print("-" * 80)
    
    total_rows_extracted = 0
    total_files_processed = 0
    rows_per_file = end_row_id - start_row_id + 1
    
    # Open output file for writing
    with open(output_file, 'w', encoding='utf-8') as outfile:
        
        for file_path in sorted(jsonl_files):
            filename = os.path.basename(file_path)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            try:
                rows_extracted = 0
                
                with open(file_path, 'r', encoding='utf-8') as infile:
                    for line_num, line in enumerate(infile, 1):
                        try:
                            row = json.loads(line.strip())
                            row_id = int(row.get('row_id', 0))
                            
                            # Check if row_id is in the target range
                            if start_row_id <= row_id <= end_row_id:
                                outfile.write(line.strip() + '\n')
                                rows_extracted += 1
                                
                                # Stop reading this file if we've extracted all needed rows
                                if rows_extracted >= rows_per_file:
                                    break
                                    
                        except json.JSONDecodeError as e:
                            print(f"Warning: Invalid JSON on line {line_num} in {filename}: {e}")
                            continue
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Invalid row_id on line {line_num} in {filename}: {e}")
                            continue
                
                print(f"{filename:<60} | {rows_extracted:>6} rows extracted | {file_size_mb:>6.1f} MB")
                total_rows_extracted += rows_extracted
                total_files_processed += 1
                
            except Exception as e:
                print(f"ERROR processing {filename}: {str(e)}")
    
    # Verify the output
    output_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    
    print("-" * 80)
    print(f"Processing complete!")
    print(f"Files processed: {total_files_processed}")
    print(f"Total rows extracted: {total_rows_extracted:,}")
    print(f"Expected rows: {len(jsonl_files) * rows_per_file:,}")
    print(f"Output file size: {output_size_mb:.1f} MB")
    
    # Count actual rows in output file
    actual_rows = 0
    with open(output_file, 'r', encoding='utf-8') as f:
        for _ in f:
            actual_rows += 1
    
    print(f"Actual rows in output file: {actual_rows:,}")
    
    return {
        'files_processed': total_files_processed,
        'rows_extracted': total_rows_extracted,
        'expected_rows': len(jsonl_files) * rows_per_file,
        'actual_rows': actual_rows,
        'output_size_mb': output_size_mb
    }

# Configuration
INPUT_DIRECTORY = "data/Movie_wo_desc_Top100_Flattened"  # Adjust path as needed
OUTPUT_FILE = "data/WDC_movie_wo_desc_for_mp.jsonl"
START_ROW_ID = 450
END_ROW_ID = 459

# Run the extraction and combination
if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    # Run the extraction
    results = extract_and_combine_movie_data(
        input_directory=INPUT_DIRECTORY,
        output_file=OUTPUT_FILE,
        start_row_id=START_ROW_ID,
        end_row_id=END_ROW_ID
    )
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Input directory: {INPUT_DIRECTORY}")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Row ID range: {START_ROW_ID} to {END_ROW_ID}")
    print(f"Rows per file: {END_ROW_ID - START_ROW_ID + 1}")
    print(f"Total files processed: {results['files_processed']}")
    print(f"Total rows extracted: {results['rows_extracted']:,}")
    print(f"Expected total rows: {results['expected_rows']:,}")
    print(f"Actual rows in output: {results['actual_rows']:,}")
    print(f"Output file size: {results['output_size_mb']:.1f} MB")