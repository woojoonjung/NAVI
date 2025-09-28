import json
import os
import re
import argparse
import random
from transformers import BertTokenizer
import unicodedata

def detect_language_simple(text, min_confidence=0.8):
    """
    Simple language detection based on character analysis.
    Returns 'en' for English, 'non-en' for non-English, 'unknown' for unclear.
    
    Args:
        text: Input text to analyze
        min_confidence: Minimum confidence threshold for language detection
    """
    if not isinstance(text, str) or not text.strip():
        return 'unknown'
    
    # Remove whitespace and get character statistics
    text_clean = re.sub(r'\s+', '', text)
    if len(text_clean) < 3:
        return 'unknown'
    
    # Count different character types
    ascii_letters = 0
    ascii_digits = 0
    cyrillic_chars = 0
    cjk_chars = 0
    latin_extended = 0
    other_chars = 0
    
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
        else:
            other_chars += 1
    
    total_chars = len(text_clean)
    ascii_ratio = (ascii_letters + ascii_digits) / total_chars
    cyrillic_ratio = cyrillic_chars / total_chars
    cjk_ratio = cjk_chars / total_chars
    latin_extended_ratio = latin_extended / total_chars
    
    # Determine language based on character ratios
    if ascii_ratio >= min_confidence:
        return 'en'
    elif cyrillic_ratio >= min_confidence:
        return 'non-en'  # Russian, Bulgarian, etc.
    elif cjk_ratio >= min_confidence:
        return 'non-en'  # Chinese, Japanese, Korean
    elif latin_extended_ratio >= 0.3:  # French, German, Spanish, etc.
        return 'non-en'
    elif ascii_ratio >= 0.5:  # Mixed but mostly ASCII
        return 'en'
    else:
        return 'non-en'

def is_english_table(table_dict, min_english_ratio=0.7, min_text_fields=2):
    """
    Check if a table contains primarily English content.
    Returns True if table is English, False if multilingual.
    
    Args:
        table_dict: Dictionary representing a table row
        min_english_ratio: Minimum ratio of English text fields required
        min_text_fields: Minimum number of text fields required for evaluation
    """
    text_fields = []
    
    # Collect meaningful text fields for language evaluation
    for field_name, field_value in table_dict.items():
        if isinstance(field_value, str) and field_value.strip():
            # Skip very short values, URLs, pure numbers, and dates
            if (len(field_value.strip()) > 10 and 
                not field_value.startswith('http') and 
                not field_value.replace('.', '').replace(',', '').replace('-', '').replace(':', '').isdigit() and
                not re.match(r'^\d{4}-\d{2}-\d{2}', field_value.strip())):
                text_fields.append(field_value)
    
    # Need at least min_text_fields to make a decision
    if len(text_fields) < min_text_fields:
        return True  # Not enough text to evaluate, keep the table
    
    # Analyze language of each text field
    english_fields = 0
    for field in text_fields:
        lang = detect_language_simple(field)
        if lang == 'en':
            english_fields += 1
    
    english_ratio = english_fields / len(text_fields)
    return english_ratio >= min_english_ratio

def is_bert_tokenizable(text, tokenizer, unk_threshold=0.2):
    """
    Check if text can be properly tokenized by BERT.
    Returns True if text is mostly tokenizable, False if too many UNK tokens.
    
    Args:
        text: Input text to check
        tokenizer: BERT tokenizer instance
        unk_threshold: Maximum ratio of UNK tokens allowed (0.0-1.0)
    """
    if not isinstance(text, str) or not text.strip():
        return True  # Skip empty text
    
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    
    if not tokens:
        return True  # Empty tokenization, keep it
    
    # Count UNK tokens
    unk_count = sum(1 for token in tokens if token == tokenizer.unk_token)
    unk_ratio = unk_count / len(tokens)
    
    return unk_ratio <= unk_threshold

def is_bert_compatible_table(table_dict, tokenizer, unk_threshold=0.3, min_text_fields=1):
    """
    Check if a table contains text that can be properly tokenized by BERT.
    Returns True if table is BERT-compatible, False if too many UNK tokens.
    
    Args:
        table_dict: Dictionary representing a table row
        tokenizer: BERT tokenizer instance
        unk_threshold: Maximum ratio of UNK tokens allowed
        min_text_fields: Minimum number of text fields required for evaluation
    """
    text_fields = []
    
    # Collect text fields for evaluation
    for field_name, field_value in table_dict.items():
        if isinstance(field_value, str) and field_value.strip():
            # Skip very short values, URLs, and pure numbers
            if (len(field_value.strip()) > 5 and 
                not field_value.startswith('http') and 
                not field_value.replace('.', '').replace(',', '').replace('-', '').isdigit()):
                text_fields.append(field_value)
    
    # Need at least min_text_fields to make a decision
    if len(text_fields) < min_text_fields:
        return True  # Not enough text to evaluate, keep the table
    
    # Check if majority of text fields are tokenizable
    tokenizable_fields = sum(1 for field in text_fields if is_bert_tokenizable(field, tokenizer, unk_threshold))
    tokenizable_ratio = tokenizable_fields / len(text_fields)
    
    return tokenizable_ratio >= 0.5  # At least 50% of fields should be tokenizable

def get_table_text_content(table_dict):
    """
    Extract and concatenate all text content from a table for tokenization testing.
    """
    text_parts = []
    
    for field_name, field_value in table_dict.items():
        if isinstance(field_value, str) and field_value.strip():
            # Skip very short values, URLs, and pure numbers
            if (len(field_value.strip()) > 5 and 
                not field_value.startswith('http') and 
                not field_value.replace('.', '').replace(',', '').replace('-', '').isdigit()):
                text_parts.append(field_value)
    
    return " ".join(text_parts)

def load_jsonl(file_path):
    """
    Load JSONL (JSON Lines) data from a file.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def preprocess_text(text):
    """
    Clean and preprocess text
    """
    if isinstance(text, str):
        text = text.lower().strip()
    return str(text)

def flatten_json(json_obj, parent_key='', sep='.'):
    """
    Recursively flattens a nested JSON object.
    """
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
                flattened[full_key] = preprocess_text(item)
    else:
        flattened[parent_key] = preprocess_text(json_obj)
    return flattened

def preprocess_jsonl_data(jsonl_data):
    """
    Preprocesses JSONL data by flattening nested structures.
    """
    return [flatten_json(row) for row in jsonl_data]

def save_preprocessed_data(json_data, output_path):
    """
    Saves preprocessed JSON data to a JSONL file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in json_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

# ===============================================

def clean_table_data(json_data, tokenizer_name="bert-base-uncased", 
                         max_tokens=512, max_indexed_fields=3, max_tokens_per_field=20,
                         skip_non_english=True, skip_non_bert=True, 
                         english_ratio_threshold=0.7, unk_threshold=0.3, min_text_fields=2):
    """
    Clean table data with language and BERT vocabulary validation.
    
    Args:
        json_data: List of (table_id, table_dict) tuples
        tokenizer_name: Name of the BERT tokenizer to use
        max_tokens: Maximum total tokens per table
        max_indexed_fields: Maximum indexed fields to keep per type
        max_tokens_per_field: Maximum tokens per individual field
        skip_non_english: Whether to skip non-English tables
        skip_non_bert: Whether to skip tables that can't be properly tokenized
        english_ratio_threshold: Minimum ratio of English text fields required
        unk_threshold: Maximum ratio of UNK tokens allowed (0.0-1.0)
        min_text_fields: Minimum number of text fields required for evaluation
    """
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    processed_data = []
    skipped_non_english = 0
    skipped_non_bert = 0
    
    print("🔄 Preprocessing table data with language and BERT vocabulary validation...")
    
    for table_id, table_dict in json_data:
        # Step 0a: Skip non-English tables if enabled
        if skip_non_english and not is_english_table(table_dict, english_ratio_threshold, min_text_fields):
            skipped_non_english += 1
            if skipped_non_english % 100 == 0:
                print(f"   Skipped {skipped_non_english} non-English tables...")
            continue
        
        # Step 0b: Skip tables that can't be properly tokenized by BERT
        if skip_non_bert and not is_bert_compatible_table(table_dict, tokenizer, unk_threshold, min_text_fields):
            skipped_non_bert += 1
            if skipped_non_bert % 100 == 0:
                print(f"   Skipped {skipped_non_bert} non-BERT-compatible tables...")
            continue
        
        # Step 1: Handle indexed fields
        processed_table = handle_indexed_fields(table_dict, max_indexed_fields)
        
        # Step 2: Truncate long text fields
        processed_table = truncate_long_fields(processed_table, max_tokens_per_field, tokenizer)
        
        # Step 3: Select fields that fit in token limit
        processed_table = select_fields_for_tokenization(processed_table, tokenizer, max_tokens)
        
        processed_data.append((table_id, processed_table))
        
        if len(processed_data) % 1000 == 0:
            print(f"   Processed {len(processed_data)} instances...")
    
    print(f"✅ Preprocessing complete. Processed {len(processed_data)} instances.")
    if skip_non_english:
        print(f"🚫 Skipped {skipped_non_english} non-English tables.")
    if skip_non_bert:
        print(f"�� Skipped {skipped_non_bert} non-BERT-compatible tables.")
    return processed_data

def handle_indexed_fields(table_dict, max_indexed_fields=3):
    """Handle indexed fields by sampling representative ones."""
    processed_table = {}
    field_groups = {}
    
    # Group indexed fields
    for field_name, field_value in table_dict.items():
        if '.' in field_name and any(char.isdigit() for char in field_name):
            # Extract field type (first word) and check if there's an index anywhere
            field_type = field_name.split('.')[0]  # First word is always the type
            
            # Check if this field has an index (anywhere after the first word)
            has_index = False
            for part in field_name.split('.')[1:]:  # Check all parts after the first word
                if part.isdigit():
                    has_index = True
                    break
            
            if has_index:
                if field_type not in field_groups:
                    field_groups[field_type] = []
                field_groups[field_type].append((field_name, field_value))
            else:
                # Field has dots but no index (e.g., "aggregaterating.bestrating")
                processed_table[field_name] = field_value
        else:
            # Field has no dots (e.g., "name", "description")
            processed_table[field_name] = field_value
    
    # Sample indexed fields
    for field_type, fields in field_groups.items():
        if len(fields) > max_indexed_fields:
            # Sample first few fields (0, 1, 2, ...)
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
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    processed_table = {}
    
    for field_name, field_value in table_dict.items():
        # Estimate token count for this field
        estimated_tokens = estimate_token_count(field_name, field_value, tokenizer)
        
        if estimated_tokens > max_tokens_per_field and isinstance(field_value, str):
            # Field exceeds token limit, need to truncate
            # Try to truncate the value while keeping the field name
            truncated_value = truncate_value_to_token_limit(
                field_name, field_value, tokenizer, max_tokens_per_field
            )
            processed_table[field_name] = truncated_value
        else:
            # Field is within token limit, keep as-is
            processed_table[field_name] = field_value
    
    return processed_table

def truncate_value_to_token_limit(field_name, field_value, tokenizer, max_tokens_per_field):
    """Truncate field value to fit within token limit."""
    # Reserve tokens for field name and separator
    # Format: "field_name : field_value"
    name_tokens = tokenizer.tokenize(field_name)
    separator_tokens = tokenizer.tokenize(" : ")
    reserved_tokens = len(name_tokens) + len(separator_tokens)
    
    # Calculate how many tokens we can use for the value
    available_tokens = max_tokens_per_field - reserved_tokens
    
    if available_tokens <= 0:
        # Field name itself is too long, return minimal value
        return "..."
    
    # Tokenize the value
    value_tokens = tokenizer.tokenize(str(field_value))
    
    if len(value_tokens) <= available_tokens:
        # Value fits within limit
        return field_value
    else:
        # Truncate value tokens
        truncated_tokens = value_tokens[:available_tokens]
        # Convert back to text (approximate)
        truncated_text = tokenizer.convert_tokens_to_string(truncated_tokens)
        # Add ellipsis to indicate truncation
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
    
    # Sort fields by priority (shorter fields first)
    field_items = list(table_dict.items())
    field_items.sort(key=lambda x: len(str(x[1])))
    
    for field_name, field_value in field_items:
        estimated_tokens = estimate_token_count(field_name, field_value, tokenizer)
        
        if current_tokens + estimated_tokens <= max_tokens:
            selected_fields[field_name] = field_value
            current_tokens += estimated_tokens
        else:
            break  # Stop when we would exceed limit
    
    return selected_fields

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Preprocess JSONL files with language and BERT vocabulary validation")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing JSONL files to process")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum total tokens per table")
    parser.add_argument("--max_indexed_fields", type=int, default=3, help="Maximum indexed fields to keep per type")
    parser.add_argument("--max_tokens_per_field", type=int, default=20, help="Maximum tokens per individual field")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased", help="Tokenizer to use for token counting")
    parser.add_argument("--skip_non_english", action="store_true", default=True, help="Skip tables with non-English content")
    parser.add_argument("--skip_non_bert", action="store_true", default=True, help="Skip tables that can't be properly tokenized by BERT")
    parser.add_argument("--english_ratio_threshold", type=float, default=0.7, help="Minimum ratio of English text fields required (0.0-1.0)")
    parser.add_argument("--unk_threshold", type=float, default=0.3, help="Maximum ratio of UNK tokens allowed (0.0-1.0)")
    parser.add_argument("--min_text_fields", type=int, default=2, help="Minimum number of text fields required for evaluation")
    args = parser.parse_args()
    
    input_dir = args.input_dir
    
    # Create output directory: {input_dir}_cleaned
    output_dir = f"{input_dir}_cleaned"
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Get all JSONL/JSON files in the input directory
    jsonl_files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl') or f.endswith('.json')]
    
    if not jsonl_files:
        print(f"No JSONL or JSON files found in {input_dir}")
        exit(1)
    
    print(f"Found {len(jsonl_files)} files to process:")
    for f in jsonl_files:
        print(f"  - {f}")
    
    # Track processing statistics
    total_processed = 0
    total_skipped_lang = 0
    total_skipped_bert = 0
    
    for jsonl_file in jsonl_files:
        input_file = os.path.join(input_dir, jsonl_file)
        # Create output filename in the cleaned directory
        base_name = os.path.splitext(jsonl_file)[0]
        output_file = os.path.join(output_dir, f"{base_name}_cleaned.jsonl")
        
        print(f"\nProcessing {input_file}...")
        
        try:
            # Step 1: Load raw data
            raw_data = load_jsonl(input_file)
            print(f"  Loaded {len(raw_data)} rows")
            
            if len(raw_data) == 0:
                print(f"  ⚠️  No data found in {input_file}, skipping...")
                continue
            
            # Step 2: Flatten nested JSON structures
            flattened_data = preprocess_jsonl_data(raw_data)
            print(f"  Flattened to {len(flattened_data)} rows")
            
            # Step 3: Convert to table format (add table_id for compatibility)
            table_data = [(i, row) for i, row in enumerate(flattened_data)]
            
            # Step 4: Apply comprehensive table cleaning with language and BERT validation
            print("  Applying comprehensive table cleaning with language and BERT vocabulary validation...")
            cleaned_table_data = clean_table_data(
                table_data, 
                tokenizer_name=args.tokenizer_name,
                max_tokens=args.max_tokens,
                max_indexed_fields=args.max_indexed_fields,
                max_tokens_per_field=args.max_tokens_per_field,
                skip_non_english=args.skip_non_english,
                skip_non_bert=args.skip_non_bert,
                english_ratio_threshold=args.english_ratio_threshold,
                unk_threshold=args.unk_threshold,
                min_text_fields=args.min_text_fields
            )
            
            # Step 5: Convert back to list format for saving
            cleaned_data = [row for table_id, row in cleaned_table_data]
            print(f"  Final cleaned data: {len(cleaned_data)} rows")
            
            # Step 6: Save cleaned data to the separate directory
            save_preprocessed_data(cleaned_data, output_file)
            print(f"✅ Cleaned data saved to {output_file}")
            
            # Update statistics
            total_processed += len(cleaned_data)
            total_skipped_lang += len(flattened_data) - len(cleaned_data)
            
        except Exception as e:
            print(f"❌ Error processing {input_file}: {str(e)}")
            continue
    
    # Print final statistics
    print(f"\n🎉 Processing complete!")
    print(f"📊 Final Statistics:")
    print(f"  - Total processed rows: {total_processed}")
    print(f"  - Total skipped rows: {total_skipped_lang}")
    print(f"  - Output directory: {output_dir}")
    
    # List all cleaned files
    cleaned_files = [f for f in os.listdir(output_dir) if f.endswith('.jsonl')]
    print(f"  - Generated {len(cleaned_files)} cleaned files:")
    for f in sorted(cleaned_files):
        print(f"    - {f}")