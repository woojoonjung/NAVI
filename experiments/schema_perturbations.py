"""
Schema perturbation utilities for schema noise robustness experiments.
"""

import json
import random
import string
from typing import List, Dict, Set
from collections import defaultdict
from dataset.dataset import NaviDataset, FieldEntropyAnalyzer


def identify_low_entropy_headers(data: List[Dict], domain: str, random_seed: int = 42) -> Dict[int, Set[str]]:
    """
    Identify low-entropy headers for each table in the dataset.
    
    Args:
        data: List of JSON row dictionaries
        domain: Domain name ('movie' or 'product')
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping table_id to set of low-entropy header names
    """
    random.seed(random_seed)
    
    # Convert data to NaviDataset format (table_id, row_dict)
    # Try to use _table_id if available, otherwise group by header similarity
    dataset_data = []
    table_id_map = {}
    
    # Check if data has _table_id field
    has_table_id = any('_table_id' in row for row in data[:10]) if data else False
    
    if has_table_id:
        # Use existing table_id
        for idx, row in enumerate(data):
            table_id = row.get('_table_id', 0)
            dataset_data.append((table_id, row))
            table_id_map[idx] = table_id
    else:
        # Group rows by similar header sets (rows with same headers likely from same table)
        # For simplicity, group every 100 rows as a table for entropy calculation
        # This ensures we have enough rows per table for meaningful entropy analysis
        for idx, row in enumerate(data):
            table_id = idx // 100  # Group every 100 rows as a table
            dataset_data.append((table_id, row))
            table_id_map[idx] = table_id
    
    # Create NaviDataset with entropy analysis
    dataset = NaviDataset(dataset_data, compute_field_entropy=True)
    
    # Get field categories
    field_categories = dataset.get_field_categories()
    
    # Extract low-entropy fields per table
    # Aggregate across all tables to get common low-entropy headers
    low_entropy_headers = {}
    all_low_entropy = set()
    
    if field_categories:
        for table_id, categories in field_categories.items():
            table_low = categories.get('low_entropy', set())
            low_entropy_headers[table_id] = table_low
            all_low_entropy.update(table_low)
        
        # Also create a global mapping for tables that don't have explicit table_id
        # Use table_id 0 as default
        if 0 not in low_entropy_headers:
            low_entropy_headers[0] = all_low_entropy
    
    return low_entropy_headers


def apply_synonym_replacement(data: List[Dict], synonym_map: Dict[str, List[str]], 
                             domain: str, sample_ratio: float = 0.5, random_seed: int = 42) -> List[Dict]:
    """
    Apply synonym replacement to sampled low-entropy headers.
    
    Args:
        data: List of JSON row dictionaries
        synonym_map: Dictionary mapping headers to lists of synonyms
        domain: Domain name ('movie' or 'product')
        sample_ratio: Ratio of low-entropy headers to replace (default: 0.5)
        random_seed: Random seed for reproducibility
    
    Returns:
        List of perturbed JSON row dictionaries
    """
    random.seed(random_seed)
    
    # Identify low-entropy headers
    low_entropy_headers = identify_low_entropy_headers(data, domain, random_seed)
    
    # Get domain-specific synonym map
    domain_synonyms = synonym_map.get(domain.lower(), {})
    
    # Create mapping: table_id -> set of headers to replace
    headers_to_replace = {}
    for table_id, low_headers in low_entropy_headers.items():
        # Sample subset of low-entropy headers
        headers_list = list(low_headers)
        num_to_replace = max(1, int(len(headers_list) * sample_ratio))
        sampled_headers = random.sample(headers_list, min(num_to_replace, len(headers_list)))
        headers_to_replace[table_id] = set(sampled_headers)
    
    # Apply synonym replacement
    perturbed_data = []
    for idx, row in enumerate(data):
        # Determine table_id
        if '_table_id' in row:
            table_id = row['_table_id']
        else:
            table_id = idx // 100
        
        new_row = {}
        
        for header, value in row.items():
            # Skip metadata fields
            if header.startswith('_'):
                new_row[header] = value
                continue
            
            # Check if this header should be replaced
            # Use prefix/suffix matching for dot-notation headers
            should_replace = False
            matched_low_entropy_header = None
            
            # Check both specific table_id and default table_id 0
            headers_to_check = set()
            if table_id in headers_to_replace:
                headers_to_check.update(headers_to_replace[table_id])
            if 0 in headers_to_replace:
                headers_to_check.update(headers_to_replace[0])
            
            # Check if header matches any low-entropy header (exact, prefix, or suffix match)
            for low_entropy_header in headers_to_check:
                if header == low_entropy_header:
                    should_replace = True
                    matched_low_entropy_header = low_entropy_header
                    break
                elif header.startswith(low_entropy_header + "."):
                    should_replace = True
                    matched_low_entropy_header = low_entropy_header
                    break
                elif header.endswith("." + low_entropy_header):
                    should_replace = True
                    matched_low_entropy_header = low_entropy_header
                    break
            
            if should_replace:
                # Try to find a matching synonym key
                matched_key = None
                for key in domain_synonyms.keys():
                    # Check if header starts with key (e.g., "director.name" starts with "director")
                    if header.startswith(key + ".") or header == key:
                        matched_key = key
                        break
                    # Check if header ends with key (e.g., "something.aggregaterating" ends with "aggregaterating")
                    elif header.endswith("." + key) or "." + key + "." in header:
                        matched_key = key
                        break
                
                if matched_key and domain_synonyms[matched_key]:
                    # Replace the matched key part with synonym
                    if header == matched_key:
                        # Exact match - replace entire header
                        synonym = random.choice(domain_synonyms[matched_key])
                        new_row[synonym] = value
                    elif header.startswith(matched_key + "."):
                        # Prefix match - replace prefix part
                        suffix = header[len(matched_key):]  # e.g., ".name" from "director.name"
                        synonym = random.choice(domain_synonyms[matched_key])
                        new_row[synonym + suffix] = value
                    elif header.endswith("." + matched_key):
                        # Suffix match - replace suffix part
                        prefix = header[:-len(matched_key) - 1]  # Remove ".aggregaterating"
                        synonym = random.choice(domain_synonyms[matched_key])
                        new_row[prefix + "." + synonym] = value
                    else:
                        # Contains match - replace the matched part
                        synonym = random.choice(domain_synonyms[matched_key])
                        new_row[header.replace(matched_key, synonym)] = value
                else:
                    # No synonym available, keep original
                    new_row[header] = value
            else:
                new_row[header] = value
        
        perturbed_data.append(new_row)
    
    return perturbed_data


def corrupt_header(header: str, num_chars: int = None) -> str:
    """
    Corrupt a header by randomly modifying 1-2 characters.
    Uses substitution, insertion, or deletion operations.
    
    Args:
        header: Original header string
        num_chars: Number of characters to corrupt (default: random 1-2)
    
    Returns:
        Corrupted header string
    """
    if len(header) == 0:
        return header
    
    if num_chars is None:
        num_chars = random.randint(1, min(2, len(header)))
    
    header_list = list(header)
    original_length = len(header_list)
    
    if original_length == 0:
        return header
    
    # Apply corruption operations one at a time
    for _ in range(num_chars):
        if len(header_list) == 0:
            break
        
        # Choose corruption type based on constraints
        corruption_options = ['substitute']
        
        # Only allow insert if we haven't grown too much
        if len(header_list) < 50:
            corruption_options.append('insert')
        # Only allow delete if we have more than 1 character
        if len(header_list) > 1:
            corruption_options.append('delete')
        
        corruption_type = random.choice(corruption_options)
        
        if corruption_type == 'substitute':
            # Replace a random character with random character
            pos = random.randint(0, len(header_list) - 1)
            header_list[pos] = random.choice(string.ascii_letters + string.digits)
        elif corruption_type == 'insert':
            # Insert random character at random position
            pos = random.randint(0, len(header_list))
            header_list.insert(pos, random.choice(string.ascii_letters + string.digits))
        elif corruption_type == 'delete':
            # Delete a random character
            pos = random.randint(0, len(header_list) - 1)
            header_list.pop(pos)
    
    return ''.join(header_list)


def apply_header_typos(data: List[Dict], domain: str, sample_ratio: float = 0.5, random_seed: int = 42) -> List[Dict]:
    """
    Apply typos to sampled low-entropy headers (corrupt 1-2 characters).
    
    Args:
        data: List of JSON row dictionaries
        domain: Domain name ('movie' or 'product')
        sample_ratio: Ratio of low-entropy headers to corrupt (default: 0.5)
        random_seed: Random seed for reproducibility
    
    Returns:
        List of perturbed JSON row dictionaries
    """
    random.seed(random_seed)
    
    # Identify low-entropy headers
    low_entropy_headers = identify_low_entropy_headers(data, domain, random_seed)
    
    # Create mapping: table_id -> set of headers to corrupt
    headers_to_corrupt = {}
    for table_id, low_headers in low_entropy_headers.items():
        # Sample half of low-entropy headers
        headers_list = list(low_headers)
        num_to_corrupt = max(1, int(len(headers_list) * sample_ratio))
        sampled_headers = random.sample(headers_list, min(num_to_corrupt, len(headers_list)))
        headers_to_corrupt[table_id] = set(sampled_headers)
    
    # Track header replacements to maintain consistency within a table
    header_replacements = {}  # (table_id, original_header) -> corrupted_header
    
    # Apply typos
    perturbed_data = []
    for idx, row in enumerate(data):
        # Determine table_id
        if '_table_id' in row:
            table_id = row['_table_id']
        else:
            table_id = idx // 100
        
        new_row = {}
        
        for header, value in row.items():
            # Skip metadata fields
            if header.startswith('_'):
                new_row[header] = value
                continue
            
            # Check if this header should be corrupted
            # Use prefix/suffix matching for dot-notation headers
            should_corrupt = False
            matched_low_entropy_header = None
            
            # Check both specific table_id and default table_id 0
            headers_to_check = set()
            if table_id in headers_to_corrupt:
                headers_to_check.update(headers_to_corrupt[table_id])
            if 0 in headers_to_corrupt:
                headers_to_check.update(headers_to_corrupt[0])
            
            # Check if header matches any low-entropy header (exact, prefix, or suffix match)
            for low_entropy_header in headers_to_check:
                if header == low_entropy_header:
                    should_corrupt = True
                    matched_low_entropy_header = low_entropy_header
                    break
                elif header.startswith(low_entropy_header + "."):
                    should_corrupt = True
                    matched_low_entropy_header = low_entropy_header
                    break
                elif header.endswith("." + low_entropy_header):
                    should_corrupt = True
                    matched_low_entropy_header = low_entropy_header
                    break
            
            if should_corrupt:
                # Use cached corruption if available (for consistency within table)
                key = (table_id, header)
                if key not in header_replacements:
                    header_replacements[key] = corrupt_header(header)
                new_row[header_replacements[key]] = value
            else:
                new_row[header] = value
        
        perturbed_data.append(new_row)
    
    return perturbed_data


def apply_column_reordering(data: List[Dict], random_seed: int = 42) -> List[Dict]:
    """
    Apply column reordering by randomly shuffling column order for each row.
    
    Args:
        data: List of JSON row dictionaries
        random_seed: Random seed for reproducibility
    
    Returns:
        List of perturbed JSON row dictionaries
    """
    random.seed(random_seed)
    
    perturbed_data = []
    for row in data:
        # Separate metadata fields from data fields
        metadata = {k: v for k, v in row.items() if k.startswith('_')}
        data_fields = {k: v for k, v in row.items() if not k.startswith('_')}
        
        # Shuffle column order
        items = list(data_fields.items())
        random.shuffle(items)
        
        # Reconstruct row with shuffled columns
        new_row = {**metadata, **dict(items)}
        perturbed_data.append(new_row)
    
    return perturbed_data

