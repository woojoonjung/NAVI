#!/usr/bin/env python3
"""
Generate structured queries with [MASK] tokens for segment-wise retrieval experiments.
Creates 10 queries per domain (Movie and Product).
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any

def load_data(file_path: str) -> List[Dict]:
    """Load JSONL data file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def generate_queries_from_row(row: Dict, domain: str, row_id: str) -> List[Dict]:
    """
    Generate queries by masking different attributes from a row.
    Returns list of query dictionaries.
    """
    queries = []
    
    # Filter out metadata fields
    skip_fields = {'row_id', '_table_id', '_row_idx'}
    data_fields = {k: v for k, v in row.items() if k not in skip_fields and v and str(v).strip() and str(v) != 'None'}
    
    if len(data_fields) < 3:
        return queries  # Need at least 3 fields to create meaningful queries
    
    # Common attributes to mask for each domain
    if domain == 'product':
        preferred_mask_fields = ['brand.name', 'category', 'offers.price', 'aggregaterating.ratingvalue', 'name']
    else:  # movie
        preferred_mask_fields = ['director.name', 'genres', 'aggregaterating.ratingvalue', 'name', 'actor.0.name']
    
    # Try to mask preferred fields first
    mask_candidates = [f for f in preferred_mask_fields if f in data_fields]
    if not mask_candidates:
        mask_candidates = list(data_fields.keys())[:5]  # Take first 5 if no preferred fields
    
    # Generate queries by masking different attributes
    for mask_field in mask_candidates[:3]:  # Mask up to 3 different fields per row
        if mask_field not in data_fields:
            continue
            
        # Create structured query with masked field
        structured_query = {}
        given_fields = []
        
        # Add masked field
        structured_query[mask_field] = "[MASK]"
        ground_truth_value = str(data_fields[mask_field])
        
        # Add 2-3 other fields as context
        other_fields = [f for f in data_fields.keys() if f != mask_field]
        num_context = min(3, len(other_fields))
        context_fields = random.sample(other_fields, num_context)
        
        for field in context_fields:
            structured_query[field] = str(data_fields[field])
            given_fields.append(field)
        
        # Only create query if we have at least 2 context fields
        if len(given_fields) >= 2:
            query = {
                "query_id": f"{domain}_{row_id}_{mask_field}",
                "domain": domain,
                "structured_query": structured_query,
                "target_attribute": mask_field,
                "ground_truth_value": ground_truth_value,
                "ground_truth_row_id": row_id,
                "context_fields": given_fields
            }
            queries.append(query)
    
    return queries

def main():
    parser = argparse.ArgumentParser(description='Generate segment retrieval queries')
    parser.add_argument('--product_data', type=str, 
                       default='data/cleaned/Product/test/WDC_product_for_cls.jsonl',
                       help='Path to product data file')
    parser.add_argument('--movie_data', type=str,
                       default='data/cleaned/Movie/test/WDC_movie_for_cls.jsonl',
                       help='Path to movie data file')
    parser.add_argument('--output', type=str,
                       default='experiments/segment_retrieval_queries.jsonl',
                       help='Output file path')
    parser.add_argument('--queries_per_domain', type=int, default=10,
                       help='Number of queries to generate per domain')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    print(f"Loading product data from {args.product_data}...")
    product_data = load_data(args.product_data)
    print(f"Loaded {len(product_data)} product rows")
    
    print(f"Loading movie data from {args.movie_data}...")
    movie_data = load_data(args.movie_data)
    print(f"Loaded {len(movie_data)} movie rows")
    
    all_queries = []
    
    # Generate product queries
    print(f"\nGenerating {args.queries_per_domain} product queries...")
    product_queries = []
    sampled_indices = random.sample(range(len(product_data)), min(len(product_data), args.queries_per_domain * 2))
    
    for idx in sampled_indices:
        row = product_data[idx]
        row_id = str(row.get('row_id', idx))
        queries = generate_queries_from_row(row, 'product', row_id)
        product_queries.extend(queries)
        if len(product_queries) >= args.queries_per_domain:
            break
    
    # Select diverse queries
    product_queries = product_queries[:args.queries_per_domain]
    all_queries.extend(product_queries)
    print(f"Generated {len(product_queries)} product queries")
    
    # Generate movie queries
    print(f"\nGenerating {args.queries_per_domain} movie queries...")
    movie_queries = []
    sampled_indices = random.sample(range(len(movie_data)), min(len(movie_data), args.queries_per_domain * 2))
    
    for idx in sampled_indices:
        row = movie_data[idx]
        row_id = str(row.get('row_id', idx))
        queries = generate_queries_from_row(row, 'movie', row_id)
        movie_queries.extend(queries)
        if len(movie_queries) >= args.queries_per_domain:
            break
    
    # Select diverse queries
    movie_queries = movie_queries[:args.queries_per_domain]
    all_queries.extend(movie_queries)
    print(f"Generated {len(movie_queries)} movie queries")
    
    # Save queries
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for query in all_queries:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')
    
    print(f"\nSaved {len(all_queries)} queries to {output_path}")
    print(f"  - Product: {len(product_queries)} queries")
    print(f"  - Movie: {len(movie_queries)} queries")

if __name__ == '__main__':
    main()
