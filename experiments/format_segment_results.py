#!/usr/bin/env python3
"""
Format segment retrieval results in a readable format similar to masked_prediction_results.txt
Shows queries, ground truth rows, and top-k retrieved rows.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List

def format_row(row: dict, max_width: int = 80) -> str:
    """Format a row dictionary for display."""
    if not row:
        return "  (empty)"
    
    lines = []
    for key, value in row.items():
        if key == 'row_id':
            continue
        value_str = str(value)
        if len(value_str) > max_width:
            value_str = value_str[:max_width] + "..."
        lines.append(f"    {key}: {value_str}")
    
    return "\n".join(lines) if lines else "  (empty)"

def format_query(query_dict: dict) -> str:
    """Format structured query for display."""
    parts = []
    for key, value in query_dict.items():
        parts.append(f"  {key}: {value}")
    return "\n".join(parts)

def main():
    parser = argparse.ArgumentParser(description='Format segment retrieval results')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to results JSON file')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Output formatted results file')
    parser.add_argument('--model', type=str, choices=['bert', 'navi', 'both'], default='both',
                       help='Which model results to format')
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    # Organize by domain and model
    output_lines = []
    output_lines.append("=" * 100)
    output_lines.append("Segment-wise Retrieval Results")
    output_lines.append("=" * 100)
    output_lines.append("")
    
    # Group by domain
    domains = {}
    for result in results:
        domain = result['domain']
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(result)
    
    # Format each domain
    for domain in ['product', 'movie']:
        if domain not in domains:
            continue
        
        domain_results = domains[domain]
        output_lines.append("=" * 100)
        output_lines.append(f"DOMAIN: {domain.upper()}")
        output_lines.append("=" * 100)
        output_lines.append("")
        
        for i, result in enumerate(domain_results, 1):
            query_id = result['query_id']
            query = result['query']
            target_attr = result['target_attribute']
            gt_row = result['ground_truth_row']
            gt_value = result['ground_truth_value']
            
            output_lines.append(f"Query {i}: {query_id}")
            output_lines.append("-" * 100)
            output_lines.append("Structured Query:")
            output_lines.append(format_query(query))
            output_lines.append("")
            output_lines.append(f"Target Attribute: {target_attr}")
            output_lines.append(f"Ground Truth Value: {gt_value}")
            output_lines.append("")
            output_lines.append("Ground Truth Row:")
            output_lines.append(format_row(gt_row))
            output_lines.append("")
            
            # BERT results
            if args.model in ['bert', 'both'] and 'bert_retrieved_rows' in result:
                output_lines.append("BERT Retrieved Rows (Top-5):")
                output_lines.append("-" * 100)
                bert_rows = result['bert_retrieved_rows']
                if bert_rows:
                    for j, retrieved in enumerate(bert_rows[:5], 1):
                        output_lines.append(f"  Rank {j} (score: {retrieved['score']:.4f}, row_id: {retrieved['row_id']}):")
                        output_lines.append(format_row(retrieved['row']))
                        output_lines.append("")
                else:
                    output_lines.append("  (no results)")
                output_lines.append("")
            
            # NAVI results
            if args.model in ['navi', 'both'] and 'navi_retrieved_rows' in result:
                output_lines.append("NAVI Retrieved Rows (Top-5):")
                output_lines.append("-" * 100)
                navi_rows = result['navi_retrieved_rows']
                if navi_rows:
                    for j, retrieved in enumerate(navi_rows[:5], 1):
                        output_lines.append(f"  Rank {j} (score: {retrieved['score']:.4f}, row_id: {retrieved['row_id']}):")
                        output_lines.append(format_row(retrieved['row']))
                        output_lines.append("")
                else:
                    output_lines.append("  (no results)")
                output_lines.append("")
            
            output_lines.append("")
            output_lines.append("=" * 100)
            output_lines.append("")
    
    # Write output
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))
    
    print(f"Formatted results saved to {output_path}")

if __name__ == '__main__':
    main()
