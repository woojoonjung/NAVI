import argparse
import json
import os
from typing import List, Dict
import re
from collections import defaultdict

def normalize_text(text: str) -> str:
    """Normalize text for comparison by removing extra whitespace and converting to lowercase."""
    return re.sub(r'\s+', ' ', text.lower().strip())

def exact_match(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer exactly matches ground truth (after normalization)."""
    return normalize_text(predicted) == normalize_text(ground_truth)

def contains_match(predicted: str, ground_truth: str) -> bool:
    """Check if predicted answer contains the ground truth (after normalization)."""
    pred_norm = normalize_text(predicted)
    gt_norm = normalize_text(ground_truth)
    return gt_norm in pred_norm

def evaluate_predictions(predictions_file: str, labels_file: str) -> Dict[str, float]:
    """Evaluate predictions against ground truth labels."""
    
    # Load predictions
    predictions = {}
    with open(predictions_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            predictions[data['query_id']] = data['answer']
    
    # Load ground truth - expand to include all question variants
    ground_truth = {}
    with open(labels_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Handle question field - process all questions if it's a list
            questions = data['question']
            if not isinstance(questions, list):
                questions = [questions]  # Convert single question to list
            
            # Create entries for each question variant
            for question_idx, question_text in enumerate(questions):
                variant_query_id = f"{data['id']}_{question_idx}"
                ground_truth[variant_query_id] = data['answer']
    
    # Calculate metrics
    exact_matches = 0
    contains_matches = 0
    total = 0
    
    for query_id in ground_truth:
        if query_id in predictions:
            total += 1
            pred = predictions[query_id]
            gt = ground_truth[query_id]
            
            if exact_match(pred, gt):
                exact_matches += 1
                contains_matches += 1  # Exact match implies contains match
            elif contains_match(pred, gt):
                contains_matches += 1
    
    metrics = {
        'exact_match': exact_matches / total if total > 0 else 0.0,
        'contains_match': contains_matches / total if total > 0 else 0.0,
        'total_queries': total,
        'coverage': total / len(ground_truth) if len(ground_truth) > 0 else 0.0
    }
    
    return metrics

def analyze_query_types(predictions_file: str, labels_file: str) -> Dict[str, Dict[str, float]]:
    """Analyze performance by query type based on question patterns."""
    
    # Load predictions and ground truth
    predictions = {}
    with open(predictions_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            predictions[data['query_id']] = data['answer']
    
    ground_truth = {}
    with open(labels_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            ground_truth[data['id']] = data
    
    # Categorize queries by type - expand to include all question variants
    query_types = {
        'attribute_lookup': [],
        'variant_phrasing': [],
        'open_ended': []
    }
    
    for query_id, gt_data in ground_truth.items():
        # Handle question field - process all questions if it's a list
        questions = gt_data['question']
        if not isinstance(questions, list):
            questions = [questions]  # Convert single question to list
        
        # Process each question variant
        for question_idx, question_text in enumerate(questions):
            variant_query_id = f"{query_id}_{question_idx}"
            question = question_text.lower()
            
            # Simple heuristics to categorize query types
            if any(word in question for word in ['which', 'what brand', 'what make']):
                query_types['attribute_lookup'].append(variant_query_id)
            elif any(word in question for word in ['how much', 'cost', 'price']):
                query_types['variant_phrasing'].append(variant_query_id)
            elif any(word in question for word in ['suggest', 'recommend', 'looking for', 'can you']):
                query_types['open_ended'].append(variant_query_id)
            else:
                # Default to attribute lookup if unclear
                query_types['attribute_lookup'].append(variant_query_id)
    
    # Evaluate each query type
    results = {}
    for query_type, query_ids in query_types.items():
        if not query_ids:
            continue
            
        exact_matches = 0
        contains_matches = 0
        total = 0
        
        for query_id in query_ids:
            if query_id in predictions:
                total += 1
                pred = predictions[query_id]
                # Get the original query ID to find the ground truth answer
                original_query_id = query_id.split('_')[0]
                gt = ground_truth[original_query_id]['answer']
                
                if exact_match(pred, gt):
                    exact_matches += 1
                    contains_matches += 1
                elif contains_match(pred, gt):
                    contains_matches += 1
        
        if total > 0:
            results[query_type] = {
                'exact_match': exact_matches / total,
                'contains_match': contains_matches / total,
                'total_queries': total
            }
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLM QA predictions')
    parser.add_argument('--predictions_file', type=str, required=True,
                       help='Path to predictions JSONL file')
    parser.add_argument('--labels_file', type=str, required=True,
                       help='Path to ground truth labels JSONL file')
    parser.add_argument('--analyze_by_type', action='store_true',
                       help='Analyze performance by query type')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save evaluation results')
    
    args = parser.parse_args()
    
    # Evaluate overall performance
    print(f"Evaluating predictions from: {args.predictions_file}")
    print(f"Against labels from: {args.labels_file}")
    print()
    
    metrics = evaluate_predictions(args.predictions_file, args.labels_file)
    
    print("=== Overall Performance ===")
    print(f"Exact Match: {metrics['exact_match']:.3f}")
    print(f"Contains Match: {metrics['contains_match']:.3f}")
    print(f"Coverage: {metrics['coverage']:.3f} ({metrics['total_queries']} queries evaluated)")
    print()
    
    # Analyze by query type if requested
    if args.analyze_by_type:
        print("=== Performance by Query Type ===")
        type_results = analyze_query_types(args.predictions_file, args.labels_file)
        
        for query_type, results in type_results.items():
            print(f"\n{query_type.replace('_', ' ').title()}:")
            print(f"  Exact Match: {results['exact_match']:.3f}")
            print(f"  Contains Match: {results['contains_match']:.3f}")
            print(f"  Total Queries: {results['total_queries']}")
    
    # Save results if output file specified
    if args.output_file:
        results = {
            'overall': metrics,
            'by_type': analyze_query_types(args.predictions_file, args.labels_file) if args.analyze_by_type else None
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {args.output_file}")

if __name__ == '__main__':
    main()
