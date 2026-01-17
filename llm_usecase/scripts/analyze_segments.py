#!/usr/bin/env python3

import os
import sys
import json
import argparse
import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModel
from dotenv import load_dotenv
from collections import defaultdict, Counter
import re

# Add the project root to the path
sys.path.append('/home/work/.default/woojun/navi')

def load_dotenv():
    """Load environment variables from .env file if it exists."""
    env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(env_path):
        from dotenv import load_dotenv
        load_dotenv(env_path)

def load_index(index_dir):
    """Load FAISS index and metadata."""
    index_path = os.path.join(index_dir, 'index.faiss')
    metadata_path = os.path.join(index_dir, 'metadata.json')
    
    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Index files not found in {index_dir}")
    
    index = faiss.read_index(index_path)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return index, metadata

def load_model_and_tokenizer(model_type, device):
    """Load model and tokenizer based on model type."""
    if model_type == 'bert':
        model_path = '/home/work/.default/woojun/navi/models/bert_quarter_product/epoch_2'
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained(model_path, local_files_only=True)
        model.to(device)
        model.eval()
        print(f"Loaded BERT model from {model_path}", file=sys.stderr)
    elif model_type == 'ours':
        model_path = '/home/work/.default/woojun/navi/models/full_Quarter_Product_HVB_hv0p8_align0p5_vr0p5/epoch_2'
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained(model_path, local_files_only=True)
        model.to(device)
        model.eval()
        print(f"Loaded Navi model from {model_path}", file=sys.stderr)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, tokenizer

def extract_given_attributes(structured_query):
    """Extract only the given attributes (non-empty values) from structured query."""
    if not structured_query or 'parse_error' in structured_query:
        return {}
    
    # Filter out parse errors and metadata
    query_data = {k: v for k, v in structured_query.items() 
                  if k not in ['raw_question', 'parse_error', 'raw_response']}
    
    # Only keep attributes with non-empty values (given attributes)
    given_attributes = {k: v for k, v in query_data.items() if v and v != ""}
    
    return given_attributes

def get_segment_embedding_bert(segment_text, model, tokenizer, device):
    """Get BERT embedding for a single segment."""
    inputs = tokenizer(segment_text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        hidden_states = model.bert(**inputs).last_hidden_state
        embedding = hidden_states[:, 0, :].detach().cpu().numpy()  # CLS token
    
    return embedding

def get_segment_embedding_navi(segment_text, model, tokenizer, device):
    """Get Navi embedding for a single segment."""
    inputs = tokenizer(segment_text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        hidden_states = model(**inputs).last_hidden_state
        embedding = hidden_states[:, 0, :].detach().cpu().numpy()  # CLS token
    
    return embedding

def classify_segment_type(header, value):
    """Classify segment into different types for analysis."""
    segment_types = []
    
    # Currency segments
    if 'currency' in header.lower() or 'pricecurrency' in header.lower():
        segment_types.append('currency')
    
    # Brand segments
    if 'brand' in header.lower():
        segment_types.append('brand')
    
    # Category segments
    if 'category' in header.lower():
        segment_types.append('category')
    
    # Price segments
    if 'price' in header.lower() and 'currency' not in header.lower():
        segment_types.append('price')
    
    # Rating segments
    if 'rating' in header.lower():
        segment_types.append('rating')
    
    # Name segments
    if 'name' in header.lower():
        segment_types.append('name')
    
    # Color segments
    if 'color' in header.lower():
        segment_types.append('color')
    
    # Size segments
    if 'size' in header.lower():
        segment_types.append('size')
    
    # Availability segments
    if 'availability' in header.lower():
        segment_types.append('availability')
    
    # Numerical value detection
    if re.match(r'^\d+(\.\d+)?$', str(value).strip()):
        segment_types.append('numerical')
    
    # Text value detection
    if isinstance(value, str) and len(value) > 3 and not re.match(r'^\d+(\.\d+)?$', value.strip()):
        segment_types.append('text')
    
    return segment_types if segment_types else ['other']

def evaluate_segment_retrieval(segment_text, segment_type, retrieved_docs, ground_truth):
    """Evaluate how well a segment retrieval matches the ground truth."""
    gt_lower = ground_truth.lower()
    
    # Check if ground truth appears in any retrieved document
    exact_match = False
    partial_match = False
    semantic_match = False
    
    for doc in retrieved_docs:
        doc_content = str(doc['content']).lower()
        
        # Exact match
        if gt_lower in doc_content:
            exact_match = True
            break
        
        # Partial match (for multi-word ground truth)
        if ' ' in gt_lower:
            words = gt_lower.split()
            if all(word in doc_content for word in words):
                partial_match = True
                break
        
        # Semantic match (for specific segment types)
        if segment_type == 'currency':
            if any(currency in doc_content for currency in ['usd', 'eur', 'gbp', 'aud', 'cad']):
                if gt_lower in ['usd', 'eur', 'gbp', 'aud', 'cad']:
                    semantic_match = True
                    break
        elif segment_type == 'brand':
            if any(brand in doc_content for brand in ['timberland', 'nike', 'adidas', 'levi', 'gucci']):
                if any(brand in gt_lower for brand in ['timberland', 'nike', 'adidas', 'levi', 'gucci']):
                    semantic_match = True
                    break
    
    return {
        'exact_match': exact_match,
        'partial_match': partial_match,
        'semantic_match': semantic_match,
        'any_match': exact_match or partial_match or semantic_match
    }

def analyze_segment_performance(queries, bert_model, bert_tokenizer, navi_model, navi_tokenizer, 
                               bert_index, bert_metadata, navi_index, navi_metadata, device):
    """Analyze segment-wise performance for both models."""
    
    results = {
        'segment_analysis': defaultdict(list),
        'model_comparison': {'navi_better': 0, 'bert_better': 0, 'both_correct': 0, 'both_wrong': 0},
        'segment_type_performance': defaultdict(lambda: {'bert': {'correct': 0, 'total': 0}, 'navi': {'correct': 0, 'total': 0}}),
        'detailed_results': []
    }
    
    for query in tqdm(queries, desc="Analyzing segments"):
        questions = query['question']
        if not isinstance(questions, list):
            questions = [questions]
        
        for question_idx, question_text in enumerate(questions):
            variant_query_id = f"{query['id']}_{question_idx}"
            ground_truth = query['answer']
            
            # Parse question to get structured query (simplified for now)
            # In a real implementation, you'd use the LLM parsing here
            structured_query = extract_given_attributes_from_question(question_text)
            
            if not structured_query:
                continue
            
            # Analyze each segment
            for header, value in structured_query.items():
                segment_text = f"{header}: {value}"
                segment_types = classify_segment_type(header, value)
                
                # Get embeddings
                bert_embedding = get_segment_embedding_bert(segment_text, bert_model, bert_tokenizer, device)
                navi_embedding = get_segment_embedding_navi(segment_text, navi_model, navi_tokenizer, device)
                
                # Normalize for cosine similarity
                faiss.normalize_L2(bert_embedding)
                faiss.normalize_L2(navi_embedding)
                
                # Retrieve documents
                bert_distances, bert_indices = bert_index.search(bert_embedding, 5)
                navi_distances, navi_indices = navi_index.search(navi_embedding, 5)
                
                # Convert to document format
                bert_docs = []
                for i in range(len(bert_distances[0])):
                    doc_idx = bert_indices[0][i]
                    doc_metadata = bert_metadata[doc_idx]
                    bert_docs.append({
                        'content': doc_metadata['full_row'],
                        'score': float(bert_distances[0][i])
                    })
                
                navi_docs = []
                for i in range(len(navi_distances[0])):
                    doc_idx = navi_indices[0][i]
                    doc_metadata = navi_metadata[doc_idx]
                    navi_docs.append({
                        'content': doc_metadata['full_row'],
                        'score': float(navi_distances[0][i])
                    })
                
                # Evaluate segment performance
                bert_eval = evaluate_segment_retrieval(segment_text, segment_types[0], bert_docs, ground_truth)
                navi_eval = evaluate_segment_retrieval(segment_text, segment_types[0], navi_docs, ground_truth)
                
                # Store results
                segment_result = {
                    'query_id': variant_query_id,
                    'question': question_text,
                    'segment_text': segment_text,
                    'segment_types': segment_types,
                    'ground_truth': ground_truth,
                    'bert_eval': bert_eval,
                    'navi_eval': navi_eval,
                    'bert_docs': bert_docs[:3],  # Top 3 for analysis
                    'navi_docs': navi_docs[:3]
                }
                
                results['detailed_results'].append(segment_result)
                
                # Update segment type performance
                for segment_type in segment_types:
                    results['segment_type_performance'][segment_type]['bert']['total'] += 1
                    results['segment_type_performance'][segment_type]['navi']['total'] += 1
                    
                    if bert_eval['any_match']:
                        results['segment_type_performance'][segment_type]['bert']['correct'] += 1
                    if navi_eval['any_match']:
                        results['segment_type_performance'][segment_type]['navi']['correct'] += 1
                
                # Update model comparison
                if navi_eval['any_match'] and not bert_eval['any_match']:
                    results['model_comparison']['navi_better'] += 1
                elif bert_eval['any_match'] and not navi_eval['any_match']:
                    results['model_comparison']['bert_better'] += 1
                elif navi_eval['any_match'] and bert_eval['any_match']:
                    results['model_comparison']['both_correct'] += 1
                else:
                    results['model_comparison']['both_wrong'] += 1
    
    return results

def extract_given_attributes_from_question(question_text):
    """Simplified extraction of attributes from question text."""
    # This is a simplified version - in practice, you'd use the LLM parsing
    attributes = {}
    
    # Currency detection
    if 'usd' in question_text.lower():
        attributes['offers.pricecurrency'] = 'USD'
    if 'eur' in question_text.lower():
        attributes['offers.pricecurrency'] = 'EUR'
    
    # Brand detection
    if 'timberland' in question_text.lower():
        attributes['brand.name'] = 'Timberland'
    if 'nike' in question_text.lower():
        attributes['brand.name'] = 'Nike'
    if 'adidas' in question_text.lower():
        attributes['brand.name'] = 'Adidas'
    
    # Category detection
    if 'jeans' in question_text.lower():
        attributes['category'] = 'jeans'
    if 'shoes' in question_text.lower():
        attributes['category'] = 'shoes'
    if 'jacket' in question_text.lower():
        attributes['category'] = 'jackets'
    
    return attributes

def generate_segment_report(results, output_file):
    """Generate a comprehensive segment analysis report."""
    
    report = {
        'summary': {
            'total_segments': len(results['detailed_results']),
            'navi_better': results['model_comparison']['navi_better'],
            'bert_better': results['model_comparison']['bert_better'],
            'both_correct': results['model_comparison']['both_correct'],
            'both_wrong': results['model_comparison']['both_wrong']
        },
        'segment_type_performance': {},
        'navi_strengths': [],
        'bert_strengths': [],
        'detailed_analysis': results['detailed_results']
    }
    
    # Calculate segment type performance
    for segment_type, perf in results['segment_type_performance'].items():
        if perf['bert']['total'] > 0 and perf['navi']['total'] > 0:
            bert_acc = perf['bert']['correct'] / perf['bert']['total']
            navi_acc = perf['navi']['correct'] / perf['navi']['total']
            
            report['segment_type_performance'][segment_type] = {
                'bert_accuracy': bert_acc,
                'navi_accuracy': navi_acc,
                'navi_advantage': navi_acc - bert_acc,
                'total_segments': perf['bert']['total']
            }
    
    # Identify Navi strengths
    for segment_type, perf in report['segment_type_performance'].items():
        if perf['navi_advantage'] > 0.1:  # Navi is at least 10% better
            report['navi_strengths'].append({
                'segment_type': segment_type,
                'navi_accuracy': perf['navi_accuracy'],
                'bert_accuracy': perf['bert_accuracy'],
                'advantage': perf['navi_advantage']
            })
    
    # Identify BERT strengths
    for segment_type, perf in report['segment_type_performance'].items():
        if perf['navi_advantage'] < -0.1:  # BERT is at least 10% better
            report['bert_strengths'].append({
                'segment_type': segment_type,
                'navi_accuracy': perf['navi_accuracy'],
                'bert_accuracy': perf['bert_accuracy'],
                'advantage': -perf['navi_advantage']
            })
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def main():
    parser = argparse.ArgumentParser(description='Analyze segment-wise performance for Navi vs BERT')
    parser.add_argument('--bert_index_dir', type=str, required=True, help='Directory containing BERT FAISS index')
    parser.add_argument('--navi_index_dir', type=str, required=True, help='Directory containing Navi FAISS index')
    parser.add_argument('--query_file', type=str, required=True, help='Path to query file (JSONL)')
    parser.add_argument('--output_file', type=str, required=True, help='Output file for analysis report')
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", file=sys.stderr)
    
    # Load indices
    print(f"Loading BERT index from {args.bert_index_dir}", file=sys.stderr)
    bert_index, bert_metadata = load_index(args.bert_index_dir)
    print(f"BERT index loaded with {bert_index.ntotal} vectors", file=sys.stderr)
    
    print(f"Loading Navi index from {args.navi_index_dir}", file=sys.stderr)
    navi_index, navi_metadata = load_index(args.navi_index_dir)
    print(f"Navi index loaded with {navi_index.ntotal} vectors", file=sys.stderr)
    
    # Load models
    bert_model, bert_tokenizer = load_model_and_tokenizer('bert', device)
    navi_model, navi_tokenizer = load_model_and_tokenizer('ours', device)
    
    # Load queries
    with open(args.query_file, 'r') as f:
        queries = [json.loads(line) for line in f]
    
    print(f"Analyzing {len(queries)} queries", file=sys.stderr)
    
    # Analyze segment performance
    results = analyze_segment_performance(
        queries, bert_model, bert_tokenizer, navi_model, navi_tokenizer,
        bert_index, bert_metadata, navi_index, navi_metadata, device
    )
    
    # Generate report
    report = generate_segment_report(results, args.output_file)
    
    # Print summary
    print("\n=== SEGMENT ANALYSIS SUMMARY ===")
    print(f"Total segments analyzed: {report['summary']['total_segments']}")
    print(f"Navi better: {report['summary']['navi_better']}")
    print(f"BERT better: {report['summary']['bert_better']}")
    print(f"Both correct: {report['summary']['both_correct']}")
    print(f"Both wrong: {report['summary']['both_wrong']}")
    
    print("\n=== NAVI STRENGTHS ===")
    for strength in report['navi_strengths']:
        print(f"{strength['segment_type']}: Navi {strength['navi_accuracy']:.2f} vs BERT {strength['bert_accuracy']:.2f} (+{strength['advantage']:.2f})")
    
    print("\n=== BERT STRENGTHS ===")
    for strength in report['bert_strengths']:
        print(f"{strength['segment_type']}: BERT {strength['bert_accuracy']:.2f} vs Navi {strength['navi_accuracy']:.2f} (+{strength['advantage']:.2f})")
    
    print(f"\nDetailed report saved to: {args.output_file}")

if __name__ == '__main__':
    main()
