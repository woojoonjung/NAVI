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

def test_specific_segments():
    """Test specific segment types that Navi might excel at."""
    
    # Define test segments based on our observations
    test_segments = [
        # Currency segments
        {"text": "offers.pricecurrency: USD", "type": "currency", "expected": "usd"},
        {"text": "offers.pricecurrency: EUR", "type": "currency", "expected": "eur"},
        
        # Brand segments  
        {"text": "brand.name: Timberland", "type": "brand", "expected": "timberland"},
        {"text": "brand.name: Nike", "type": "brand", "expected": "nike"},
        {"text": "brand.name: Adidas", "type": "brand", "expected": "adidas"},
        
        # Category segments
        {"text": "category: jeans", "type": "category", "expected": "jeans"},
        {"text": "category: shoes", "type": "category", "expected": "shoes"},
        {"text": "category: jackets", "type": "category", "expected": "jackets"},
        
        # Price segments
        {"text": "offers.price: 89.0", "type": "price", "expected": "89.0"},
        {"text": "offers.price: 110.00", "type": "price", "expected": "110.00"},
        
        # Rating segments
        {"text": "rating: 4.5", "type": "rating", "expected": "4.5"},
        {"text": "rating: 3.4", "type": "rating", "expected": "3.4"},
    ]
    
    return test_segments

def evaluate_segment_retrieval(segment_text, segment_type, retrieved_docs, expected_value):
    """Evaluate how well a segment retrieval matches the expected value."""
    expected_lower = expected_value.lower()
    
    # Check if expected value appears in any retrieved document
    exact_match = False
    partial_match = False
    semantic_match = False
    
    for doc in retrieved_docs:
        doc_content = str(doc['content']).lower()
        
        # Exact match
        if expected_lower in doc_content:
            exact_match = True
            break
        
        # Partial match (for multi-word expected values)
        if ' ' in expected_lower:
            words = expected_lower.split()
            if all(word in doc_content for word in words):
                partial_match = True
                break
        
        # Semantic match (for specific segment types)
        if segment_type == 'currency':
            if any(currency in doc_content for currency in ['usd', 'eur', 'gbp', 'aud', 'cad']):
                if expected_lower in ['usd', 'eur', 'gbp', 'aud', 'cad']:
                    semantic_match = True
                    break
        elif segment_type == 'brand':
            if any(brand in doc_content for brand in ['timberland', 'nike', 'adidas', 'levi', 'gucci']):
                if any(brand in expected_lower for brand in ['timberland', 'nike', 'adidas', 'levi', 'gucci']):
                    semantic_match = True
                    break
        elif segment_type == 'category':
            if any(cat in doc_content for cat in ['jeans', 'shoes', 'jackets', 'shirts', 'dresses']):
                if any(cat in expected_lower for cat in ['jeans', 'shoes', 'jackets', 'shirts', 'dresses']):
                    semantic_match = True
                    break
    
    return {
        'exact_match': exact_match,
        'partial_match': partial_match,
        'semantic_match': semantic_match,
        'any_match': exact_match or partial_match or semantic_match
    }

def analyze_focused_segments(bert_model, bert_tokenizer, navi_model, navi_tokenizer, 
                            bert_index, bert_metadata, navi_index, navi_metadata, device):
    """Analyze specific segment types to identify Navi strengths."""
    
    test_segments = test_specific_segments()
    results = {
        'segment_results': [],
        'type_performance': defaultdict(lambda: {'bert': {'correct': 0, 'total': 0}, 'navi': {'correct': 0, 'total': 0}}),
        'navi_strengths': [],
        'bert_strengths': []
    }
    
    print(f"Testing {len(test_segments)} specific segments...", file=sys.stderr)
    
    for segment in tqdm(test_segments, desc="Testing segments"):
        segment_text = segment['text']
        segment_type = segment['type']
        expected_value = segment['expected']
        
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
        bert_eval = evaluate_segment_retrieval(segment_text, segment_type, bert_docs, expected_value)
        navi_eval = evaluate_segment_retrieval(segment_text, segment_type, navi_docs, expected_value)
        
        # Store results
        segment_result = {
            'segment_text': segment_text,
            'segment_type': segment_type,
            'expected_value': expected_value,
            'bert_eval': bert_eval,
            'navi_eval': navi_eval,
            'bert_docs': bert_docs[:3],  # Top 3 for analysis
            'navi_docs': navi_docs[:3]
        }
        
        results['segment_results'].append(segment_result)
        
        # Update type performance
        results['type_performance'][segment_type]['bert']['total'] += 1
        results['type_performance'][segment_type]['navi']['total'] += 1
        
        if bert_eval['any_match']:
            results['type_performance'][segment_type]['bert']['correct'] += 1
        if navi_eval['any_match']:
            results['type_performance'][segment_type]['navi']['correct'] += 1
    
    # Calculate performance by type
    for segment_type, perf in results['type_performance'].items():
        if perf['bert']['total'] > 0 and perf['navi']['total'] > 0:
            bert_acc = perf['bert']['correct'] / perf['bert']['total']
            navi_acc = perf['navi']['correct'] / perf['navi']['total']
            advantage = navi_acc - bert_acc
            
            if advantage > 0.1:  # Navi is at least 10% better
                results['navi_strengths'].append({
                    'type': segment_type,
                    'navi_accuracy': navi_acc,
                    'bert_accuracy': bert_acc,
                    'advantage': advantage
                })
            elif advantage < -0.1:  # BERT is at least 10% better
                results['bert_strengths'].append({
                    'type': segment_type,
                    'navi_accuracy': navi_acc,
                    'bert_accuracy': bert_acc,
                    'advantage': -advantage
                })
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Focused segment analysis for Navi vs BERT')
    parser.add_argument('--bert_index_dir', type=str, required=True, help='Directory containing BERT FAISS index')
    parser.add_argument('--navi_index_dir', type=str, required=True, help='Directory containing Navi FAISS index')
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
    
    # Analyze focused segments
    results = analyze_focused_segments(
        bert_model, bert_tokenizer, navi_model, navi_tokenizer,
        bert_index, bert_metadata, navi_index, navi_metadata, device
    )
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n=== FOCUSED SEGMENT ANALYSIS SUMMARY ===")
    print(f"Total segments tested: {len(results['segment_results'])}")
    
    print("\n=== NAVI STRENGTHS ===")
    for strength in results['navi_strengths']:
        print(f"{strength['type']}: Navi {strength['navi_accuracy']:.2f} vs BERT {strength['bert_accuracy']:.2f} (+{strength['advantage']:.2f})")
    
    print("\n=== BERT STRENGTHS ===")
    for strength in results['bert_strengths']:
        print(f"{strength['type']}: BERT {strength['bert_accuracy']:.2f} vs Navi {strength['navi_accuracy']:.2f} (+{strength['advantage']:.2f})")
    
    print("\n=== DETAILED RESULTS ===")
    for result in results['segment_results']:
        if result['navi_eval']['any_match'] and not result['bert_eval']['any_match']:
            print(f"Navi Better: {result['segment_text']} (expected: {result['expected_value']})")
            print(f"  Navi: {[doc['content'].get('name', 'N/A')[:30] for doc in result['navi_docs']]}")
            print(f"  BERT:  {[doc['content'].get('name', 'N/A')[:30] for doc in result['bert_docs']]}")
            print()
    
    print(f"Detailed report saved to: {args.output_file}")

if __name__ == '__main__':
    main()
