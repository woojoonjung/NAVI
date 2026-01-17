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

def get_bert_cls_embedding_from_structured_query(structured_query, model, tokenizer, device):
    """Get BERT CLS embedding from structured query given attributes."""
    given_attributes = extract_given_attributes(structured_query)
    
    if not given_attributes:
        return None
    
    # Create text from given attributes
    attr_parts = [f"{k}: {v}" for k, v in given_attributes.items()]
    attr_text = " | ".join(attr_parts)
    
    inputs = tokenizer(attr_text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        hidden_states = model.bert(**inputs).last_hidden_state
        cls_embedding = hidden_states[:, 0, :].detach().cpu().numpy()
    
    return cls_embedding

def get_navi_segment_embeddings_from_structured_query(structured_query, model, tokenizer, device):
    """Get Navi segment embeddings from structured query given attributes."""
    given_attributes = extract_given_attributes(structured_query)
    
    if not given_attributes:
        return None, None
    
    # Create individual segment embeddings for each given attribute
    segment_embeddings = []
    segment_info = []
    
    for header, value in given_attributes.items():
        segment_text = f"{header}: {value}"
        inputs = tokenizer(segment_text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            hidden_states = model(**inputs).last_hidden_state
            # Use CLS token for each segment
            segment_embedding = hidden_states[:, 0, :].detach().cpu().numpy()
            segment_embeddings.append(segment_embedding)
            segment_info.append({'header': header, 'value': value})
    
    return np.array(segment_embeddings), segment_info

def parse_question_to_json(llm_client, question_text):
    """Parse natural language question to structured JSON using LLM."""
    system_prompt = """You are an expert at parsing natural language questions about products into structured JSON queries.

Given a question about a product, extract the given attributes (what the user is asking about) and target attributes (what they want to find).

Schema for product data:
- name: Product name
- brand.name: Brand name  
- category: Product category
- offers.price: Price
- offers.pricecurrency: Price currency
- offers.availability: Availability
- rating: Rating
- color: Color
- size: Size
- material: Material
- description: Description

Examples:
Q: "What products have a rating of 4.5?"
A: {"rating": "4.5", "name.1": "", "name.2": "", "name.3": "", "name.4": "", "name.5": ""}

Q: "Which products cost $20 in USD?"
A: {"offers.price": "20", "offers.pricecurrency": "USD", "name.1": "", "name.2": "", "name.3": "", "name.4": "", "name.5": ""}

Q: "What are some red shirts?"
A: {"color": "red", "category": "shirts", "name.1": "", "name.2": "", "name.3": "", "name.4": "", "name.5": ""}

Return only valid JSON, no other text."""

    try:
        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question_text}
            ],
            temperature=0.1
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Try to parse as JSON
        try:
            structured_query = json.loads(response_text)
            return structured_query
        except json.JSONDecodeError:
            return {"parse_error": f"Failed to parse JSON: {response_text}"}
            
    except Exception as e:
        return {"parse_error": f"LLM call failed: {str(e)}"}

def main():
    parser = argparse.ArgumentParser(description='Hybrid retrieval: BERT (structured query → CLS) vs Navi (structured query → segment embeddings)')
    parser.add_argument('--bert_index_dir', type=str, required=True, help='Directory containing BERT FAISS index')
    parser.add_argument('--navi_index_dir', type=str, required=True, help='Directory containing Navi FAISS index')
    parser.add_argument('--query_file', type=str, required=True, help='Path to query file (JSONL)')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top documents to retrieve')
    parser.add_argument('--output_file', type=str, help='Output file path (optional)')
    parser.add_argument('--enable_struct_parse', action='store_true', help='Enable structured query parsing')
    parser.add_argument('--llm_provider', type=str, default='openai', help='LLM provider for structured parsing')
    parser.add_argument('--llm_model', type=str, default='gpt-4o-mini', help='LLM model for structured parsing')
    
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
    
    # Load LLM client if structured parsing is enabled
    llm_client = None
    if args.enable_struct_parse:
        try:
            from openai import OpenAI
            llm_client = OpenAI()
            print("LLM client loaded for structured parsing", file=sys.stderr)
        except Exception as e:
            print(f"Failed to load LLM client: {e}", file=sys.stderr)
            print("Continuing without structured parsing", file=sys.stderr)
    
    # Load queries
    with open(args.query_file, 'r') as f:
        queries = [json.loads(line) for line in f]
    
    print(f"Processing {len(queries)} queries", file=sys.stderr)
    
    # Process each query
    for query in tqdm(queries, desc="Processing queries"):
        questions = query['question']
        if not isinstance(questions, list):
            questions = [questions]
        
        for question_idx, question_text in enumerate(questions):
            # Parse question to structured query if enabled
            structured_query = None
            if llm_client is not None:
                structured_query = parse_question_to_json(llm_client, question_text)
                print(f"Structured query for {query['id']}_{question_idx}: {structured_query}", file=sys.stderr)
            
            # BERT: structured query → CLS embedding
            if structured_query and not structured_query.get('parse_error'):
                bert_embedding = get_bert_cls_embedding_from_structured_query(
                    structured_query, bert_model, bert_tokenizer, device
                )
            else:
                # Fallback to question text
                inputs = bert_tokenizer(question_text, return_tensors='pt', truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    hidden_states = bert_model.bert(**inputs).last_hidden_state
                    bert_embedding = hidden_states[:, 0, :].detach().cpu().numpy()
            
            # Navi: structured query → segment embeddings
            if structured_query and not structured_query.get('parse_error'):
                navi_segment_embeddings, segment_info = get_navi_segment_embeddings_from_structured_query(
                    structured_query, navi_model, navi_tokenizer, device
                )
            else:
                # Fallback to question text CLS
                inputs = navi_tokenizer(question_text, return_tensors='pt', truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    hidden_states = navi_model(**inputs).last_hidden_state
                    navi_segment_embeddings = hidden_states[:, 0, :].detach().cpu().numpy().reshape(1, -1)
                    segment_info = [{'header': 'question', 'value': question_text}]
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(bert_embedding)
            faiss.normalize_L2(navi_segment_embeddings)
            
            # BERT retrieval
            bert_distances, bert_indices = bert_index.search(bert_embedding, args.top_k)
            bert_retrieved_docs = []
            for i in range(len(bert_distances[0])):
                doc_idx = bert_indices[0][i]
                doc_metadata = bert_metadata[doc_idx]
                bert_retrieved_docs.append({
                    'table_id': doc_metadata['table_id'],
                    'row_id': doc_metadata['row_id'],
                    'content': doc_metadata['full_row'],
                    'score': float(bert_distances[0][i])
                })
            
            # Navi retrieval - use mean of segment embeddings
            navi_mean_embedding = np.mean(navi_segment_embeddings, axis=0).reshape(1, -1)
            faiss.normalize_L2(navi_mean_embedding)
            navi_distances, navi_indices = navi_index.search(navi_mean_embedding, args.top_k)
            navi_retrieved_docs = []
            for i in range(len(navi_distances[0])):
                doc_idx = navi_indices[0][i]
                doc_metadata = navi_metadata[doc_idx]
                navi_retrieved_docs.append({
                    'table_id': doc_metadata['table_id'],
                    'row_id': doc_metadata['row_id'],
                    'content': doc_metadata['full_row'],
                    'score': float(navi_distances[0][i])
                })
            
            # Create output
            variant_query_id = f"{query['id']}_{question_idx}"
            output = {
                'query_id': variant_query_id,
                'original_query_id': query['id'],
                'question_idx': question_idx,
                'question_text': question_text,
                'structured_query': structured_query,
                'bert_retrieved_docs': bert_retrieved_docs,
                'navi_retrieved_docs': navi_retrieved_docs,
                'segment_info': segment_info
            }
            
            # Output to stdout or file
            if args.output_file:
                with open(args.output_file, 'a') as f:
                    f.write(json.dumps(output) + '\n')
            else:
                print(json.dumps(output))

if __name__ == '__main__':
    main()
