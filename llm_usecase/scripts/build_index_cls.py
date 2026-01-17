import os
import faiss
import argparse
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer
from model.navi import NaviForMaskedLM
import torch
import numpy as np
import json
import sys
sys.path.append('/home/work/.default/woojun/navi')

def create_row_text(row_data):
    """
    Create a text representation of a table row for CLS embedding.
    """
    # Skip metadata fields
    skip_fields = {'row_id', 'table_id'}
    
    # Create a structured text representation
    text_parts = []
    for key, value in row_data.items():
        if key in skip_fields or value is None or value == "" or value == "nan":
            continue
        
        # Convert value to string and clean it
        value_str = str(value).strip()
        if not value_str or value_str.lower() == 'none':
            continue
            
        text_parts.append(f"{key}: {value_str}")
    
    return " | ".join(text_parts)

def main(args):
    # Load table data - use WDC product data
    table_file = os.path.join(args.data_dir, 'WDC_product_for_cls_cleaned.jsonl')
    with open(table_file, 'r') as f:
        table_data = [json.loads(line) for line in f]
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", file=sys.stderr)
    
    if args.model_type == 'bert':
        # Load fine-tuned BERT model
        model_path = '/home/work/.default/woojun/navi/models/bert_quarter_product/epoch_2'
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained(model_path, local_files_only=True)
        model.to(device)
        model.eval()
        print(f"Loaded fine-tuned BERT from {model_path}", file=sys.stderr)
    elif args.model_type == 'ours':
        # Load Navi model
        model_path = '/home/work/.default/woojun/navi/models/full_Quarter_Product_HVB_hv0p8_align0p5_vr0p5/epoch_2'
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = NaviForMaskedLM(model_path=model_path)
        model.to(device)
        model.eval()
        print(f"Loaded Navi model from {model_path}", file=sys.stderr)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Generate CLS embeddings for all rows
    print("Generating CLS embeddings...", file=sys.stderr)
    all_embeddings = []
    all_metadata = []
    
    with torch.no_grad():
        for i, row_data in enumerate(tqdm(table_data, desc="Processing rows")):
            # Create text representation of the row
            row_text = create_row_text(row_data)
            
            if not row_text.strip():
                continue
                
            # Tokenize and get embeddings
            inputs = tokenizer(row_text, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            if args.model_type == 'bert':
                hidden_states = model.bert(**inputs).last_hidden_state
            else:  # Navi
                hidden_states = model(**inputs).last_hidden_state
            
            # Use CLS token embedding
            row_embedding = hidden_states[:, 0, :].detach().cpu().numpy()
            
            all_embeddings.append(row_embedding)
            all_metadata.append({
                'table_id': 0,
                'row_id': row_data.get('row_id', i),
                'full_row': row_data
            })

    if not all_embeddings:
        print("No embeddings generated!", file=sys.stderr)
        return

    # Convert to numpy array
    embeddings_array = np.vstack(all_embeddings)
    print(f"Generated {len(all_embeddings)} CLS embeddings with dimension {embeddings_array.shape[1]}", file=sys.stderr)

    # Create FAISS index
    print("Creating FAISS index...", file=sys.stderr)
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings_array)
    index.add(embeddings_array)

    # Save index and metadata
    faiss.write_index(index, os.path.join(args.output_dir, 'index.faiss'))
    
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(all_metadata, f, indent=2)

    print(f"Index saved to {args.output_dir}", file=sys.stderr)
    print(f"Index contains {index.ntotal} vectors", file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True, choices=['bert', 'ours'])
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)
