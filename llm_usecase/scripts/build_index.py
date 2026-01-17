#!/usr/bin/env python3

import os
import sys
import json
import argparse
import faiss
import torch
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer, AutoModel
import sys

# Add the project root to the path
sys.path.append('/home/work/.default/woojun/navi')

from dataset.dataset import NaviDataset, BertDataset
from model.navi import NaviForMaskedLM

def main(args):
    # Map dataset names to data files
    dataset_files = {
        'wdc_product': 'WDC_product_for_cls_cleaned.jsonl',
        'amazon_product': 'Amazon_product_for_cls_cleaned.jsonl'
    }
    
    # Load table data
    table_file = os.path.join(args.data_dir, 'tables_cleaned', dataset_files[args.dataset])
    with open(table_file, 'r') as f:
        table_data = [json.loads(line) for line in f]
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", file=sys.stderr)
    
    # Map dataset to model paths
    model_paths = {
        'wdc_product': {
            'bert': '/home/work/.default/woojun/navi/models/bert_quarter_product/epoch_2',
            'ours': '/home/work/.default/woojun/navi/models/full_Quarter_Product_HVB_hv0p8_align0p5_vr0p5/epoch_2'
        },
        'amazon_product': {
            'bert': '/home/work/.default/woojun/navi/models/bert_quarter_product/epoch_2',
            'ours': '/home/work/.default/woojun/navi/models/full_Quarter_Product_HVB_hv0p8_align0p5_vr0p5/epoch_2'
        }
    }
    
    if args.model_type == 'bert':
        # Load fine-tuned BERT model
        model_path = model_paths[args.dataset]['bert']
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained(model_path, local_files_only=True)
        model.to(device)
        model.eval()
        print(f"Loaded fine-tuned BERT from {model_path}", file=sys.stderr)
        
        # Create BERT dataset
        dataset = BertDataset(table_data, tokenizer_name='bert-base-uncased', max_length=512)
        
    elif args.model_type == 'ours':
        # Load Navi model
        model_path = model_paths[args.dataset]['ours']
        model = NaviForMaskedLM(model_path=model_path)
        model.to(device)
        model.eval()
        print(f"Loaded Navi model from {model_path}", file=sys.stderr)
        
        # Create Navi dataset
        dataset = NaviDataset(table_data, tokenizer_name='bert-base-uncased', max_length=512, ablation_mode="full")
        
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Use CPU FAISS index for now (GPU FAISS requires additional setup)
    index = faiss.IndexFlatL2(model.config.hidden_size)
    print(f"Created FAISS index with dimension {model.config.hidden_size}", file=sys.stderr)
    
    metadata = []
    embeddings = []
    
    print("Generating embeddings...", file=sys.stderr)
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Processing rows"):
            # Get data from dataset
            data = dataset[i]
            
            # Get row_id from original data
            row_id = table_data[i].get('row_id', i)
            
            # Generate embedding based on model type
            if args.model_type == 'bert':
                # For BERT, use the input_ids directly
                input_ids = data['input_ids'].unsqueeze(0).to(device)
                attention_mask = data['attention_mask'].unsqueeze(0).to(device)
                
                # Get BERT embeddings
                outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
                # Use CLS token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
            elif args.model_type == 'ours':
                # For Navi, use the model's forward pass
                input_ids = data['input_ids'].unsqueeze(0).to(device)
                attention_mask = data['attention_mask'].unsqueeze(0).to(device)
                position_ids = data['position_ids'].unsqueeze(0).to(device)
                segment_ids = data['segment_ids'].unsqueeze(0).to(device)
                
                # Get Navi embeddings
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    segment_ids=segment_ids
                )
                # Use CLS token embedding (Navi returns tuple: (contextualized_embeddings, logits))
                embedding = outputs[0][:, 0, :].cpu().numpy()
            
            embeddings.append(embedding.flatten())
            metadata.append({
                'row_id': row_id,
                'full_row': table_data[i]
            })
    
    # Convert to numpy array
    embeddings = np.array(embeddings).astype('float32')
    print(f"Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}", file=sys.stderr)
    
    # Add embeddings to FAISS index
    print("Creating FAISS index...", file=sys.stderr)
    index.add(embeddings)
    
    # Save index
    faiss.write_index(index, os.path.join(args.output_dir, 'index.faiss'))
    
    # Save metadata
    with open(os.path.join(args.output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
    
    print(f"Index saved to {args.output_dir}", file=sys.stderr)
    print(f"Index contains {index.ntotal} vectors", file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True, choices=['bert', 'ours'])
    parser.add_argument('--dataset', type=str, required=True, choices=['wdc_product', 'amazon_product'])
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)