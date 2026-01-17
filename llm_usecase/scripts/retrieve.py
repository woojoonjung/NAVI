import os
import sys
import faiss
import json
import argparse
import torch
import numpy as np
from transformers import BertTokenizer, BertForMaskedLM
from dotenv import load_dotenv

# your codebase imports
from dataprep import BertDataset, NaviDataset
sys.path.append('/home/work/.default/woojun/navi')
from model.navi import NaviForMaskedLM

# LLM client (question -> structured JSON)
from llm_client import LLMClient

load_dotenv()


def parse_question_to_json(llm_client: LLMClient, question_text: str) -> dict:
    """
    Convert a natural-language product question into a structured JSON query.
    Example:
      "List all products in the foodsandgrocery category."
      -> {"category": "foodsandgrocery", "name.1": "", "name.2": "", ...}

    The parser aims to fill slots aligned with your schema keys.
    If the LLM returns non-JSON, fall back to {"raw_question": "..."}.
    """
    system_prompt = """
        You are a question-to-JSON parsing assistant for the product domain.
        Your task: Convert natural-language product questions into a structured JSON query aligned with a fixed schema.

        Schema keys:
        ["name","director.name","genres","aggregaterating.ratingvalue","aggregaterating.ratingcount","duration","actor.0.name","actor.1.name","actor.2.name","actor.3.name","musicby.name","creator.name","description","datepublished","contentrating","language","country","page_url"]

        Parsing rules:
        1. Identify all attributes mentioned in the question.
        - If the attribute is provided in the question (e.g., "Action"), record it as "attribute": "value".
        - If the attribute is being asked about (the target of the question), include it with an empty string "" as the value.
        2. Always include the product name as "name" when present.
        3. If multiple attributes are asked about, include all as empty string "" values.
        4. Return ONLY valid JSON. No explanations, no text outside JSON.

        Format of structured query:
        {
            "given_attribute_1": "value_1",
            "given_attribute_2": "value_2",
            ...
            "target_attribute.1": "value_target_attribute.1",
            "target_attribute.2": "value_target_attribute.2",
            ...
        }

        Example with query types:
        Q1 (single attribute - single target): "What is the rating of The Dark Knight?"
        → {"name": "The Dark Knight", "aggregaterating.ratingvalue": ""}

        Q2 (single attribute - multiple targets): "What is the director and genre of The Dark Knight?"
        → {"name": "The Dark Knight", "director.name": "", "genres": ""}

        Q3 (single attribute - single target with multiple entries): "List all products in the Electronics category."
        → {"genres": "Action", "name.1": "", "name.2": "", ...}

        Q4 (multiple attributes - single target): "What is the price of products made by Apple?"
        → {"brand": "Apple", "price": ""}

        Q5 (multiple attributes - single target with multiple entries): "What products made by Apple are in the Electronics category?"
        → {"brand": "Apple", "category": "Electronics", "name.1": "", "name.2": "", ...}
    """
    user_prompt = f"Question: {question_text}\n\nReturn JSON:"

    try:
        resp = llm_client.chat(system_prompt, user_prompt)
    except Exception as e:
        return {"raw_question": question_text, "parse_error": str(e)}

    try:
        parsed = json.loads(resp)
        # sanity: ensure we at least return title or raw_question
        if not isinstance(parsed, dict):
            return {"raw_question": question_text, "parse_error": "non-dict JSON"}
        return parsed
    except Exception:
        # fallback if LLM returns non-JSON
        return {"raw_question": question_text, "parse_error": "non-JSON response", "raw_response": resp}


def extract_header_value_embeddings_from_positions(embeddings, header_positions, value_positions):
    """
    Extracts header and value embeddings from contextualized embeddings based on their token positions.
    This is a utility function adapted for inference, assuming a batch size of 1.
    """
    hidden_size = embeddings.size(-1)
    
    # Assuming batch size is 1, so we access the first element
    embeddings = embeddings.squeeze(0) # From (1, seq_len, hidden) to (seq_len, hidden)
    
    header_pos_dict = header_positions[0] if isinstance(header_positions, list) else header_positions
    value_pos_dict = value_positions[0] if isinstance(value_positions, list) else value_positions

    header_keys = list(header_pos_dict.keys())
    num_headers = len(header_keys)
    
    header_embeds = torch.zeros((1, num_headers, hidden_size), device=embeddings.device)
    val_embeds = torch.zeros((1, num_headers, hidden_size), device=embeddings.device)
    
    for i, header_name in enumerate(header_keys):
        # Extract header embedding
        h_token_indices = header_pos_dict.get(header_name, [])
        if h_token_indices:
            header_tokens = embeddings[h_token_indices]
            header_embeds[0, i] = header_tokens.mean(dim=0)
            
        # Extract value embedding
        v_token_indices = value_pos_dict.get(header_name, [])
        if v_token_indices:
            value_tokens = embeddings[v_token_indices]
            val_embeds[0, i] = value_tokens.mean(dim=0)
            
    return header_embeds, val_embeds


def vote_retrieval_results(segment_results: list, top_k: int) -> list:
    """
    Combine retrieval results from multiple segments using voting.
    
    Args:
        segment_results: List of dicts, each containing 'distances', 'indices', 'segment_info'
        top_k: Number of final results to return
        
    Returns:
        list: Combined and ranked retrieval results
    """
    if not segment_results:
        return []
    
    # Collect all unique documents with their scores
    doc_scores = {}  # {doc_id: {'score': float, 'votes': int, 'content': dict}}
    
    for segment_result in segment_results:
        distances = segment_result['distances']
        indices = segment_result['indices']
        segment_info = segment_result['segment_info']
        
        for i in range(len(distances)):
            doc_idx = indices[i]
            score = float(distances[i])
            
            if doc_idx not in doc_scores:
                doc_scores[doc_idx] = {
                    'score': 0.0,
                    'votes': 0,
                    'content': None,
                    'segments': []
                }
            
            # Add score and increment vote count
            doc_scores[doc_idx]['score'] += score
            doc_scores[doc_idx]['votes'] += 1
            doc_scores[doc_idx]['segments'].append(segment_info)
    
    # Normalize scores by number of votes and sort
    for doc_idx in doc_scores:
        doc_scores[doc_idx]['score'] = doc_scores[doc_idx]['score'] / doc_scores[doc_idx]['votes']
    
    # Sort by score (lower is better for FAISS distances) and then by votes (higher is better)
    sorted_docs = sorted(doc_scores.items(), 
                        key=lambda x: (x[1]['score'], -x[1]['votes']))
    
    # Return top_k results
    return sorted_docs[:top_k]


def extract_given_attributes(structured_query: dict) -> dict:
    """
    Extract only the given attributes (non-empty values) from structured query.
    This is used to create embeddings from only the given attributes, not the target attributes.
    """
    if not structured_query or 'parse_error' in structured_query:
        return {}
    
    # Filter out parse errors and metadata
    query_data = {k: v for k, v in structured_query.items() 
                  if k not in ['raw_question', 'parse_error', 'raw_response']}
    
    # Only keep attributes with non-empty values (given attributes)
    given_attributes = {k: v for k, v in query_data.items() if v and v != ""}
    
    return given_attributes


def get_segment_embeddings_from_structured_query(structured_query: dict, model: NaviForMaskedLM, device):
    """
    Extract individual segment embeddings from structured query for Navi models.
    
    Args:
        structured_query: Dictionary with key-value pairs from parsed question
        model: NaviForMaskedLM model
        device: Device to run on
        
    Returns:
        tuple: (segment_embeddings, segment_info)
            - segment_embeddings: numpy array of shape (num_segments, hidden)
            - segment_info: list of dicts with segment metadata
    """
    if not isinstance(model, NaviForMaskedLM):
        raise TypeError("This function is only for NaviForMaskedLM models.")
    
    # Filter out parse errors and metadata
    query_data = {k: v for k, v in structured_query.items() 
                  if k not in ['raw_question', 'parse_error', 'raw_response']}
    
    if not query_data:
        # Fallback to simple query format
        query_data = {"question": structured_query.get('raw_question', 'unknown')}
    
    # Create NaviDataset from structured query
    dataset = NaviDataset([query_data])
    data_item = dataset[0]
    
    allowed_keys = ['input_ids', 'attention_mask', 'position_ids', 'segment_ids', 'header_strings']
    inputs = {
        k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in data_item.items()
        if k in allowed_keys
    }
    
    if 'header_strings' in inputs and isinstance(inputs['header_strings'], list):
        if all(isinstance(k, str) for k in inputs['header_strings']):
            inputs['header_strings'] = [inputs['header_strings']]
    
    # Get contextualized embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        contextualized_embeddings = outputs[0]  # (1, seq_len, hidden)
    
    # Extract segment components
    header_positions = data_item.get('header_positions')
    value_positions = data_item.get('value_positions')
    header_strings = data_item.get('header_strings')
    
    if not all([header_positions, value_positions, header_strings]):
        raise ValueError("Missing header_positions, value_positions, or header_strings for segment embedding.")
    
    # Ensure header_strings is in the right format for the header encoder
    if isinstance(header_strings, list) and all(isinstance(k, str) for k in header_strings):
        header_strings_for_encoder = [header_strings]
    else:
        header_strings_for_encoder = header_strings

    with torch.no_grad():
        E_univ, _ = model.bert.embeddings.header_encoder(header_strings_for_encoder)
        H_ctx, V_ctx = extract_header_value_embeddings_from_positions(
            contextualized_embeddings,
            header_positions,
            value_positions
        )

        # Create segment embeddings
        segment_embeddings = model.create_segment_embeddings(E_univ, H_ctx, V_ctx) # (1, num_segments, hidden)
        
    # Convert to numpy and extract individual segments
    segment_embeddings_np = segment_embeddings.squeeze(0).cpu().numpy()  # (num_segments, hidden)
    
    # Create segment info
    segment_info = []
    for i, header_name in enumerate(header_strings):
        segment_info.append({
            'header': header_name,
            'value': query_data.get(header_name, ''),
            'segment_idx': i
        })
    
    return segment_embeddings_np, segment_info


def get_segment_embedding_from_structured_query(structured_query: dict, model: NaviForMaskedLM, device):
    """
    Extract segment embedding from structured query for Navi models (mean-pooled version).
    This is kept for backward compatibility.
    """
    segment_embeddings, _ = get_segment_embeddings_from_structured_query(structured_query, model, device)
    # Mean-pool segment embeddings
    mean_pooled = segment_embeddings.mean(axis=0, keepdims=True)  # (1, hidden)
    return mean_pooled


def main(args):
    # ------------------
    # Device
    # ------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}", file=sys.stderr)

    # ------------------
    # Encoder model
    # ------------------
    if args.model_type == 'bert':
        model_path = '/home/work/.default/woojun/navi/models/bert_quarter_product/epoch_2'
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained(model_path, local_files_only=True)
        model.to(device)
        model.eval()
        print(f"Loaded fine-tuned BERT from {model_path}", file=sys.stderr)

    elif args.model_type == 'ours':
        model_path = '/home/work/.default/woojun/navi/models/full_Quarter_Product_HVB_hv0p8_align0p5_vr0p5/epoch_2'
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = NaviForMaskedLM(model_path=model_path)
        model.to(device)
        model.eval()
        print(f"Loaded Navi model from {model_path}", file=sys.stderr)

    else:
        raise ValueError("model_type must be one of: bert, ours")

    # ------------------
    # FAISS + metadata
    # ------------------
    index = faiss.read_index(os.path.join(args.index_dir, 'index.faiss'))
    with open(os.path.join(args.index_dir, 'metadata.json'), 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # ------------------
    # Table data (full rows)
    # ------------------
    table_file = os.path.join(
        os.path.dirname(args.query_file),
        '..',
        'tables_cleaned',
        'WDC_product_for_cls_cleaned.jsonl'
    )
    with open(table_file, 'r', encoding='utf-8') as f:
        table_data = [json.loads(line) for line in f]

    # Map row_id (string) -> row content (Amazon-style fallback keeps enumerate index as id)
    table_content_map = {str(idx): row for idx, row in enumerate(table_data)}

    # ------------------
    # Queries
    # ------------------
    with open(args.query_file, 'r', encoding='utf-8') as f:
        queries = [json.loads(line) for line in f]

    # ------------------
    # Optional LLM for parsing
    # ------------------
    llm_client = None
    if args.enable_struct_parse:
        llm_client = LLMClient(provider=args.llm_provider, model=args.llm_model)
        print(f"Structured parsing enabled with {args.llm_provider}:{args.llm_model}", file=sys.stderr)

    # ------------------
    # Retrieval loop
    # ------------------
    with torch.no_grad():
        for query in queries:
            # questions can be a single string or a list (two variants, etc.)
            # we accept either 'question' or 'questions' in input
            if 'questions' in query:
                questions = query['questions']
            else:
                questions = query['question']
            if not isinstance(questions, list):
                questions = [questions]

            for question_idx, question_text in enumerate(questions):
                # 1) (Optional) parse NL question to structured JSON
                structured_query = None
                if llm_client is not None:
                    structured_query = parse_question_to_json(llm_client, question_text)

                # 2) Encode query to embedding
                if args.model_type == 'bert':
                    # For BERT: Use only given attributes if structured query is available, otherwise use full question
                    if structured_query is not None and 'parse_error' not in structured_query:
                        given_attributes = extract_given_attributes(structured_query)
                        if given_attributes:
                            # Create a text representation of given attributes
                            attr_text = " ".join([f"{k}: {v}" for k, v in given_attributes.items()])
                            print(f"Using given attributes for BERT: {given_attributes}", file=sys.stderr)
                        else:
                            attr_text = question_text
                    else:
                        attr_text = question_text
                    
                    inputs = tokenizer(attr_text, return_tensors='pt', truncation=True, max_length=512)
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    hidden_states = model.bert(**inputs).last_hidden_state
                    query_embedding = hidden_states[:, 0, :].detach().cpu().numpy()  # CLS
                    
                    # FAISS search for BERT
                    distances, indices = index.search(query_embedding, args.top_k)
                else:
                    # Navi model: choose between multi-segment, single segment, or CLS embedding
                    if args.use_multi_segment_retrieval and structured_query is not None and 'parse_error' not in structured_query:
                        try:
                            # Multi-segment retrieval: retrieve separately for each segment
                            given_attributes = extract_given_attributes(structured_query)
                            if given_attributes:
                                # Get individual segment embeddings
                                segment_embeddings, segment_info = get_segment_embeddings_from_structured_query(given_attributes, model, device)
                                print(f"Using multi-segment retrieval for {len(segment_embeddings)} segments: {[s['header'] for s in segment_info]}", file=sys.stderr)
                                
                                # Retrieve for each segment
                                segment_results = []
                                for i, (segment_embedding, info) in enumerate(zip(segment_embeddings, segment_info)):
                                    # Reshape for FAISS (needs 2D array)
                                    segment_embedding_2d = segment_embedding.reshape(1, -1)
                                    distances, indices = index.search(segment_embedding_2d, args.top_k)
                                    
                                    segment_results.append({
                                        'distances': distances[0],
                                        'indices': indices[0],
                                        'segment_info': info
                                    })
                                
                                # Vote and combine results
                                voted_results = vote_retrieval_results(segment_results, args.top_k)
                                
                                # Convert voted results to the expected format
                                distances = np.array([[r[1]['score'] for r in voted_results]])
                                indices = np.array([[r[0] for r in voted_results]])
                                
                                print(f"Multi-segment voting completed. Final results: {len(voted_results)} documents", file=sys.stderr)
                            else:
                                # Fallback to CLS embedding
                                qtext = f"Question: {question_text} [SEP]"
                                inputs = tokenizer(qtext, return_tensors='pt', truncation=True, max_length=512)
                                input_ids = inputs['input_ids'].to(device)
                                attention_mask = inputs['attention_mask'].to(device)
                                batch_size, seq_len = input_ids.shape
                                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                                segment_ids = torch.zeros_like(input_ids, device=device)
                                header_strings = [["question"]]

                                outputs = model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    header_strings=header_strings,
                                    segment_ids=segment_ids
                                )
                                query_embedding = outputs[0][:, 0, :].detach().cpu().numpy()  # CLS
                                distances, indices = index.search(query_embedding, args.top_k)
                        except Exception as e:
                            print(f"Failed to get multi-segment retrieval, falling back to CLS: {e}", file=sys.stderr)
                            # Fallback to CLS embedding
                            qtext = f"Question: {question_text} [SEP]"
                            inputs = tokenizer(qtext, return_tensors='pt', truncation=True, max_length=512)
                            input_ids = inputs['input_ids'].to(device)
                            attention_mask = inputs['attention_mask'].to(device)
                            batch_size, seq_len = input_ids.shape
                            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                            segment_ids = torch.zeros_like(input_ids, device=device)
                            header_strings = [["question"]]

                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                header_strings=header_strings,
                                segment_ids=segment_ids
                            )
                            query_embedding = outputs[0][:, 0, :].detach().cpu().numpy()  # CLS
                            distances, indices = index.search(query_embedding, args.top_k)
                    elif args.use_segment_embedding and structured_query is not None and 'parse_error' not in structured_query:
                        try:
                            # Single segment embedding (mean-pooled)
                            given_attributes = extract_given_attributes(structured_query)
                            if given_attributes:
                                # Create a new structured query with only given attributes
                                given_only_query = given_attributes
                                query_embedding = get_segment_embedding_from_structured_query(given_only_query, model, device)
                                print(f"Using single segment embedding for given attributes: {given_attributes}", file=sys.stderr)
                            else:
                                # Fallback to CLS embedding
                                qtext = f"Question: {question_text} [SEP]"
                                inputs = tokenizer(qtext, return_tensors='pt', truncation=True, max_length=512)
                                input_ids = inputs['input_ids'].to(device)
                                attention_mask = inputs['attention_mask'].to(device)
                                batch_size, seq_len = input_ids.shape
                                position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                                segment_ids = torch.zeros_like(input_ids, device=device)
                                header_strings = [["question"]]

                                outputs = model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    header_strings=header_strings,
                                    segment_ids=segment_ids
                                )
                                query_embedding = outputs[0][:, 0, :].detach().cpu().numpy()  # CLS
                            distances, indices = index.search(query_embedding, args.top_k)
                        except Exception as e:
                            print(f"Failed to get segment embedding, falling back to CLS: {e}", file=sys.stderr)
                            # Fallback to CLS embedding
                            qtext = f"Question: {question_text} [SEP]"
                            inputs = tokenizer(qtext, return_tensors='pt', truncation=True, max_length=512)
                            input_ids = inputs['input_ids'].to(device)
                            attention_mask = inputs['attention_mask'].to(device)
                            batch_size, seq_len = input_ids.shape
                            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                            segment_ids = torch.zeros_like(input_ids, device=device)
                            header_strings = [["question"]]

                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                header_strings=header_strings,
                                segment_ids=segment_ids
                            )
                            query_embedding = outputs[0][:, 0, :].detach().cpu().numpy()  # CLS
                            distances, indices = index.search(query_embedding, args.top_k)
                    else:
                        # Use CLS embedding (original approach)
                        qtext = f"Question: {question_text} [SEP]"
                        inputs = tokenizer(qtext, return_tensors='pt', truncation=True, max_length=512)
                        input_ids = inputs['input_ids'].to(device)
                        attention_mask = inputs['attention_mask'].to(device)
                        batch_size, seq_len = input_ids.shape
                        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                        segment_ids = torch.zeros_like(input_ids, device=device)
                        header_strings = [["question"]]

                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            header_strings=header_strings,
                            segment_ids=segment_ids
                        )
                        query_embedding = outputs[0][:, 0, :].detach().cpu().numpy()  # CLS
                        distances, indices = index.search(query_embedding, args.top_k)

                # 4) Gather retrieved docs (attach full row content when available)
                retrieved_docs = []
                for i in range(args.top_k):
                    idx_i = indices[0][i]
                    if idx_i < 0 or idx_i >= len(metadata):
                        continue
                    meta = metadata[idx_i]
                    row_id = meta.get('row_id')
                    row_id_str = str(row_id)
                    if row_id_str in table_content_map:
                        doc_content = table_content_map[row_id_str]
                        retrieved_docs.append({
                            'table_id': meta.get('table_id'),
                            'row_id': row_id,
                            'content': doc_content,
                            'score': float(distances[0][i])
                        })
                    else:
                        # Fallback when raw row isn't available in the map
                        tmp = dict(meta)
                        tmp['score'] = float(distances[0][i])
                        retrieved_docs.append(tmp)

                # 5) Emit record per question variant
                variant_query_id = f"{query.get('id', 'q')}_{question_idx}"
                out = {
                    'query_id': variant_query_id,
                    'original_query_id': query.get('id'),
                    'question_idx': question_idx,
                    'question_text': question_text,
                    'retrieved_docs': retrieved_docs
                }
                if structured_query is not None:
                    out['structured_query'] = structured_query

                print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_dir', type=str, required=True)
    parser.add_argument('--query_file', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True, choices=['bert', 'ours'])
    parser.add_argument('--top_k', type=int, default=10)

    # LLM parsing flags
    parser.add_argument('--enable_struct_parse', action='store_true',
                        help='Enable LLM-based question→JSON parsing and include it in output.')
    parser.add_argument('--llm_provider', type=str, default='openai',
                        help='Provider for LLMClient when struct parse is enabled (e.g., openai, together).')
    parser.add_argument('--llm_model', type=str, default='gpt-4o-mini',
                        help='Model name for LLMClient when struct parse is enabled.')
    
    # Segment embedding flags
    parser.add_argument('--use_segment_embedding', action='store_true',
                        help='Use segment embeddings for Navi queries instead of CLS embeddings.')
    parser.add_argument('--use_multi_segment_retrieval', action='store_true',
                        help='Retrieve separately for each segment and combine results through voting.')

    args = parser.parse_args()
    main(args)