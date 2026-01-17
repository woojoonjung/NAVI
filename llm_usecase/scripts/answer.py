import os
import argparse
import json
from tqdm import tqdm
from dataprep import BertDataset, NaviDataset
from llm_client import LLMClient
from dotenv import load_dotenv

load_dotenv()

def main(args):
    llm_client = LLMClient(provider=args.llm_provider, model=args.llm_model)

    with open(os.path.join(args.prompts_dir, 'default_system.txt'), 'r') as f:
        system_prompt = f.read()

    with open(os.path.join(args.prompts_dir, 'default_user.txt'), 'r') as f:
        user_prompt_template = f.read()

    if args.paradigm == 'zero_shot':
        with open(args.query_file, 'r') as f:
            queries = [json.loads(line) for line in f]
        
        output_file = os.path.join(args.output_dir, 'predictions.jsonl')
        with open(output_file, 'w') as f_out:
            for query in tqdm(queries):
                # Handle question field - process all questions if it's a list
                questions = query['question']
                if not isinstance(questions, list):
                    questions = [questions]  # Convert single question to list
                
                # Process each question variant
                for question_idx, question_text in enumerate(questions):
                    user_prompt = user_prompt_template.format(context="", question=question_text)
                    response = llm_client.chat(system_prompt, user_prompt)
                    
                    # Create unique query ID for each question variant
                    variant_query_id = f"{query['id']}_{question_idx}"
                    f_out.write(json.dumps({
                        'query_id': variant_query_id,
                        'original_query_id': query['id'],
                        'question_idx': question_idx,
                        'question_text': question_text,
                        'answer': response
                    }) + '\n')
    
    elif args.paradigm in ['rag_bert', 'rag_ours']:
        # Load table data for context retrieval
        table_file = os.path.join(args.data_dir, 'tables_cleaned', 'WDC_product_for_cls_cleaned.jsonl')
        with open(table_file, 'r') as f:
            table_data = [json.loads(line) for line in f]
        
        if args.paradigm == 'rag_bert':
            dataset = BertDataset(table_data)
        else:
            table_data_with_ids = [(0, row) for row in table_data]
            dataset = NaviDataset(table_data_with_ids)
        
        with open(args.retrieved_file, 'r') as f:
            retrieved_data = [json.loads(line) for line in f]
        
        with open(args.query_file, 'r') as f:
            queries = {q['id']: q for q in (json.loads(line) for line in f)}

        output_file = os.path.join(args.output_dir, 'predictions.jsonl')
        with open(output_file, 'w') as f_out:
            for item in tqdm(retrieved_data):
                variant_query_id = item['query_id']
                original_query_id = item.get('original_query_id', variant_query_id)
                question_text = item.get('question_text', '')
                
                if original_query_id not in queries:
                    continue
                query = queries[original_query_id]

                context = ""
                for doc in item['retrieved_docs']:
                    # Always use the content directly from retrieved docs
                    if 'content' in doc:
                        if args.paradigm == 'rag_bert':
                            # Create a simple serialization for BERT
                            content = doc['content']
                            serialized = " ".join([f"{k}: {v}" for k, v in content.items() if isinstance(v, str)])
                            context += serialized + " "
                        else:
                            # Create a simple serialization for Navi
                            content = doc['content']
                            serialized = " ".join([f"{k}: {v}" for k, v in content.items() if isinstance(v, str)])
                            context += serialized + " "
                    else:
                        # Fallback to dataset access
                        row_id = doc['row_id']
                        if args.paradigm == 'rag_bert':
                            row_data = table_data[row_id]
                            context += dataset._serialize_vanilla(row_data) + " "
                        else:
                            dataset_item = dataset[row_id]
                            context += " ".join([seg.serialize() for seg in dataset_item['raw_segments']]) + " "
                
                user_prompt = user_prompt_template.format(context=f"Context: {context.strip()}", question=question_text)
                response = llm_client.chat(system_prompt, user_prompt)
                
                # Include structured query if available
                output_data = {
                    'query_id': variant_query_id,
                    'original_query_id': original_query_id,
                    'question_text': question_text,
                    'answer': response
                }
                
                # Add structured query if it exists in the retrieved data
                if 'structured_query' in item:
                    output_data['structured_query'] = item['structured_query']
                
                f_out.write(json.dumps(output_data) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paradigm', type=str, required=True, choices=['zero_shot', 'rag_bert', 'rag_ours'])
    parser.add_argument('--query_file', type=str)
    parser.add_argument('--retrieved_file', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--llm_provider', type=str, default='openai')
    parser.add_argument('--llm_model', type=str, default='gpt-4.1-nano')
    parser.add_argument('--prompts_dir', type=str, default='prompts')
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    main(args)
