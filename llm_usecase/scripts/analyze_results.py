import argparse
import json
import os
from typing import Dict, List
import pandas as pd

def load_evaluation_results(results_dir: str) -> Dict[str, Dict]:
    """Load evaluation results from all paradigms."""
    paradigms = ['zero_shot', 'rag_bert', 'rag_ours']
    results = {}
    
    for paradigm in paradigms:
        eval_file = os.path.join(results_dir, paradigm, 'evaluation.json')
        if os.path.exists(eval_file):
            with open(eval_file, 'r') as f:
                results[paradigm] = json.load(f)
        else:
            print(f"Warning: Evaluation file not found for {paradigm}: {eval_file}")
    
    return results

def create_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create a comparison table of all paradigms."""
    data = []
    
    for paradigm, result in results.items():
        if result is None:
            continue
            
        overall = result.get('overall', {})
        by_type = result.get('by_type', {})
        
        row = {
            'Paradigm': paradigm.replace('_', '-').title(),
            'Overall EM': overall.get('exact_match', 0),
            'Overall Contains': overall.get('contains_match', 0),
            'Coverage': overall.get('coverage', 0),
            'Total Queries': overall.get('total_queries', 0)
        }
        
        # Add query type breakdown
        for query_type in ['attribute_lookup', 'variant_phrasing', 'open_ended']:
            if query_type in by_type:
                type_result = by_type[query_type]
                row[f'{query_type.replace("_", " ").title()} EM'] = type_result.get('exact_match', 0)
                row[f'{query_type.replace("_", " ").title()} Contains'] = type_result.get('contains_match', 0)
            else:
                row[f'{query_type.replace("_", " ").title()} EM'] = 0
                row[f'{query_type.replace("_", " ").title()} Contains'] = 0
        
        data.append(row)
    
    return pd.DataFrame(data)

def analyze_retrieval_quality(results_dir: str) -> Dict[str, Dict]:
    """Analyze retrieval quality by examining retrieved documents."""
    paradigms = ['rag_bert', 'rag_ours']
    retrieval_stats = {}
    
    for paradigm in paradigms:
        retrieved_file = os.path.join(results_dir, paradigm, 'retrieved.jsonl')
        if not os.path.exists(retrieved_file):
            continue
            
        with open(retrieved_file, 'r') as f:
            retrieved_data = [json.loads(line) for line in f]
        
        # Load ground truth for comparison
        qa_file = os.path.join('data/qa/qas.jsonl')
        with open(qa_file, 'r') as f:
            qa_data = {item['id']: item for item in [json.loads(line) for line in f]}
        
        correct_retrievals = 0
        total_queries = 0
        
        for item in retrieved_data:
            query_id = item['query_id']
            if query_id not in qa_data:
                continue
                
            total_queries += 1
            ground_truth_row = qa_data[query_id]['support']['row_id']
            
            # Check if ground truth row is in top-k retrieved
            retrieved_rows = [doc['row_id'] for doc in item['retrieved_docs']]
            if ground_truth_row in retrieved_rows:
                correct_retrievals += 1
        
        retrieval_stats[paradigm] = {
            'retrieval_accuracy': correct_retrievals / total_queries if total_queries > 0 else 0,
            'total_queries': total_queries
        }
    
    return retrieval_stats

def generate_report(results_dir: str, output_file: str = None):
    """Generate a comprehensive analysis report."""
    
    print("=== LLM Case Study Analysis Report ===")
    print()
    
    # Load evaluation results
    results = load_evaluation_results(results_dir)
    
    if not results:
        print("No evaluation results found!")
        return
    
    # Create comparison table
    print("=== Performance Comparison ===")
    df = create_comparison_table(results)
    print(df.to_string(index=False, float_format='%.3f'))
    print()
    
    # Analyze retrieval quality
    print("=== Retrieval Quality Analysis ===")
    retrieval_stats = analyze_retrieval_quality(results_dir)
    for paradigm, stats in retrieval_stats.items():
        print(f"{paradigm.replace('_', '-').title()}:")
        print(f"  Retrieval Accuracy: {stats['retrieval_accuracy']:.3f}")
        print(f"  Total Queries: {stats['total_queries']}")
    print()
    
    # Key insights
    print("=== Key Insights ===")
    if len(results) >= 2:
        # Compare zero-shot vs RAG
        if 'zero_shot' in results and 'rag_bert' in results:
            zs_em = results['zero_shot']['overall']['exact_match']
            rag_em = results['rag_bert']['overall']['exact_match']
            
            if rag_em > zs_em:
                improvement = ((rag_em - zs_em) / zs_em) * 100 if zs_em > 0 else 0
                print(f"RAG-BERT improves over Zero-shot by {improvement:.1f}% in exact match")
            else:
                decline = ((zs_em - rag_em) / zs_em) * 100 if zs_em > 0 else 0
                print(f"RAG-BERT performs {decline:.1f}% worse than Zero-shot in exact match")
        
        # Compare BERT vs Ours
        if 'rag_bert' in results and 'rag_ours' in results:
            bert_em = results['rag_bert']['overall']['exact_match']
            ours_em = results['rag_ours']['overall']['exact_match']
            
            if ours_em > bert_em:
                improvement = ((ours_em - bert_em) / bert_em) * 100 if bert_em > 0 else 0
                print(f"RAG-Ours improves over RAG-BERT by {improvement:.1f}% in exact match")
            else:
                decline = ((bert_em - ours_em) / bert_em) * 100 if bert_em > 0 else 0
                print(f"RAG-Ours performs {decline:.1f}% worse than RAG-BERT in exact match")
    
    # Query type analysis
    print("\n=== Query Type Analysis ===")
    for paradigm, result in results.items():
        if result and 'by_type' in result:
            print(f"\n{paradigm.replace('_', '-').title()}:")
            for query_type, type_result in result['by_type'].items():
                em = type_result.get('exact_match', 0)
                contains = type_result.get('contains_match', 0)
                count = type_result.get('total_queries', 0)
                print(f"  {query_type.replace('_', ' ').title()}: EM={em:.3f}, Contains={contains:.3f} ({count} queries)")
    
    # Save report
    if output_file:
        report_data = {
            'comparison_table': df.to_dict('records'),
            'retrieval_stats': retrieval_stats,
            'detailed_results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nDetailed report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze LLM case study results')
    parser.add_argument('--results_dir', type=str, default='llm_usecase/runs',
                       help='Directory containing experiment results')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save detailed report')
    
    args = parser.parse_args()
    
    generate_report(args.results_dir, args.output_file)

if __name__ == '__main__':
    main()
