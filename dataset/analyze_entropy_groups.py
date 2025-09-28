#!/usr/bin/env python3
"""
Analyze entropy groups from segment embeddings data.

This script loads the segments.json file and provides utilities to analyze
low-entropy vs high-entropy segments separately.
"""

import numpy as np
import json
import os
import argparse
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def load_segments_data(input_path):
    """
    Load segment data from JSON file.
    
    Args:
        input_path: Path to segments.json file
        
    Returns:
        list: List of segment data dictionaries
    """
    print(f"Loading segment data from {input_path}...")
    
    with open(input_path, 'r') as f:
        segments_data = json.load(f)
    
    # Convert embeddings back to numpy arrays
    for seg in segments_data:
        seg['segment_embedding'] = np.array(seg['segment_embedding'])
    
    print(f"Loaded {len(segments_data)} segments")
    return segments_data


def separate_by_entropy_groups(segments_data):
    """
    Separate segments by entropy groups.
    
    Args:
        segments_data: List of segment data dictionaries
        
    Returns:
        dict: Mapping from entropy group to segments
    """
    entropy_groups = defaultdict(list)
    
    for seg in segments_data:
        entropy_groups[seg['entropy_group']].append(seg)
    
    print(f"Segments by entropy group:")
    for group, segments in entropy_groups.items():
        print(f"  {group}: {len(segments)} segments")
    
    return dict(entropy_groups)


def analyze_entropy_group(group_name, segments):
    """
    Analyze a specific entropy group.
    
    Args:
        group_name: Name of the entropy group
        segments: List of segments in this group
        
    Returns:
        dict: Analysis results
    """
    if not segments:
        return None
    
    print(f"\nAnalyzing {group_name} entropy group...")
    
    # Extract embeddings
    embeddings = np.array([seg['segment_embedding'] for seg in segments])
    
    # Basic statistics
    mean_embedding = np.mean(embeddings, axis=0)
    std_embedding = np.std(embeddings, axis=0)
    
    # Compute pairwise similarities
    similarities = cosine_similarity(embeddings)
    np.fill_diagonal(similarities, np.nan)  # Remove self-similarities
    
    # Get unique headers and genres
    headers = list(set(seg['header'] for seg in segments))
    genres = list(set(seg['primary_genre'] for seg in segments))
    table_ids = list(set(seg['table_id'] for seg in segments))
    
    # Entropy statistics
    entropy_values = [seg['entropy'] for seg in segments]
    
    analysis = {
        'num_segments': len(segments),
        'embedding_shape': embeddings.shape,
        'mean_embedding_norm': float(np.linalg.norm(mean_embedding)),
        'std_embedding_norm': float(np.linalg.norm(std_embedding)),
        'mean_pairwise_similarity': float(np.nanmean(similarities)),
        'std_pairwise_similarity': float(np.nanstd(similarities)),
        'entropy_stats': {
            'mean': float(np.mean(entropy_values)),
            'std': float(np.std(entropy_values)),
            'min': float(np.min(entropy_values)),
            'max': float(np.max(entropy_values))
        },
        'unique_headers': len(headers),
        'unique_genres': len(genres),
        'unique_tables': len(table_ids),
        'headers': headers,
        'genres': genres,
        'table_ids': table_ids
    }
    
    print(f"  Segments: {analysis['num_segments']}")
    print(f"  Mean pairwise similarity: {analysis['mean_pairwise_similarity']:.4f}")
    print(f"  Entropy range: {analysis['entropy_stats']['min']:.3f} - {analysis['entropy_stats']['max']:.3f}")
    print(f"  Unique headers: {analysis['unique_headers']}")
    print(f"  Unique genres: {analysis['unique_genres']}")
    
    return analysis


def compare_entropy_groups(entropy_groups):
    """
    Compare different entropy groups.
    
    Args:
        entropy_groups: Dict mapping group name to segments
        
    Returns:
        dict: Comparison results
    """
    print("\n" + "="*60)
    print("ENTROPY GROUPS COMPARISON")
    print("="*60)
    
    analyses = {}
    for group_name, segments in entropy_groups.items():
        analyses[group_name] = analyze_entropy_group(group_name, segments)
    
    # Compare low vs high entropy
    if 'low' in analyses and 'high' in analyses:
        low_analysis = analyses['low']
        high_analysis = analyses['high']
        
        print(f"\nLOW vs HIGH ENTROPY COMPARISON:")
        print(f"  Segments: {low_analysis['num_segments']} vs {high_analysis['num_segments']}")
        print(f"  Mean similarity: {low_analysis['mean_pairwise_similarity']:.4f} vs {high_analysis['mean_pairwise_similarity']:.4f}")
        print(f"  Embedding norm: {low_analysis['mean_embedding_norm']:.4f} vs {high_analysis['mean_embedding_norm']:.4f}")
        print(f"  Entropy range: {low_analysis['entropy_stats']['min']:.3f}-{low_analysis['entropy_stats']['max']:.3f} vs {high_analysis['entropy_stats']['min']:.3f}-{high_analysis['entropy_stats']['max']:.3f}")
        
        # Check if low entropy is more similar (higher similarity)
        if low_analysis['mean_pairwise_similarity'] > high_analysis['mean_pairwise_similarity']:
            print(f"  ✅ Low entropy segments are more similar (as expected)")
        else:
            print(f"  ⚠️  High entropy segments are more similar (unexpected)")
    
    return analyses


def create_entropy_comparison_plot(entropy_groups, output_dir):
    """
    Create comparison plots for entropy groups.
    
    Args:
        entropy_groups: Dict mapping group name to segments
        output_dir: Output directory for plots
    """
    print(f"\nCreating entropy comparison plots...")
    
    # Prepare data for plotting
    plot_data = []
    for group_name, segments in entropy_groups.items():
        if not segments:
            continue
            
        embeddings = np.array([seg['segment_embedding'] for seg in segments])
        
        # Compute pairwise similarities
        similarities = cosine_similarity(embeddings)
        np.fill_diagonal(similarities, np.nan)
        
        # Flatten similarities for histogram
        flat_similarities = similarities[~np.isnan(similarities)]
        
        for sim in flat_similarities:
            plot_data.append({
                'entropy_group': group_name,
                'similarity': sim
            })
    
    if not plot_data:
        print("No data available for plotting")
        return
    
    # Create similarity distribution plot
    plt.figure(figsize=(12, 8))
    
    for group_name in ['low', 'mid', 'high']:
        group_similarities = [d['similarity'] for d in plot_data if d['entropy_group'] == group_name]
        if group_similarities:
            plt.hist(group_similarities, bins=30, alpha=0.6, density=True, 
                    label=f'{group_name.title()} entropy (n={len(group_similarities)})')
    
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Pairwise Similarity Distribution by Entropy Group')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'entropy_groups_similarity_comparison.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Similarity comparison plot saved to: {plot_path}")


def save_analysis_results(analyses, output_path):
    """
    Save analysis results to JSON file.
    
    Args:
        analyses: Dict of analysis results
        output_path: Path to save results
    """
    print(f"\nSaving analysis results to {output_path}...")
    
    # Convert numpy arrays to lists for JSON serialization
    json_analyses = {}
    for group_name, analysis in analyses.items():
        if analysis is None:
            continue
            
        json_analysis = analysis.copy()
        # Remove numpy arrays that can't be serialized
        json_analyses[group_name] = json_analysis
    
    with open(output_path, 'w') as f:
        json.dump(json_analyses, f, indent=2)
    
    print(f"✅ Analysis results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze entropy groups from segment embeddings')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to segments.json file')
    parser.add_argument('--output_dir', type=str, default='./artifacts/entropy_analysis',
                       help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    print("=== Entropy Groups Analysis ===")
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    
    # Load data
    segments_data = load_segments_data(args.input)
    
    # Separate by entropy groups
    entropy_groups = separate_by_entropy_groups(segments_data)
    
    # Analyze each group
    analyses = compare_entropy_groups(entropy_groups)
    
    # Create comparison plots
    create_entropy_comparison_plot(entropy_groups, args.output_dir)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    save_analysis_results(analyses, os.path.join(args.output_dir, 'entropy_analysis.json'))
    
    print(f"\n✅ Analysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
