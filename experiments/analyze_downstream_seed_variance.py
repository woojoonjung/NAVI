#!/usr/bin/env python3
"""
Analyze seed variance in downstream task performance.

This script aggregates downstream task results (row classification, clustering, etc.)
across models trained with different seeds and generates variance statistics.
"""

import argparse
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from collections import defaultdict
from pathlib import Path


def find_model_checkpoints(base_dir, domain, seeds, epoch=2):
    """
    Find model checkpoints for different seeds.
    
    Args:
        base_dir: Base models directory (e.g., './models')
        domain: Domain name (e.g., 'movie', 'product')
        seeds: List of seeds to search for
        epoch: Epoch number (default: 2)
    
    Returns:
        dict: {seed: checkpoint_path} mapping
    """
    checkpoints = {}
    
    # Pattern: {base_dir}/navi_{domain}*/full_HVB_seed{seed}_*_epoch_{epoch}
    for seed in seeds:
        pattern = os.path.join(base_dir, f"navi_{domain}*", f"*seed{seed}*_epoch_{epoch}")
        matches = glob.glob(pattern)
        
        if matches:
            # Prefer the most specific match (longest path)
            matches.sort(key=len, reverse=True)
            checkpoints[seed] = matches[0]
            print(f"✓ Found seed {seed}: {matches[0]}")
        else:
            print(f"⚠️  No checkpoint found for seed {seed} (pattern: {pattern})")
    
    return checkpoints


def load_experiment_results(results_dir, pattern="*.json"):
    """
    Load experiment result JSON files.
    
    Args:
        results_dir: Directory containing result files
        pattern: File pattern to match
    
    Returns:
        dict: {filename: loaded_json}
    """
    results = {}
    result_files = glob.glob(os.path.join(results_dir, pattern))
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                results[os.path.basename(result_file)] = data
        except Exception as e:
            print(f"⚠️ Error loading {result_file}: {e}")
    
    return results


def extract_metrics_from_results(results_data, task_type="classification"):
    """
    Extract metrics from experiment results.
    
    Args:
        results_data: Loaded JSON results
        task_type: Type of task ('classification' or 'clustering')
    
    Returns:
        dict: Extracted metrics
    """
    metrics = {}
    
    if task_type == "classification":
        # Extract F1 scores for each model and ML model combination
        for model_name, model_results in results_data.items():
            if isinstance(model_results, dict):
                for ml_model_key, ml_results in model_results.items():
                    if isinstance(ml_results, dict) and 'mean' in ml_results:
                        metric_key = f"{model_name}_{ml_model_key}"
                        metrics[metric_key] = {
                            'mean': ml_results.get('mean'),
                            'std': ml_results.get('std'),
                            'scores': ml_results.get('scores', [])
                        }
    
    elif task_type == "clustering":
        # Extract clustering metrics (NMI, B-cubed F1, etc.)
        for model_name, model_results in results_data.items():
            if isinstance(model_results, dict):
                for metric_name, metric_value in model_results.items():
                    if isinstance(metric_value, (int, float)):
                        metric_key = f"{model_name}_{metric_name}"
                        metrics[metric_key] = metric_value
    
    return metrics


def aggregate_seed_results(all_results, seeds):
    """
    Aggregate results across seeds.
    
    Args:
        all_results: Dict of {seed: metrics_dict}
        seeds: List of seeds
    
    Returns:
        dict: Aggregated statistics
    """
    # Group metrics by metric name (across seeds)
    metric_groups = defaultdict(list)
    
    for seed in seeds:
        if seed not in all_results:
            continue
        for metric_name, metric_value in all_results[seed].items():
            if isinstance(metric_value, dict) and 'mean' in metric_value:
                # Classification result with mean/std
                metric_groups[metric_name].append(metric_value['mean'])
            elif isinstance(metric_value, (int, float)):
                # Direct metric value
                metric_groups[metric_name].append(metric_value)
    
    # Calculate statistics
    stats = {}
    for metric_name, values in metric_groups.items():
        if values:
            stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values,
                'num_seeds': len(values)
            }
    
    return stats


def plot_seed_variance(stats, output_path, task_type="classification"):
    """Generate plots showing seed variance in downstream task performance."""
    if not stats:
        print("⚠️ No statistics to plot")
        return
    
    # Filter to numeric metrics only
    numeric_stats = {k: v for k, v in stats.items() 
                    if isinstance(v.get('mean'), (int, float))}
    
    if not numeric_stats:
        print("⚠️ No numeric metrics to plot")
        return
    
    num_metrics = len(numeric_stats)
    cols = min(3, num_metrics)
    rows = (num_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if num_metrics == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    for idx, (metric_name, stat) in enumerate(numeric_stats.items()):
        ax = axes[idx]
        values = stat['values']
        mean = stat['mean']
        std = stat['std']
        
        # Box plot
        bp = ax.boxplot(values, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightcoral')
        bp['boxes'][0].set_alpha(0.7)
        
        # Add mean line
        ax.axhline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
        ax.axhline(mean + std, color='orange', linestyle=':', linewidth=1, alpha=0.7, label=f'±1 std')
        ax.axhline(mean - std, color='orange', linestyle=':', linewidth=1, alpha=0.7)
        
        ax.set_ylabel('Performance')
        ax.set_title(f'{metric_name}\n(Mean: {mean:.4f}, Std: {std:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Seed Variance in {task_type.capitalize()} Performance', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Seed variance plots saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze seed variance in downstream task performance")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing experiment result JSON files")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for variance analysis (default: same as results_dir)")
    parser.add_argument("--task_type", type=str, default="classification",
                       choices=["classification", "clustering"],
                       help="Type of downstream task")
    parser.add_argument("--seeds", type=int, nargs='+', default=[0, 1, 2],
                       help="List of seeds to analyze")
    parser.add_argument("--pattern", type=str, default="*seed*.json",
                       help="Pattern to match result files (should include 'seed' in name)")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.results_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all result files
    all_results = {}
    for seed in args.seeds:
        # Look for files containing this seed
        pattern = args.pattern.replace('seed*', f'seed{seed}')
        result_files = glob.glob(os.path.join(args.results_dir, pattern))
        
        if not result_files:
            # Try alternative pattern
            pattern = f"*seed{seed}*.json"
            result_files = glob.glob(os.path.join(args.results_dir, pattern))
        
        if result_files:
            # Load the first matching file for this seed
            result_file = result_files[0]
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)
                    metrics = extract_metrics_from_results(data, args.task_type)
                    all_results[seed] = metrics
                    print(f"✓ Loaded results for seed {seed} from {os.path.basename(result_file)}")
            except Exception as e:
                print(f"⚠️ Error loading {result_file}: {e}")
        else:
            print(f"⚠️ No result file found for seed {seed}")
    
    if not all_results:
        print("⚠️ No results found for any seed")
        return
    
    # Aggregate across seeds
    stats = aggregate_seed_results(all_results, args.seeds)
    
    # Generate plots
    plot_path = os.path.join(args.output_dir, f"downstream_seed_variance_{args.task_type}.png")
    plot_seed_variance(stats, plot_path, args.task_type)
    
    # Save summary JSON
    summary = {
        'task_type': args.task_type,
        'seeds': args.seeds,
        'num_runs': len(all_results),
        'statistics': stats,
        'all_results': all_results
    }
    
    summary_path = os.path.join(args.output_dir, f"downstream_seed_variance_{args.task_type}_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Seed variance summary saved to: {summary_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print(f"Downstream Task Seed Variance Summary ({args.task_type})")
    print("="*80)
    for metric_name, stat in stats.items():
        print(f"\n{metric_name}:")
        print(f"  Mean: {stat['mean']:.6f}")
        print(f"  Std:  {stat['std']:.6f}")
        print(f"  Min:  {stat['min']:.6f}")
        print(f"  Max:  {stat['max']:.6f}")
        print(f"  Values: {stat['values']}")
    print("="*80)


if __name__ == "__main__":
    main()

