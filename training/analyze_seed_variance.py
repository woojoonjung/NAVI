#!/usr/bin/env python3
"""
Analyze seed variance across multiple training runs.

This script aggregates metrics from multiple training runs with different seeds
and generates variance statistics and plots for ICLR rebuttal.
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


def load_metrics_file(metrics_path):
    """Load metrics JSON file."""
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Error loading {metrics_path}: {e}")
        return None


def extract_metrics(metrics_data, metric_paths):
    """
    Extract metrics from nested structure.
    
    Args:
        metrics_data: Loaded JSON metrics
        metric_paths: List of tuples (path, name) where path is list of keys
                      Special value -1 means last element of list
    """
    results = {}
    for path, name in metric_paths:
        value = metrics_data
        try:
            for key in path:
                if key == -1 and isinstance(value, list) and len(value) > 0:
                    value = value[-1]
                elif isinstance(value, (list, dict)):
                    value = value[key]
                else:
                    raise KeyError(f"Cannot access {key} in {type(value)}")
            results[name] = value
        except (KeyError, TypeError, IndexError):
            results[name] = None
    return results


def aggregate_seed_metrics(metrics_files, metric_paths):
    """
    Aggregate metrics across multiple seeds.
    
    Args:
        metrics_files: List of (seed, metrics_file_path) tuples
        metric_paths: List of (path, name) tuples for metrics to extract
    """
    all_metrics = {}
    
    for seed, metrics_path in metrics_files:
        metrics_data = load_metrics_file(metrics_path)
        if metrics_data is None:
            continue
        
        extracted = extract_metrics(metrics_data, metric_paths)
        all_metrics[seed] = extracted
    
    # Calculate statistics
    stats = {}
    for metric_name in metric_paths[0][1] if metric_paths else []:
        values = [m.get(metric_name) for m in all_metrics.values() 
                 if m.get(metric_name) is not None and isinstance(m.get(metric_name), (int, float))]
        
        if values:
            stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values,
                'num_seeds': len(values)
            }
    
    return all_metrics, stats


def plot_seed_variance(stats, output_path):
    """Generate plots showing seed variance."""
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
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][0].set_alpha(0.7)
        
        # Add mean line
        ax.axhline(mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean:.4f}')
        ax.axhline(mean + std, color='orange', linestyle=':', linewidth=1, alpha=0.7, label=f'±1 std')
        ax.axhline(mean - std, color='orange', linestyle=':', linewidth=1, alpha=0.7)
        
        ax.set_ylabel('Value')
        ax.set_title(f'{metric_name}\n(Mean: {mean:.4f}, Std: {std:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Seed variance plots saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze seed variance across training runs")
    parser.add_argument("--metrics_dir", type=str, required=True,
                       help="Directory containing training_metrics_*.json files")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for variance analysis (default: same as metrics_dir)")
    parser.add_argument("--pattern", type=str, default="training_metrics_*.json",
                       help="Pattern to match metrics files")
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.metrics_dir
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find all metrics files
    metrics_files = glob.glob(os.path.join(args.metrics_dir, args.pattern))
    
    if not metrics_files:
        print(f"⚠️ No metrics files found matching pattern: {args.pattern}")
        return
    
    print(f"📊 Found {len(metrics_files)} metrics files")
    
    # Extract seeds from filenames
    seed_metrics = []
    for metrics_file in metrics_files:
        filename = os.path.basename(metrics_file)
        # Try to extract seed from filename (e.g., "training_metrics_full_HVB_seed42_cleaned.json")
        seed = None
        if 'seed' in filename:
            try:
                seed_part = filename.split('seed')[1].split('_')[0]
                seed = int(seed_part)
            except (ValueError, IndexError):
                pass
        
        if seed is None:
            print(f"⚠️ Could not extract seed from {filename}, skipping")
            continue
        
        seed_metrics.append((seed, metrics_file))
    
    if not seed_metrics:
        print("⚠️ No valid seed metrics found")
        return
    
    seed_metrics.sort(key=lambda x: x[0])
    print(f"📊 Processing {len(seed_metrics)} runs with seeds: {[s for s, _ in seed_metrics]}")
    
    # Define metrics to extract
    metric_paths = [
        (['training_info', 'validation', -1, 'metrics', 'total_loss'], 'final_val_total_loss'),
        (['training_info', 'validation', -1, 'metrics', 'mlm_loss'], 'final_val_mlm_loss'),
        (['training_info', 'validation', -1, 'metrics', 'entropy_contrast_loss'], 'final_val_entropy_loss'),
        (['training_info', 'epochs', -1, 'metrics', 'total_loss'], 'final_train_total_loss'),
        (['training_info', 'epochs', -1, 'metrics', 'mlm_loss'], 'final_train_mlm_loss'),
        (['training_info', 'compute_metrics', 'total_training_time_hours'], 'total_training_hours'),
        (['training_info', 'compute_metrics', 'total_training_steps'], 'total_training_steps'),
    ]
    
    # Aggregate metrics
    all_metrics, stats = aggregate_seed_metrics(seed_metrics, metric_paths)
    
    # Generate plots
    plot_path = os.path.join(args.output_dir, "seed_variance_analysis.png")
    plot_seed_variance(stats, plot_path)
    
    # Save summary JSON
    summary = {
        'seeds': [s for s, _ in seed_metrics],
        'num_runs': len(seed_metrics),
        'statistics': stats,
        'all_metrics': all_metrics
    }
    
    summary_path = os.path.join(args.output_dir, "seed_variance_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Seed variance summary saved to: {summary_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("Seed Variance Summary")
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

