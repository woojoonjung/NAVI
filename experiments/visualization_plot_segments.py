#!/usr/bin/env python3
"""
Visualize segment embeddings with entropy-aware coloring and optional table convex hulls.

This script creates UMAP and t-SNE visualizations showing:
- Low-entropy segments (blue) forming a central core
- High-entropy segments (red) populating the periphery
- Optional convex hulls around segments from the same table
"""

import torch
import numpy as np
import json
import os
import argparse
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import ConvexHull
import umap

# Set matplotlib parameters
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['figure.dpi'] = 300

# Entropy group colors
ENTROPY_COLORS = {
    'low': '#1f77b4',    # Blue
    'mid': '#CCCCCC',    # Light gray
    'high': '#d62728'    # Red
}

# Entropy group alphas
ENTROPY_ALPHAS = {
    'low': 0.7,
    'mid': 0.3,
    'high': 0.7
}

# Table hull colors (gray only)
TABLE_HULL_COLORS = [
    '#808080'  # Gray
]

def load_segments_data(input_path):
    """
    Load segment data from JSON file.
    
    Args:
        input_path: Path to segments.json file
        
    Returns:
        tuple: (segments_data, metadata)
    """
    print(f"Loading segment data from {input_path}...")
    
    with open(input_path, 'r') as f:
        segments_data = json.load(f)
    
    # Convert embeddings back to numpy arrays
    for seg in segments_data:
        seg['segment_embedding'] = np.array(seg['segment_embedding'])
    
    # Load metadata
    metadata_path = os.path.join(os.path.dirname(input_path), 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    print(f"Loaded {len(segments_data)} segments")
    return segments_data, metadata


def preprocess_embeddings(embeddings, method='l2_normalize'):
    """
    Preprocess embeddings for dimensionality reduction.
    
    Args:
        embeddings: Array of embeddings (N, D)
        method: Preprocessing method ('l2_normalize', 'standardize_l2', 'pca_l2')
        
    Returns:
        np.ndarray: Preprocessed embeddings
    """
    if method == 'l2_normalize':
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return embeddings / norms
    
    elif method == 'standardize_l2':
        # Standardize then L2 normalize
        scaler = StandardScaler()
        standardized = scaler.fit_transform(embeddings)
        norms = np.linalg.norm(standardized, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return standardized / norms
    
    elif method == 'pca_l2':
        # PCA to 50 components then L2 normalize
        pca = PCA(n_components=min(50, embeddings.shape[1]))
        pca_embeddings = pca.fit_transform(embeddings)
        norms = np.linalg.norm(pca_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return pca_embeddings / norms
    
    else:
        raise ValueError(f"Unknown preprocessing method: {method}")


def compute_umap_projection(embeddings, n_neighbors=30, min_dist=0.05, metric='cosine', random_state=42):
    """
    Compute UMAP projection.
    
    Args:
        embeddings: Preprocessed embeddings (N, D)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: UMAP metric
        random_state: Random seed
        
    Returns:
        np.ndarray: 2D UMAP coordinates (N, 2)
    """
    print(f"Computing UMAP projection (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_components=2
    )
    
    return reducer.fit_transform(embeddings)


def compute_tsne_projection(embeddings, perplexity=30, metric='cosine', n_iter=1000, random_state=42):
    """
    Compute t-SNE projection.
    
    Args:
        embeddings: Preprocessed embeddings (N, D)
        perplexity: t-SNE perplexity parameter
        metric: t-SNE metric
        n_iter: Number of iterations
        random_state: Random seed
        
    Returns:
        np.ndarray: 2D t-SNE coordinates (N, 2)
    """
    print(f"Computing t-SNE projection (perplexity={perplexity})...")
    
    # Adjust perplexity if needed
    n_samples = len(embeddings)
    if perplexity >= n_samples:
        perplexity = max(1, (n_samples - 1) // 3)
        print(f"Adjusted perplexity to {perplexity} due to sample size")
    
    reducer = TSNE(
        perplexity=perplexity,
        metric=metric,
        n_iter=n_iter,
        random_state=random_state,
        init='pca',
        n_components=2
    )
    
    return reducer.fit_transform(embeddings)


def get_primary_genre(genres_str):
    """
    Extract primary genre from genres string.
    
    Args:
        genres_str: Comma-separated genres string
        
    Returns:
        str: Primary genre (lowercase)
    """
    if not genres_str or not isinstance(genres_str, str):
        return 'other'
    
    primary = genres_str.split(',')[0].strip().lower()
    return primary


def draw_table_convex_hulls(ax, coords, table_ids, entropy_groups, include_mid=True):
    """
    Draw convex hulls around segments from the same table.
    
    Args:
        ax: Matplotlib axis
        coords: 2D coordinates (N, 2)
        table_ids: List of table IDs
        entropy_groups: List of entropy group labels
        include_mid: Whether to include mid-entropy points
    """
    print("Drawing convex hulls for each table...")
    
    # Filter data based on include_mid
    if not include_mid:
        mask = np.array(entropy_groups) != 'mid'
        coords = coords[mask]
        table_ids = np.array(table_ids)[mask]
        entropy_groups = np.array(entropy_groups)[mask]
    
    # Group coordinates by table
    table_coords = {}
    for i, table_id in enumerate(table_ids):
        if table_id not in table_coords:
            table_coords[table_id] = []
        table_coords[table_id].append(coords[i])
    
    # Draw convex hull for each table
    for table_idx, (table_id, coords_list) in enumerate(table_coords.items()):
        if len(coords_list) < 3:  # Need at least 3 points for convex hull
            continue
            
        coords_array = np.array(coords_list)
        
        try:
            # Compute convex hull
            hull = ConvexHull(coords_array)
            
            # Get hull vertices
            hull_points = coords_array[hull.vertices]
            
            # Create polygon
            hull_polygon = plt.Polygon(hull_points, 
                                     alpha=0.15,
                                     facecolor=TABLE_HULL_COLORS[table_idx % len(TABLE_HULL_COLORS)],
                                     edgecolor=TABLE_HULL_COLORS[table_idx % len(TABLE_HULL_COLORS)],
                                     linewidth=1)
            
            ax.add_patch(hull_polygon)
            
        except Exception as e:
            print(f"Warning: Could not create convex hull for table {table_id}: {e}")
            continue


def create_entropy_plot(coords, entropy_groups, table_ids, method, title_suffix="", 
                       include_mid=True, figsize=(12, 10), show_table_hulls=True):
    """
    Create entropy visualization plot with optional table convex hulls.
    
    Args:
        coords: 2D coordinates (N, 2)
        entropy_groups: List of entropy group labels
        table_ids: List of table IDs
        method: Projection method ('umap' or 'tsne')
        title_suffix: Additional title text
        include_mid: Whether to include mid-entropy points
        figsize: Figure size
        show_table_hulls: Whether to show convex hulls around tables
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter data based on include_mid
    if not include_mid:
        mask = np.array(entropy_groups) != 'mid'
        coords = coords[mask]
        entropy_groups = np.array(entropy_groups)[mask]
        table_ids = np.array(table_ids)[mask]
    
    # Get unique entropy groups
    unique_entropy_groups = sorted(set(entropy_groups))
    
    # Draw convex hulls first (so they appear behind points)
    if show_table_hulls:
        draw_table_convex_hulls(ax, coords, table_ids, entropy_groups, include_mid)
    
    # Create plots for each entropy group (all as circles)
    for entropy_group in unique_entropy_groups:
        if entropy_group == 'mid' and not include_mid:
            continue
            
        mask = np.array(entropy_groups) == entropy_group
        group_coords = coords[mask]
        
        color = ENTROPY_COLORS.get(entropy_group, '#000000')
        alpha = ENTROPY_ALPHAS.get(entropy_group, 0.5)
        
        label = f"{entropy_group.title()} Entropy"
        
        ax.scatter(group_coords[:, 0], group_coords[:, 1], 
                  c=color, marker='o', alpha=alpha, s=50, 
                  label=label, edgecolors='none')
    
    # Customize plot
    ax.set_title(f'{method.upper()} — Segment Space{title_suffix}', fontsize=16, fontweight='bold')
    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=14)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=14)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create legend
    create_entropy_legend(ax, unique_entropy_groups, include_mid)
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def create_entropy_legend(ax, entropy_groups, include_mid=True):
    """
    Create legend for entropy groups.
    
    Args:
        ax: Matplotlib axis
        entropy_groups: List of entropy groups
        include_mid: Whether mid-entropy is included
    """
    # Entropy legend (colors)
    entropy_patches = []
    for group in entropy_groups:
        if group == 'mid' and not include_mid:
            continue
        color = ENTROPY_COLORS.get(group, '#000000')
        alpha = ENTROPY_ALPHAS.get(group, 0.5)
        patch = mpatches.Patch(color=color, alpha=alpha, label=f'{group.title()} Entropy')
        entropy_patches.append(patch)
    
    # Add legend
    if entropy_patches:
        ax.legend(handles=entropy_patches, loc='upper right')


def compute_core_periphery_analysis(coords, entropy_groups, method='umap'):
    """
    Compute core-periphery analysis for UMAP coordinates.
    
    Args:
        coords: 2D coordinates (N, 2)
        entropy_groups: List of entropy group labels
        method: Projection method
        
    Returns:
        dict: Analysis results
    """
    print("Computing core-periphery analysis...")
    
    # Filter to low and high entropy only
    mask = np.isin(entropy_groups, ['low', 'high'])
    filtered_coords = coords[mask]
    filtered_groups = np.array(entropy_groups)[mask]
    
    # Compute centroid of low-entropy points
    low_mask = filtered_groups == 'low'
    if not np.any(low_mask):
        print("Warning: No low-entropy points found for core-periphery analysis")
        return {}
    
    low_coords = filtered_coords[low_mask]
    centroid = np.mean(low_coords, axis=0)
    
    # Compute radial distances
    low_distances = np.linalg.norm(low_coords - centroid, axis=1)
    high_distances = np.linalg.norm(filtered_coords[~low_mask] - centroid, axis=1)
    
    # Compute statistics
    results = {
        'centroid': centroid.tolist(),
        'low_entropy': {
            'count': len(low_distances),
            'mean_radius': float(np.mean(low_distances)),
            'std_radius': float(np.std(low_distances)),
            'min_radius': float(np.min(low_distances)),
            'max_radius': float(np.max(low_distances))
        },
        'high_entropy': {
            'count': len(high_distances),
            'mean_radius': float(np.mean(high_distances)),
            'std_radius': float(np.std(high_distances)),
            'min_radius': float(np.min(high_distances)),
            'max_radius': float(np.max(high_distances))
        }
    }
    
    print(f"Core-periphery analysis ({method}):")
    print(f"  Low entropy: {results['low_entropy']['count']} points, "
          f"radius = {results['low_entropy']['mean_radius']:.3f} ± {results['low_entropy']['std_radius']:.3f}")
    print(f"  High entropy: {results['high_entropy']['count']} points, "
          f"radius = {results['high_entropy']['mean_radius']:.3f} ± {results['high_entropy']['std_radius']:.3f}")
    
    return results


def compute_core_periphery_analysis_original(embeddings, entropy_groups, method='original'):
    """
    Compute core-periphery analysis using original embeddings.
    
    Args:
        embeddings: Original embeddings (N, D)
        entropy_groups: List of entropy group labels
        method: Analysis method name
        
    Returns:
        dict: Analysis results
    """
    print("Computing core-periphery analysis on original embeddings...")
    
    # Filter to low and high entropy only
    mask = np.isin(entropy_groups, ['low', 'high'])
    filtered_embeddings = embeddings[mask]
    filtered_groups = np.array(entropy_groups)[mask]
    
    # Compute centroid of low-entropy points
    low_mask = filtered_groups == 'low'
    if not np.any(low_mask):
        print("Warning: No low-entropy points found for core-periphery analysis")
        return {}
    
    low_embeddings = filtered_embeddings[low_mask]
    centroid = np.mean(low_embeddings, axis=0)
    
    # Compute radial distances in original embedding space
    low_distances = np.linalg.norm(low_embeddings - centroid, axis=1)
    high_distances = np.linalg.norm(filtered_embeddings[~low_mask] - centroid, axis=1)
    
    # Compute statistics
    results = {
        'centroid': centroid.tolist(),
        'low_entropy': {
            'count': len(low_distances),
            'mean_radius': float(np.mean(low_distances)),
            'std_radius': float(np.std(low_distances)),
            'min_radius': float(np.min(low_distances)),
            'max_radius': float(np.max(low_distances))
        },
        'high_entropy': {
            'count': len(high_distances),
            'mean_radius': float(np.mean(high_distances)),
            'std_radius': float(np.std(high_distances)),
            'min_radius': float(np.min(high_distances)),
            'max_radius': float(np.max(high_distances))
        }
    }
    
    print(f"Core-periphery analysis ({method}):")
    print(f"  Low entropy: {results['low_entropy']['count']} points, "
          f"radius = {results['low_entropy']['mean_radius']:.3f} ± {results['low_entropy']['std_radius']:.3f}")
    print(f"  High entropy: {results['high_entropy']['count']} points, "
          f"radius = {results['high_entropy']['mean_radius']:.3f} ± {results['high_entropy']['std_radius']:.3f}")
    
    return results


def create_density_plot(coords, entropy_groups, method, output_dir):
    """
    Create density plot showing core-periphery distribution.
    
    Args:
        coords: 2D coordinates (N, 2)
        entropy_groups: List of entropy group labels
        method: Projection method
        output_dir: Output directory
    """
    print("Creating density plot...")
    
    # Filter to low and high entropy only
    mask = np.isin(entropy_groups, ['low', 'high'])
    filtered_coords = coords[mask]
    filtered_groups = np.array(entropy_groups)[mask]
    
    # Compute centroid and radial distances
    low_mask = filtered_groups == 'low'
    if not np.any(low_mask):
        print("Warning: No low-entropy points found for density plot")
        return
    
    low_coords = filtered_coords[low_mask]
    centroid = np.mean(low_coords, axis=0)
    
    low_distances = np.linalg.norm(low_coords - centroid, axis=1)
    high_distances = np.linalg.norm(filtered_coords[~low_mask] - centroid, axis=1)
    
    # Create density plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histograms
    ax.hist(low_distances, bins=30, alpha=0.7, density=True, 
           label=f'Low Entropy (n={len(low_distances)})', color=ENTROPY_COLORS['low'])
    ax.hist(high_distances, bins=30, alpha=0.7, density=True, 
           label=f'High Entropy (n={len(high_distances)})', color=ENTROPY_COLORS['high'])
    
    ax.set_xlabel('Radial Distance from Low-Entropy Centroid')
    ax.set_ylabel('Density')
    ax.set_title(f'Core-Periphery Distribution ({method.upper()})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'density_core_periphery_{method}.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    fig.savefig(plot_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.close()
    
    print(f"✅ Density plot saved to: {plot_path}")


def create_density_plot_original(embeddings, entropy_groups, method, output_dir):
    """
    Create density plot showing core-periphery distribution using original embeddings.
    
    Args:
        embeddings: Original embeddings (N, D)
        entropy_groups: List of entropy group labels
        method: Analysis method name
        output_dir: Output directory
    """
    print("Creating density plot from original embeddings...")
    
    # Filter to low and high entropy only
    mask = np.isin(entropy_groups, ['low', 'high'])
    filtered_embeddings = embeddings[mask]
    filtered_groups = np.array(entropy_groups)[mask]
    
    # Compute centroid and radial distances
    low_mask = filtered_groups == 'low'
    if not np.any(low_mask):
        print("Warning: No low-entropy points found for density plot")
        return
    
    low_embeddings = filtered_embeddings[low_mask]
    centroid = np.mean(low_embeddings, axis=0)
    
    low_distances = np.linalg.norm(low_embeddings - centroid, axis=1)
    high_distances = np.linalg.norm(filtered_embeddings[~low_mask] - centroid, axis=1)
    
    # Create density plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histograms
    ax.hist(low_distances, bins=30, alpha=0.7, density=True, 
           label=f'Low Entropy (n={len(low_distances)})', color=ENTROPY_COLORS['low'])
    ax.hist(high_distances, bins=30, alpha=0.7, density=True, 
           label=f'High Entropy (n={len(high_distances)})', color=ENTROPY_COLORS['high'])
    
    ax.set_xlabel('Radial Distance from Low-Entropy Centroid (Original Embedding Space)')
    ax.set_ylabel('Density')
    ax.set_title(f'Core-Periphery Distribution ({method.upper()}) - Original Embeddings')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'density_core_periphery_{method}_original.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    fig.savefig(plot_path.replace('.png', '.svg'), bbox_inches='tight')
    plt.close()
    
    print(f"✅ Density plot saved to: {plot_path}")


def create_combined_tsne_density_plot(tsne_coords, embeddings, entropy_groups, table_ids, 
                                    title_suffix="", include_mid=True, show_table_hulls=True, 
                                    output_dir=None):
    """
    Create a combined plot showing t-SNE visualization (top) and density plot from original embeddings (bottom).
    
    Args:
        tsne_coords: 2D t-SNE coordinates (N, 2)
        embeddings: Original embeddings (N, D)
        entropy_groups: List of entropy group labels
        table_ids: List of table IDs
        title_suffix: Additional title text
        include_mid: Whether to include mid-entropy points
        show_table_hulls: Whether to show convex hulls around tables
        output_dir: Output directory for saving the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    print("Creating combined t-SNE and density plot...")
    
    # Create figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    
    # === TOP PLOT: t-SNE Visualization ===
    # Filter data based on include_mid
    if not include_mid:
        mask = np.array(entropy_groups) != 'mid'
        filtered_tsne_coords = tsne_coords[mask]
        filtered_entropy_groups = np.array(entropy_groups)[mask]
        filtered_table_ids = np.array(table_ids)[mask]
    else:
        filtered_tsne_coords = tsne_coords
        filtered_entropy_groups = np.array(entropy_groups)
        filtered_table_ids = np.array(table_ids)
    
    # Get unique entropy groups
    unique_entropy_groups = sorted(set(filtered_entropy_groups))
    
    # Draw convex hulls first (so they appear behind points)
    if show_table_hulls:
        draw_table_convex_hulls(ax1, filtered_tsne_coords, filtered_table_ids, 
                               filtered_entropy_groups, include_mid)
    
    # Create plots for each entropy group
    for entropy_group in unique_entropy_groups:
        if entropy_group == 'mid' and not include_mid:
            continue
            
        mask = np.array(filtered_entropy_groups) == entropy_group
        group_coords = filtered_tsne_coords[mask]
        
        color = ENTROPY_COLORS.get(entropy_group, '#000000')
        alpha = ENTROPY_ALPHAS.get(entropy_group, 0.5)
        
        label = f"{entropy_group.title()} Entropy"
        
        ax1.scatter(group_coords[:, 0], group_coords[:, 1], 
                   c=color, marker='o', alpha=alpha, s=50, 
                   label=label, edgecolors='none')
    
    # Customize t-SNE plot
    ax1.set_title(f't-SNE — Segment Space{title_suffix}', fontsize=16, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=14)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=14)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Create legend for t-SNE plot
    create_entropy_legend(ax1, unique_entropy_groups, include_mid)
    
    # === BOTTOM PLOT: Density from Original Embeddings ===
    # Filter to low and high entropy only
    mask = np.isin(entropy_groups, ['low', 'high'])
    filtered_embeddings = embeddings[mask]
    filtered_groups = np.array(entropy_groups)[mask]
    
    # Compute centroid and radial distances
    low_mask = filtered_groups == 'low'
    if not np.any(low_mask):
        print("Warning: No low-entropy points found for density plot")
        ax2.text(0.5, 0.5, 'No low-entropy points found', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        ax2.set_title('Core-Periphery Distribution (Original Embeddings)', fontsize=16, fontweight='bold')
    else:
        low_embeddings = filtered_embeddings[low_mask]
        centroid = np.mean(low_embeddings, axis=0)
        
        low_distances = np.linalg.norm(low_embeddings - centroid, axis=1)
        high_distances = np.linalg.norm(filtered_embeddings[~low_mask] - centroid, axis=1)
        
        # Plot histograms
        ax2.hist(low_distances, bins=30, alpha=0.7, density=True, 
                label=f'Low Entropy (n={len(low_distances)})', color=ENTROPY_COLORS['low'])
        ax2.hist(high_distances, bins=30, alpha=0.7, density=True, 
                label=f'High Entropy (n={len(high_distances)})', color=ENTROPY_COLORS['high'])
        
        ax2.set_xlabel('Radial Distance from Low-Entropy Centroid (Original Embedding Space)', fontsize=14)
        ax2.set_ylabel('Density', fontsize=14)
        ax2.set_title('Core-Periphery Distribution (Original Embeddings)', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=18)
        ax2.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if output directory is provided
    if output_dir:
        plot_path = os.path.join(output_dir, 'combined_tsne_density_original.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        fig.savefig(plot_path.replace('.png', '.svg'), bbox_inches='tight')
        print(f"✅ Combined plot saved to: {plot_path}")
    
    return fig


def save_coordinates_csv(coords, entropy_groups, genres, table_ids, row_ids, headers, 
                        method, output_dir):
    """
    Save 2D coordinates and metadata to CSV.
    
    Args:
        coords: 2D coordinates (N, 2)
        entropy_groups: List of entropy group labels
        genres: List of primary genres
        table_ids: List of table IDs
        row_ids: List of row IDs
        headers: List of header names
        method: Projection method
        output_dir: Output directory
    """
    print(f"Saving {method} coordinates to CSV...")
    
    df = pd.DataFrame({
        'table_id': table_ids,
        'row_id': row_ids,
        'header': headers,
        'entropy_group': entropy_groups,
        'primary_genre': genres,
        'proj_x': coords[:, 0],
        'proj_y': coords[:, 1],
        'method': method
    })
    
    csv_path = os.path.join(output_dir, f'coords_{method}.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Coordinates saved to: {csv_path}")


def save_metrics(segments_data, umap_analysis, tsne_analysis, original_analysis, output_dir):
    """
    Save metrics summary to text file.
    
    Args:
        segments_data: List of segment data
        umap_analysis: UMAP core-periphery analysis results
        tsne_analysis: t-SNE core-periphery analysis results
        original_analysis: Original embedding core-periphery analysis results
        output_dir: Output directory
    """
    print("Saving metrics...")
    
    # Count segments by entropy group
    entropy_counts = {}
    for seg in segments_data:
        group = seg['entropy_group']
        entropy_counts[group] = entropy_counts.get(group, 0) + 1
    
    # Write metrics file
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Segment Embedding Visualization Metrics\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total segments plotted: {len(segments_data)}\n")
        f.write(f"Segments by entropy group:\n")
        for group, count in sorted(entropy_counts.items()):
            f.write(f"  {group}: {count}\n")
        
        f.write(f"\nUnique headers: {len(set(seg['header'] for seg in segments_data))}\n")
        f.write(f"Unique genres: {len(set(seg['primary_genre'] for seg in segments_data))}\n")
        
        if original_analysis:
            f.write(f"\nOriginal Embedding Core-Periphery Analysis:\n")
            f.write(f"  Low entropy radius: {original_analysis['low_entropy']['mean_radius']:.3f} ± {original_analysis['low_entropy']['std_radius']:.3f}\n")
            f.write(f"  High entropy radius: {original_analysis['high_entropy']['mean_radius']:.3f} ± {original_analysis['high_entropy']['std_radius']:.3f}\n")
        
        if umap_analysis:
            f.write(f"\nUMAP Core-Periphery Analysis:\n")
            f.write(f"  Low entropy radius: {umap_analysis['low_entropy']['mean_radius']:.3f} ± {umap_analysis['low_entropy']['std_radius']:.3f}\n")
            f.write(f"  High entropy radius: {umap_analysis['high_entropy']['mean_radius']:.3f} ± {umap_analysis['high_entropy']['std_radius']:.3f}\n")
        
        if tsne_analysis:
            f.write(f"\nt-SNE Core-Periphery Analysis:\n")
            f.write(f"  Low entropy radius: {tsne_analysis['low_entropy']['mean_radius']:.3f} ± {tsne_analysis['low_entropy']['std_radius']:.3f}\n")
            f.write(f"  High entropy radius: {tsne_analysis['high_entropy']['mean_radius']:.3f} ± {tsne_analysis['high_entropy']['std_radius']:.3f}\n")
    
    print(f"✅ Metrics saved to: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize segment embeddings with entropy-aware coloring and optional table convex hulls')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to segments.json file')
    parser.add_argument('--outdir', type=str, required=True,
                       help='Output directory for plots and data')
    parser.add_argument('--model-name', type=str, default='Navi',
                       help='Model name for titles/filenames')
    parser.add_argument('--include-mid', action='store_true', default=False,
                       help='Include mid-entropy segments (default: false)')
    parser.add_argument('--umap-n-neighbors', type=int, default=30,
                       help='UMAP n_neighbors parameter')
    parser.add_argument('--umap-min-dist', type=float, default=0.05,
                       help='UMAP min_dist parameter')
    parser.add_argument('--tsne-perplexity', type=int, default=30,
                       help='t-SNE perplexity parameter')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--preprocessing', type=str, default='l2_normalize',
                       choices=['l2_normalize', 'standardize_l2', 'pca_l2'],
                       help='Embedding preprocessing method')
    parser.add_argument('--no-table-hulls', action='store_true', default=False,
                       help='Disable table convex hulls')
    
    args = parser.parse_args()
    
    print(f"Visualizing segment embeddings from {args.input}")
    print(f"Model: {args.model_name}")
    print(f"Include mid-entropy: {args.include_mid}")
    print(f"Preprocessing: {args.preprocessing}")
    print(f"Show table hulls: {not args.no_table_hulls}")
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load data
    segments_data, metadata = load_segments_data(args.input)
    
    # Extract arrays
    embeddings = np.array([seg['segment_embedding'] for seg in segments_data])
    entropy_groups = [seg['entropy_group'] for seg in segments_data]
    genres = [get_primary_genre(seg['primary_genre']) for seg in segments_data]
    table_ids = [seg['table_id'] for seg in segments_data]
    row_ids = [seg['row_id'] for seg in segments_data]
    headers = [seg['header'] for seg in segments_data]
    
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Entropy groups: {set(entropy_groups)}")
    print(f"Genres: {set(genres)}")
    print(f"Unique tables: {len(set(table_ids))}")
    
    # Preprocess embeddings
    print(f"Preprocessing embeddings with {args.preprocessing}...")
    processed_embeddings = preprocess_embeddings(embeddings, method=args.preprocessing)
    
    # Compute UMAP projection
    umap_coords = compute_umap_projection(
        processed_embeddings,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        random_state=args.seed
    )
    
    # Compute t-SNE projection
    tsne_coords = compute_tsne_projection(
        processed_embeddings,
        perplexity=args.tsne_perplexity,
        random_state=args.seed
    )
    
    # Create UMAP plot
    print("Creating UMAP plot...")
    umap_fig = create_entropy_plot(
        umap_coords, entropy_groups, table_ids, 'umap',
        title_suffix=f" ({args.model_name}, {len(segments_data)} segments)",
        include_mid=args.include_mid,
        show_table_hulls=not args.no_table_hulls
    )
    
    umap_png_path = os.path.join(args.outdir, 'umap_entropy_genre.png')
    umap_svg_path = os.path.join(args.outdir, 'umap_entropy_genre.svg')
    umap_fig.savefig(umap_png_path, dpi=300, bbox_inches='tight')
    umap_fig.savefig(umap_svg_path, bbox_inches='tight')
    plt.close(umap_fig)
    print(f"✅ UMAP plot saved to: {umap_png_path}")
    
    # Create t-SNE plot
    print("Creating t-SNE plot...")
    tsne_fig = create_entropy_plot(
        tsne_coords, entropy_groups, table_ids, 'tsne',
        title_suffix=f" ({args.model_name}, {len(segments_data)} segments)",
        include_mid=args.include_mid,
        show_table_hulls=not args.no_table_hulls
    )
    
    tsne_png_path = os.path.join(args.outdir, 'tsne_entropy_genre.png')
    tsne_svg_path = os.path.join(args.outdir, 'tsne_entropy_genre.svg')
    tsne_fig.savefig(tsne_png_path, dpi=300, bbox_inches='tight')
    tsne_fig.savefig(tsne_svg_path, bbox_inches='tight')
    plt.close(tsne_fig)
    print(f"✅ t-SNE plot saved to: {tsne_png_path}")
    
    # Core-periphery analysis using original embeddings
    original_analysis = compute_core_periphery_analysis_original(embeddings, entropy_groups, 'original')
    
    # Create density plot using original embeddings
    create_density_plot_original(embeddings, entropy_groups, 'original', args.outdir)
    
    # Core-periphery analysis using projected coordinates (keep for comparison)
    umap_analysis = compute_core_periphery_analysis(umap_coords, entropy_groups, 'umap')
    tsne_analysis = compute_core_periphery_analysis(tsne_coords, entropy_groups, 'tsne')
    
    # Create density plots using projected coordinates (keep for comparison)
    create_density_plot(umap_coords, entropy_groups, 'umap', args.outdir)
    create_density_plot(tsne_coords, entropy_groups, 'tsne', args.outdir)

    # Create combined t-SNE and density plot
    print("Creating combined t-SNE and density plot...")
    combined_fig = create_combined_tsne_density_plot(
        tsne_coords, processed_embeddings, entropy_groups, table_ids,
        title_suffix=f" (5 tables, {len(segments_data)} segments)",
        include_mid=args.include_mid,
        show_table_hulls=not args.no_table_hulls,
        output_dir=args.outdir
    )
    plt.close(combined_fig)
    
    # Save coordinates
    save_coordinates_csv(umap_coords, entropy_groups, genres, table_ids, row_ids, headers, 'umap', args.outdir)
    save_coordinates_csv(tsne_coords, entropy_groups, genres, table_ids, row_ids, headers, 'tsne', args.outdir)
    
    # Save metrics (updated to include original analysis)
    save_metrics(segments_data, umap_analysis, tsne_analysis, original_analysis, args.outdir)
    
    print("✅ Visualization complete!")


if __name__ == "__main__":
    main()
