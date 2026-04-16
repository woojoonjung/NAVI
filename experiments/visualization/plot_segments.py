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
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
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
    'low': 0.5,
    'mid': 0.3,
    'high': 0.5
}

# Table hull colors (gray only)
TABLE_HULL_COLORS = [
    '#808080'  # Gray
]

# Model hull style (for cross-model plots)
MODEL_HULL_COLOR = '#404040'  # Dark gray
MODEL_HULL_LINESTYLE = '--'   # Dashed
MODEL_HULL_LINEWIDTH = 1.5

# Semantic category colors (blue-ish palette)
SEMANTIC_CATEGORY_COLORS = {
    'Best Rating': '#00ffff',           # cyan
    'Worst Rating': '#6200ea',          # purple
    'Others': '#1f77b4'                   # blue
}

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
                       include_mid=True, figsize=(6, 5), show_table_hulls=True):
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
    # if show_table_hulls:
    #     draw_table_convex_hulls(ax, coords, table_ids, entropy_groups, include_mid)
    
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
                  c=color, marker='o', alpha=alpha, s=40, 
                  label=label, edgecolors='none')
    
    # Customize plot
    # ax.set_title(f'{method.upper()} — Segment Space{title_suffix}', fontsize=16, fontweight='bold')
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
        ax.legend(handles=entropy_patches, loc='lower right')


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
    
    # Compute centroid of all segments (low and high entropy)
    if len(filtered_coords) == 0:
        print("Warning: No points found for core-periphery analysis")
        return {}
    
    centroid = np.mean(filtered_coords, axis=0)
    
    # Get low and high entropy coordinates for distance calculations
    low_mask = filtered_groups == 'low'
    if not np.any(low_mask):
        print("Warning: No low-entropy points found for core-periphery analysis")
        return {}
    
    low_coords = filtered_coords[low_mask]
    
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
    
    # Compute centroid of all segments (low and high entropy)
    if len(filtered_embeddings) == 0:
        print("Warning: No points found for core-periphery analysis")
        return {}
    
    centroid = np.mean(filtered_embeddings, axis=0)
    
    # Get low and high entropy embeddings for distance calculations
    low_mask = filtered_groups == 'low'
    if not np.any(low_mask):
        print("Warning: No low-entropy points found for core-periphery analysis")
        return {}
    
    low_embeddings = filtered_embeddings[low_mask]
    
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
    
    # Compute centroid of all segments (low and high entropy)
    if len(filtered_coords) == 0:
        print("Warning: No points found for density plot")
        return
    
    centroid = np.mean(filtered_coords, axis=0)
    
    # Get low and high entropy coordinates for distance calculations
    low_mask = filtered_groups == 'low'
    if not np.any(low_mask):
        print("Warning: No low-entropy points found for density plot")
        return
    
    low_coords = filtered_coords[low_mask]
    
    low_distances = np.linalg.norm(low_coords - centroid, axis=1)
    high_distances = np.linalg.norm(filtered_coords[~low_mask] - centroid, axis=1)
    
    # Create density plot
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Plot histograms
    ax.hist(low_distances, bins=40, alpha=0.7, density=True, 
           label=f'Low Entropy (n={len(low_distances)})', color=ENTROPY_COLORS['low'])
    ax.hist(high_distances, bins=40, alpha=0.7, density=True, 
           label=f'High Entropy (n={len(high_distances)})', color=ENTROPY_COLORS['high'])
    
    ax.set_xlabel('Radial Distance from Segment Centroid', fontsize=14)
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution ({method.upper()})')
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
    
    # Compute centroid of all segments (low and high entropy)
    if len(filtered_embeddings) == 0:
        print("Warning: No points found for density plot")
        return
    
    centroid = np.mean(filtered_embeddings, axis=0)
    
    # Get low and high entropy embeddings for distance calculations
    low_mask = filtered_groups == 'low'
    if not np.any(low_mask):
        print("Warning: No low-entropy points found for density plot")
        return
    
    low_embeddings = filtered_embeddings[low_mask]
    
    low_distances = np.linalg.norm(low_embeddings - centroid, axis=1)
    high_distances = np.linalg.norm(filtered_embeddings[~low_mask] - centroid, axis=1)
    
    # Create density plot
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Plot histograms
    ax.hist(low_distances, bins=30, alpha=0.7, density=True, 
           label=f'Low Entropy (n={len(low_distances)})', color=ENTROPY_COLORS['low'])
    ax.hist(high_distances, bins=30, alpha=0.7, density=True, 
           label=f'High Entropy (n={len(high_distances)})', color=ENTROPY_COLORS['high'])
    
    ax.set_xlabel('Radial Distance from Segment Centroid', fontsize=17)
    ax.set_ylabel('Density')
    # ax.set_title(f'Segment Embedding Distribution ({method.upper()})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, f'density_core_periphery_{method}.png')
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
    
    # Compute centroid of all segments (low and high entropy)
    if len(filtered_embeddings) == 0:
        print("Warning: No points found for density plot")
        ax2.text(0.5, 0.5, 'No points found', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        ax2.set_title('Core-Periphery Distribution (Original Embeddings)', fontsize=16, fontweight='bold')
    else:
        centroid = np.mean(filtered_embeddings, axis=0)
        
        # Get low and high entropy embeddings for distance calculations
        low_mask = filtered_groups == 'low'
        if not np.any(low_mask):
            print("Warning: No low-entropy points found for density plot")
            ax2.text(0.5, 0.5, 'No low-entropy points found', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_title('Core-Periphery Distribution (Original Embeddings)', fontsize=16, fontweight='bold')
        else:
            low_embeddings = filtered_embeddings[low_mask]
        
            low_distances = np.linalg.norm(low_embeddings - centroid, axis=1)
            high_distances = np.linalg.norm(filtered_embeddings[~low_mask] - centroid, axis=1)
            
            # Plot histograms
            ax2.hist(low_distances, bins=30, alpha=0.7, density=True, 
                    label=f'Low Entropy (n={len(low_distances)})', color=ENTROPY_COLORS['low'])
            ax2.hist(high_distances, bins=30, alpha=0.7, density=True, 
                    label=f'High Entropy (n={len(high_distances)})', color=ENTROPY_COLORS['high'])
            
            ax2.set_xlabel('Radial Distance from Segment Centroid', fontsize=14)
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


def create_three_panel_plot(tsne_coords, embeddings, entropy_groups, table_ids,
                            title_suffix="", show_table_hulls=True, output_dir=None):
    """
    Create a three-panel plot with:
    (a) Core-Periphery Distribution - Original Embeddings (density plot)
    (b) Low entropy segments (t-SNE scatter plot)
    (c) High entropy segments (t-SNE scatter plot)
    
    Args:
        tsne_coords: 2D t-SNE coordinates (N, 2)
        embeddings: Original embeddings (N, D)
        entropy_groups: List of entropy group labels
        table_ids: List of table IDs
        title_suffix: Additional title text
        show_table_hulls: Whether to show convex hulls around tables
        output_dir: Output directory for saving the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    print("Creating three-panel plot...")
    
    # Create figure with three subplots horizontally
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # === LEFT PLOT (a): Core-Periphery Distribution from Original Embeddings ===
    # Filter to low and high entropy only
    mask = np.isin(entropy_groups, ['low', 'high'])
    filtered_embeddings = embeddings[mask]
    filtered_groups = np.array(entropy_groups)[mask]
    
    # Compute centroid of all segments (low and high entropy)
    if len(filtered_embeddings) == 0:
        print("Warning: No points found for density plot")
        ax1.text(0.5, 0.5, 'No points found', 
                transform=ax1.transAxes, ha='center', va='center', fontsize=12)
        ax1.set_title('Core-Periphery Distribution - Original Embeddings', 
                     fontsize=19, fontweight='bold')
    else:
        centroid = np.mean(filtered_embeddings, axis=0)
        
        # Get low and high entropy embeddings for distance calculations
        low_mask = filtered_groups == 'low'
        if not np.any(low_mask):
            print("Warning: No low-entropy points found for density plot")
            ax1.text(0.5, 0.5, 'No low-entropy points found', 
                    transform=ax1.transAxes, ha='center', va='center', fontsize=12)
            ax1.set_title('Core-Periphery Distribution - Original Embeddings', 
                         fontsize=19, fontweight='bold')
        else:
            low_embeddings = filtered_embeddings[low_mask]
            
            low_distances = np.linalg.norm(low_embeddings - centroid, axis=1)
            high_distances = np.linalg.norm(filtered_embeddings[~low_mask] - centroid, axis=1)
            
            # Plot histograms
            ax1.hist(low_distances, bins=30, alpha=0.7, density=True, 
                    label=f'Low Entropy (n={len(low_distances)})', color=ENTROPY_COLORS['low'])
            ax1.hist(high_distances, bins=30, alpha=0.7, density=True, 
                    label=f'High Entropy (n={len(high_distances)})', color=ENTROPY_COLORS['high'])
            
            ax1.set_xlabel('Radial Distance from Centroid (Original Embedding Space)', fontsize=14)
            ax1.set_ylabel('Density', fontsize=16)
            ax1.set_title('Core-Periphery Distribution - Original Embeddings', 
                         fontsize=16, fontweight='bold')
            ax1.legend(fontsize=12)
            ax1.grid(True, alpha=0.3)
    
    # === MIDDLE PLOT (b): Low Entropy Segments (t-SNE) ===
    # Filter to low entropy only
    low_mask_tsne = np.array(entropy_groups) == 'low'
    if not np.any(low_mask_tsne):
        print("Warning: No low-entropy points found for t-SNE plot")
        ax2.text(0.5, 0.5, 'No low-entropy points found', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        ax2.set_title('Low Entropy Segments (t-SNE)', 
                     fontsize=16, fontweight='bold')
    else:
        low_tsne_coords = tsne_coords[low_mask_tsne]
        low_table_ids = np.array(table_ids)[low_mask_tsne]
        low_entropy_groups = np.array(entropy_groups)[low_mask_tsne]
        
        # Draw convex hulls first (so they appear behind points)
        if show_table_hulls:
            draw_table_convex_hulls(ax2, low_tsne_coords, low_table_ids, 
                                   low_entropy_groups, include_mid=False)
        
        # Plot low entropy points
        color = ENTROPY_COLORS['low']
        alpha = ENTROPY_ALPHAS['low']
        
        ax2.scatter(low_tsne_coords[:, 0], low_tsne_coords[:, 1], 
                   c=color, marker='o', alpha=alpha, s=50, 
                   edgecolors='none')
        
        ax2.set_xlabel('t-SNE Dimension 1', fontsize=14)
        ax2.set_ylabel('t-SNE Dimension 2', fontsize=14)
        ax2.set_title('Low Entropy Segments (t-SNE)', 
                     fontsize=16, fontweight='bold')
        # Show ticks for t-SNE dimensions
        ax2.tick_params(labelsize=12)
    
    # === RIGHT PLOT (c): High Entropy Segments (t-SNE) ===
    # Filter to high entropy only
    high_mask_tsne = np.array(entropy_groups) == 'high'
    if not np.any(high_mask_tsne):
        print("Warning: No high-entropy points found for t-SNE plot")
        ax3.text(0.5, 0.5, 'No high-entropy points found', 
                transform=ax3.transAxes, ha='center', va='center', fontsize=12)
        ax3.set_title('High Entropy Segments (t-SNE)', 
                     fontsize=16, fontweight='bold')
    else:
        high_tsne_coords = tsne_coords[high_mask_tsne]
        high_table_ids = np.array(table_ids)[high_mask_tsne]
        high_entropy_groups = np.array(entropy_groups)[high_mask_tsne]
        
        # Filter outliers using percentile-based method (keep 2.5th to 97.5th percentile)
        # This removes extreme outliers while keeping the main clusters
        x_coords = high_tsne_coords[:, 0]
        y_coords = high_tsne_coords[:, 1]
        
        x_lower = np.percentile(x_coords, 4)
        x_upper = np.percentile(x_coords, 96)
        y_lower = np.percentile(y_coords, 4)
        y_upper = np.percentile(y_coords, 96)
        
        # Create mask to filter outliers
        outlier_mask = (
            (x_coords >= x_lower) & (x_coords <= x_upper) &
            (y_coords >= y_lower) & (y_coords <= y_upper)
        )
        
        filtered_coords = high_tsne_coords[outlier_mask]
        filtered_table_ids = np.array(high_table_ids)[outlier_mask]
        filtered_entropy_groups = np.array(high_entropy_groups)[outlier_mask]
        
        print(f"Filtered {np.sum(~outlier_mask)} outliers from {len(high_tsne_coords)} high-entropy points")
        
        # Draw convex hulls first (so they appear behind points)
        if show_table_hulls:
            draw_table_convex_hulls(ax3, filtered_coords, filtered_table_ids, 
                                   filtered_entropy_groups, include_mid=False)
        
        # Plot high entropy points
        color = ENTROPY_COLORS['high']
        alpha = ENTROPY_ALPHAS['high']
        
        ax3.scatter(filtered_coords[:, 0], filtered_coords[:, 1], 
                   c=color, marker='o', alpha=alpha, s=50, 
                   edgecolors='none')
        
        # Set axis limits to zoom in on the main cluster area
        # Use a slightly tighter range than the filtered data for better focus
        x_margin = (x_upper - x_lower) * 0.05
        y_margin = (y_upper - y_lower) * 0.05
        ax3.set_xlim(x_lower - x_margin, x_upper + x_margin)
        ax3.set_ylim(y_lower - y_margin, y_upper + y_margin)
        
        ax3.set_xlabel('t-SNE Dimension 1', fontsize=14)
        ax3.set_ylabel('t-SNE Dimension 2', fontsize=14)
        ax3.set_title('High Entropy Segments (t-SNE)', 
                     fontsize=16, fontweight='bold')
        # Show ticks for t-SNE dimensions
        ax3.tick_params(labelsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if output directory is provided
    if output_dir:
        plot_path = os.path.join(output_dir, 'three_panel_plot.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        fig.savefig(plot_path.replace('.png', '.svg'), bbox_inches='tight')
        print(f"✅ Three-panel plot saved to: {plot_path}")
    
    return fig


def create_two_panel_plot(tsne_coords, entropy_groups, table_ids,
                         title_suffix="", show_table_hulls=True, output_dir=None):
    """
    Create a two-panel plot with:
    (a) Low entropy segments (t-SNE scatter plot)
    (b) High entropy segments (t-SNE scatter plot)
    
    Args:
        tsne_coords: 2D t-SNE coordinates (N, 2)
        entropy_groups: List of entropy group labels
        table_ids: List of table IDs
        title_suffix: Additional title text
        show_table_hulls: Whether to show convex hulls around tables
        output_dir: Output directory for saving the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    print("Creating two-panel plot...")
    
    # Create figure with two subplots horizontally
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # === LEFT PLOT (a): Low Entropy Segments (t-SNE) ===
    # Filter to low entropy only
    low_mask_tsne = np.array(entropy_groups) == 'low'
    if not np.any(low_mask_tsne):
        print("Warning: No low-entropy points found for t-SNE plot")
        ax1.text(0.5, 0.5, 'No low-entropy points found', 
                transform=ax1.transAxes, ha='center', va='center', fontsize=12)
        # ax1.set_title('(a) Low Entropy Segments (t-SNE)', 
                    #  fontsize=14)
    else:
        low_tsne_coords = tsne_coords[low_mask_tsne]
        low_table_ids = np.array(table_ids)[low_mask_tsne]
        low_entropy_groups = np.array(entropy_groups)[low_mask_tsne]
        
        # Draw convex hulls first (so they appear behind points)
        if show_table_hulls:
            draw_table_convex_hulls(ax1, low_tsne_coords, low_table_ids, 
                                   low_entropy_groups, include_mid=False)
        
        # Plot low entropy points
        color = ENTROPY_COLORS['low']
        alpha = ENTROPY_ALPHAS['low']
        
        ax1.scatter(low_tsne_coords[:, 0], low_tsne_coords[:, 1], 
                   c=color, marker='o', alpha=alpha, s=50, 
                   edgecolors='none')
        
        ax1.set_xlabel('t-SNE Dimension 1', fontsize=14)
        ax1.set_ylabel('t-SNE Dimension 2', fontsize=14)
        # ax1.set_title('(a) Low Entropy Segments (t-SNE)', 
                    #  fontsize=14)
        # Show ticks for t-SNE dimensions
        ax1.tick_params(labelsize=12)
    
    # === RIGHT PLOT (b): High Entropy Segments (t-SNE) ===
    # Filter to high entropy only
    high_mask_tsne = np.array(entropy_groups) == 'high'
    if not np.any(high_mask_tsne):
        print("Warning: No high-entropy points found for t-SNE plot")
        ax2.text(0.5, 0.5, 'No high-entropy points found', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        # ax2.set_title('(b) High Entropy Segments (t-SNE)', 
                    #  fontsize=14)
    else:
        high_tsne_coords = tsne_coords[high_mask_tsne]
        high_table_ids = np.array(table_ids)[high_mask_tsne]
        high_entropy_groups = np.array(entropy_groups)[high_mask_tsne]
        
        # Filter outliers using percentile-based method (keep 2.5th to 97.5th percentile)
        # This removes extreme outliers while keeping the main clusters
        x_coords = high_tsne_coords[:, 0]
        y_coords = high_tsne_coords[:, 1]
        
        x_lower = np.percentile(x_coords, 4)
        x_upper = np.percentile(x_coords, 96)
        y_lower = np.percentile(y_coords, 4)
        y_upper = np.percentile(y_coords, 96)
        
        # Create mask to filter outliers
        outlier_mask = (
            (x_coords >= x_lower) & (x_coords <= x_upper) &
            (y_coords >= y_lower) & (y_coords <= y_upper)
        )
        
        filtered_coords = high_tsne_coords[outlier_mask]
        filtered_table_ids = np.array(high_table_ids)[outlier_mask]
        filtered_entropy_groups = np.array(high_entropy_groups)[outlier_mask]
        
        print(f"Filtered {np.sum(~outlier_mask)} outliers from {len(high_tsne_coords)} high-entropy points")
        
        # Draw convex hulls first (so they appear behind points)
        if show_table_hulls:
            draw_table_convex_hulls(ax2, filtered_coords, filtered_table_ids, 
                                   filtered_entropy_groups, include_mid=False)
        
        # Plot high entropy points
        color = ENTROPY_COLORS['high']
        alpha = ENTROPY_ALPHAS['high']
        
        ax2.scatter(filtered_coords[:, 0], filtered_coords[:, 1], 
                   c=color, marker='o', alpha=alpha, s=50, 
                   edgecolors='none')
        
        # Set axis limits to zoom in on the main cluster area
        # Use a slightly tighter range than the filtered data for better focus
        x_margin = (x_upper - x_lower) * 0.05
        y_margin = (y_upper - y_lower) * 0.05
        ax2.set_xlim(x_lower - x_margin, x_upper + x_margin)
        ax2.set_ylim(y_lower - y_margin, y_upper + y_margin)
        
        ax2.set_xlabel('t-SNE Dimension 1', fontsize=14)
        ax2.set_ylabel('t-SNE Dimension 2', fontsize=14)
        # ax2.set_title('(b) High Entropy Segments (t-SNE)', 
                    #  fontsize=14)
        # Show ticks for t-SNE dimensions
        ax2.tick_params(labelsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if output directory is provided
    if output_dir:
        plot_path = os.path.join(output_dir, 'two_panel_plot.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        fig.savefig(plot_path.replace('.png', '.svg'), bbox_inches='tight')
        print(f"✅ Two-panel plot saved to: {plot_path}")
        
        # Save individual subplots as separate SVG files
        # Save low entropy subplot
        if np.any(low_mask_tsne):
            low_fig, low_ax = plt.subplots(figsize=(6, 5))
            low_tsne_coords = tsne_coords[low_mask_tsne]
            low_table_ids = np.array(table_ids)[low_mask_tsne]
            low_entropy_groups = np.array(entropy_groups)[low_mask_tsne]
            
            if show_table_hulls:
                draw_table_convex_hulls(low_ax, low_tsne_coords, low_table_ids, 
                                       low_entropy_groups, include_mid=False)
            
            low_ax.scatter(low_tsne_coords[:, 0], low_tsne_coords[:, 1], 
                          c=ENTROPY_COLORS['low'], marker='o', alpha=ENTROPY_ALPHAS['low'], 
                          s=50, edgecolors='none')
            low_ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
            low_ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
            # low_ax.set_title('(a) Low Entropy Segments (t-SNE)', fontsize=14)
            low_ax.tick_params(labelsize=12)
            plt.tight_layout()
            
            low_plot_path = os.path.join(output_dir, 'low_entropy_tsne.svg')
            low_fig.savefig(low_plot_path, dpi=300, bbox_inches='tight')
            plt.close(low_fig)
            print(f"✅ Low entropy subplot saved to: {low_plot_path}")
        
        # Save high entropy subplot
        if np.any(high_mask_tsne):
            high_fig, high_ax = plt.subplots(figsize=(6, 5))
            high_tsne_coords = tsne_coords[high_mask_tsne]
            high_table_ids = np.array(table_ids)[high_mask_tsne]
            high_entropy_groups = np.array(entropy_groups)[high_mask_tsne]
            
            # Apply same outlier filtering
            x_coords = high_tsne_coords[:, 0]
            y_coords = high_tsne_coords[:, 1]
            x_lower = np.percentile(x_coords, 4)
            x_upper = np.percentile(x_coords, 96)
            y_lower = np.percentile(y_coords, 4)
            y_upper = np.percentile(y_coords, 96)
            
            outlier_mask = (
                (x_coords >= x_lower) & (x_coords <= x_upper) &
                (y_coords >= y_lower) & (y_coords <= y_upper)
            )
            
            filtered_coords = high_tsne_coords[outlier_mask]
            filtered_table_ids = np.array(high_table_ids)[outlier_mask]
            filtered_entropy_groups = np.array(high_entropy_groups)[outlier_mask]
            
            if show_table_hulls:
                draw_table_convex_hulls(high_ax, filtered_coords, filtered_table_ids, 
                                       filtered_entropy_groups, include_mid=False)
            
            high_ax.scatter(filtered_coords[:, 0], filtered_coords[:, 1], 
                           c=ENTROPY_COLORS['high'], marker='o', alpha=ENTROPY_ALPHAS['high'], 
                           s=50, edgecolors='none')
            
            x_margin = (x_upper - x_lower) * 0.05
            y_margin = (y_upper - y_lower) * 0.05
            high_ax.set_xlim(x_lower - x_margin, x_upper + x_margin)
            high_ax.set_ylim(y_lower - y_margin, y_upper + y_margin)
            
            high_ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
            high_ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
            # high_ax.set_title('(b) High Entropy Segments (t-SNE)', fontsize=14)
            high_ax.tick_params(labelsize=12)
            plt.tight_layout()
            
            high_plot_path = os.path.join(output_dir, 'high_entropy_tsne.svg')
            high_fig.savefig(high_plot_path, dpi=300, bbox_inches='tight')
            plt.close(high_fig)
            print(f"✅ High entropy subplot saved to: {high_plot_path}")
    
    return fig


def get_semantic_category(header):
    """
    Categorize a header field by its semantic meaning.
    
    Args:
        header: Header/field name string
        
    Returns:
        tuple: (category_name, marker_style)
    """
    header_lower = header.lower()
    
    # # Actor-related fields
    # if 'actor' in header_lower:
    #     return ('Actor-related', 'x')

    # if 'director' in header_lower or 'producer' in header_lower:
    #     return ('Director-related', '*')
    
    # Aggregaterating bestrating
    if 'bestrating' in header_lower or header == 'aggregaterating.bestrating':
        return ('Best Rating', '^')
    
    # Aggregaterating worstrating
    if 'worstrating' in header_lower or header == 'aggregaterating.worstrating':
        return ('Worst Rating', 'v')
    
    # Default: other low entropy fields
    return ('Others', 'o')


def create_two_panel_plot_vertical(tsne_coords, entropy_groups, table_ids, headers,
                                   title_suffix="", show_table_hulls=True, output_dir=None):
    """
    Create a two-panel plot with vertical stacking:
    (a) Low entropy segments (t-SNE scatter plot) - top, with semantic shape markers
    (b) High entropy segments (t-SNE scatter plot) - bottom
    
    Args:
        tsne_coords: 2D t-SNE coordinates (N, 2)
        entropy_groups: List of entropy group labels
        table_ids: List of table IDs
        headers: List of header/field names
        title_suffix: Additional title text
        show_table_hulls: Whether to show convex hulls around tables
        output_dir: Output directory for saving the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    print("Creating two-panel plot (vertical)...")
    
    # Create figure with two subplots vertically stacked
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))
    
    # === TOP PLOT (a): Low Entropy Segments (t-SNE) ===
    # Filter to low entropy only
    low_mask_tsne = np.array(entropy_groups) == 'low'
    if not np.any(low_mask_tsne):
        print("Warning: No low-entropy points found for t-SNE plot")
        ax1.text(0.5, 0.5, 'No low-entropy points found', 
                transform=ax1.transAxes, ha='center', va='center', fontsize=12)
        # ax1.set_title('(a) Low Entropy Segments (t-SNE)', 
        #              fontsize=14)
    else:
        low_tsne_coords = tsne_coords[low_mask_tsne]
        low_table_ids = np.array(table_ids)[low_mask_tsne]
        low_entropy_groups = np.array(entropy_groups)[low_mask_tsne]
        low_headers = np.array(headers)[low_mask_tsne]
        
        # Draw convex hulls first (so they appear behind points)
        if show_table_hulls:
            draw_table_convex_hulls(ax1, low_tsne_coords, low_table_ids, 
                                   low_entropy_groups, include_mid=False)
        
        # Group segments by semantic category
        category_groups = {}
        category_labels = []  # For silhouette score calculation
        for i, header in enumerate(low_headers):
            category, _ = get_semantic_category(header)
            if category not in category_groups:
                category_groups[category] = {
                    'indices': [],
                    'color': SEMANTIC_CATEGORY_COLORS.get(category, ENTROPY_COLORS['low'])
                }
            category_groups[category]['indices'].append(i)
            category_labels.append(category)
        
        # Calculate silhouette score for low entropy segments (excluding "Other" category)
        # Need at least 2 clusters and at least 2 samples per cluster
        # Filter out "Other" category for silhouette score calculation
        filtered_indices = [i for i, cat in enumerate(category_labels) if cat != 'Others']
        if len(filtered_indices) > 0:
            filtered_coords = low_tsne_coords[filtered_indices]
            filtered_labels = [category_labels[i] for i in filtered_indices]
            unique_categories = list(set(filtered_labels))
            
            if len(unique_categories) >= 2 and len(filtered_coords) >= 2:
                # Check if all clusters have at least 2 samples
                category_counts = {cat: filtered_labels.count(cat) for cat in unique_categories}
                if all(count >= 2 for count in category_counts.values()):
                    try:
                        silhouette_avg = silhouette_score(filtered_coords, filtered_labels)
                        print(f"Low entropy segments silhouette score (excluding Others): {silhouette_avg:.4f}")
                    except Exception as e:
                        print(f"Warning: Could not calculate silhouette score: {e}")
                else:
                    print(f"Warning: Some clusters have < 2 samples. Cannot calculate silhouette score.")
            else:
                print(f"Warning: Need at least 2 clusters and 2 samples. Cannot calculate silhouette score.")
        else:
            print(f"Warning: No segments remaining after filtering 'Others' category.")
        
        # Plot each category with its assigned color (including "Others")
        # Define custom order for legend: Best Rating, Worst Rating, Others, then others alphabetically
        category_order = ['Best Rating', 'Worst Rating', 'Others']
        other_categories = sorted([cat for cat in category_groups.keys() if cat not in category_order])
        ordered_categories = [cat for cat in category_order if cat in category_groups] + other_categories
        
        alpha = ENTROPY_ALPHAS['low']
        legend_elements = []
        
        for category in ordered_categories:
            data = category_groups[category]
            indices = np.array(data['indices'])
            category_coords = low_tsne_coords[indices]
            category_color = data['color']
            
            # Plot this category with same marker but different color
            ax1.scatter(category_coords[:, 0], category_coords[:, 1], 
                       c=category_color, marker='o', alpha=alpha, s=50, 
                       edgecolors='none', label=category)
            
            # Create legend element with color
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=category_color, markersize=8, 
                                        alpha=alpha, label=category, linestyle='None'))
        
        # Create legend
        if legend_elements:
            ax1.legend(handles=legend_elements, loc='best', fontsize=14)
        
        ax1.set_xlabel('t-SNE Dimension 1', fontsize=14)
        ax1.set_ylabel('t-SNE Dimension 2', fontsize=14)
        # ax1.set_title('(a) Low Entropy Segments (t-SNE)', 
        #              fontsize=14)
        # Show ticks for t-SNE dimensions
        ax1.tick_params(labelsize=12)
    
    # === BOTTOM PLOT (b): High Entropy Segments (t-SNE) ===
    # Filter to high entropy only
    high_mask_tsne = np.array(entropy_groups) == 'high'
    if not np.any(high_mask_tsne):
        print("Warning: No high-entropy points found for t-SNE plot")
        ax2.text(0.5, 0.5, 'No high-entropy points found', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        # ax2.set_title('(b) High Entropy Segments (t-SNE)', 
        #              fontsize=14)
    else:
        high_tsne_coords = tsne_coords[high_mask_tsne]
        high_table_ids = np.array(table_ids)[high_mask_tsne]
        high_entropy_groups = np.array(entropy_groups)[high_mask_tsne]
        
        # Filter outliers using percentile-based method (keep 4th to 96th percentile)
        # This removes extreme outliers while keeping the main clusters
        x_coords = high_tsne_coords[:, 0]
        y_coords = high_tsne_coords[:, 1]
        
        x_lower = np.percentile(x_coords, 4)
        x_upper = np.percentile(x_coords, 96)
        y_lower = np.percentile(y_coords, 4)
        y_upper = np.percentile(y_coords, 96)
        
        # Create mask to filter outliers
        outlier_mask = (
            (x_coords >= x_lower) & (x_coords <= x_upper) &
            (y_coords >= y_lower) & (y_coords <= y_upper)
        )
        
        filtered_coords = high_tsne_coords[outlier_mask]
        filtered_table_ids = np.array(high_table_ids)[outlier_mask]
        filtered_entropy_groups = np.array(high_entropy_groups)[outlier_mask]
        
        print(f"Filtered {np.sum(~outlier_mask)} outliers from {len(high_tsne_coords)} high-entropy points")
        
        # Draw convex hulls first (so they appear behind points)
        if show_table_hulls:
            draw_table_convex_hulls(ax2, filtered_coords, filtered_table_ids, 
                                   filtered_entropy_groups, include_mid=False)
        
        # Plot high entropy points
        color = ENTROPY_COLORS['high']
        alpha = ENTROPY_ALPHAS['high']
        
        ax2.scatter(filtered_coords[:, 0], filtered_coords[:, 1], 
                   c=color, marker='o', alpha=alpha, s=50, 
                   edgecolors='none')
        
        # Set axis limits to zoom in on the main cluster area
        # Use a slightly tighter range than the filtered data for better focus
        x_margin = (x_upper - x_lower) * 0.05
        y_margin = (y_upper - y_lower) * 0.05
        ax2.set_xlim(x_lower - x_margin, x_upper + x_margin)
        ax2.set_ylim(y_lower - y_margin, y_upper + y_margin)
        
        ax2.set_xlabel('t-SNE Dimension 1', fontsize=14)
        ax2.set_ylabel('t-SNE Dimension 2', fontsize=14)
        # ax2.set_title('(b) High Entropy Segments (t-SNE)', 
        #              fontsize=14)
        # Show ticks for t-SNE dimensions
        ax2.tick_params(labelsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if output directory is provided
    if output_dir:
        plot_path = os.path.join(output_dir, 'two_panel_plot_vertical.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        fig.savefig(plot_path.replace('.png', '.svg'), bbox_inches='tight')
        print(f"✅ Two-panel plot (vertical) saved to: {plot_path}")
        
        # Save individual subplots as separate SVG files
        # Save low entropy subplot
        if np.any(low_mask_tsne):
            low_fig, low_ax = plt.subplots(figsize=(6, 5))
            low_tsne_coords = tsne_coords[low_mask_tsne]
            low_table_ids = np.array(table_ids)[low_mask_tsne]
            low_entropy_groups = np.array(entropy_groups)[low_mask_tsne]
            low_headers = np.array(headers)[low_mask_tsne]
            
            if show_table_hulls:
                draw_table_convex_hulls(low_ax, low_tsne_coords, low_table_ids, 
                                       low_entropy_groups, include_mid=False)
            
            # Group segments by semantic category
            category_groups = {}
            category_labels = []  # For silhouette score calculation
            for i, header in enumerate(low_headers):
                category, _ = get_semantic_category(header)
                if category not in category_groups:
                    category_groups[category] = {
                        'indices': [],
                        'color': SEMANTIC_CATEGORY_COLORS.get(category, ENTROPY_COLORS['low'])
                    }
                category_groups[category]['indices'].append(i)
                category_labels.append(category)
            
            # Calculate silhouette score for low entropy segments (excluding "Other" category)
            # Filter out "Other" category for silhouette score calculation
            filtered_indices = [i for i, cat in enumerate(category_labels) if cat != 'Others']
            if len(filtered_indices) > 0:
                filtered_coords = low_tsne_coords[filtered_indices]
                filtered_labels = [category_labels[i] for i in filtered_indices]
                unique_categories = list(set(filtered_labels))
                
                if len(unique_categories) >= 2 and len(filtered_coords) >= 2:
                    category_counts = {cat: filtered_labels.count(cat) for cat in unique_categories}
                    if all(count >= 2 for count in category_counts.values()):
                        try:
                            silhouette_avg = silhouette_score(filtered_coords, filtered_labels)
                            print(f"Low entropy segments silhouette score (individual plot, excluding Others): {silhouette_avg:.4f}")
                        except Exception as e:
                            print(f"Warning: Could not calculate silhouette score: {e}")
                    else:
                        print(f"Warning: Some clusters have < 2 samples. Cannot calculate silhouette score.")
                else:
                    print(f"Warning: Need at least 2 clusters and 2 samples. Cannot calculate silhouette score.")
            else:
                print(f"Warning: No segments remaining after filtering 'Others' category.")
            
            # Plot each category with its assigned color (including "Others")
            # Define custom order for legend: Best Rating, Worst Rating, Others, then others alphabetically
            category_order = ['Best Rating', 'Worst Rating', 'Others']
            other_categories = sorted([cat for cat in category_groups.keys() if cat not in category_order])
            ordered_categories = [cat for cat in category_order if cat in category_groups] + other_categories
            
            alpha = ENTROPY_ALPHAS['low']
            legend_elements = []
            
            for category in ordered_categories:
                data = category_groups[category]
                indices = np.array(data['indices'])
                category_coords = low_tsne_coords[indices]
                category_color = data['color']
                
                low_ax.scatter(category_coords[:, 0], category_coords[:, 1], 
                              c=category_color, marker='o', alpha=alpha, s=50, 
                              edgecolors='none', label=category)
                
                legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=category_color, markersize=8, 
                                            alpha=alpha, label=category, linestyle='None'))
            
            if legend_elements:
                low_ax.legend(handles=legend_elements, loc='best', fontsize=14)
            
            low_ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
            low_ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
            # low_ax.set_title('(a) Low Entropy Segments (t-SNE)', fontsize=14)
            low_ax.tick_params(labelsize=12)
            plt.tight_layout()
            
            low_plot_path = os.path.join(output_dir, 'low_entropy_tsne_vertical.svg')
            low_fig.savefig(low_plot_path, dpi=300, bbox_inches='tight')
            plt.close(low_fig)
            print(f"✅ Low entropy subplot (vertical) saved to: {low_plot_path}")
        
        # Save high entropy subplot
        if np.any(high_mask_tsne):
            high_fig, high_ax = plt.subplots(figsize=(6, 5))
            high_tsne_coords = tsne_coords[high_mask_tsne]
            high_table_ids = np.array(table_ids)[high_mask_tsne]
            high_entropy_groups = np.array(entropy_groups)[high_mask_tsne]
            
            # Apply same outlier filtering
            x_coords = high_tsne_coords[:, 0]
            y_coords = high_tsne_coords[:, 1]
            x_lower = np.percentile(x_coords, 4)
            x_upper = np.percentile(x_coords, 96)
            y_lower = np.percentile(y_coords, 4)
            y_upper = np.percentile(y_coords, 96)
            
            outlier_mask = (
                (x_coords >= x_lower) & (x_coords <= x_upper) &
                (y_coords >= y_lower) & (y_coords <= y_upper)
            )
            
            filtered_coords = high_tsne_coords[outlier_mask]
            filtered_table_ids = np.array(high_table_ids)[outlier_mask]
            filtered_entropy_groups = np.array(high_entropy_groups)[outlier_mask]
            
            if show_table_hulls:
                draw_table_convex_hulls(high_ax, filtered_coords, filtered_table_ids, 
                                       filtered_entropy_groups, include_mid=False)
            
            high_ax.scatter(filtered_coords[:, 0], filtered_coords[:, 1], 
                           c=ENTROPY_COLORS['high'], marker='o', alpha=ENTROPY_ALPHAS['high'], 
                           s=50, edgecolors='none')
            
            x_margin = (x_upper - x_lower) * 0.05
            y_margin = (y_upper - y_lower) * 0.05
            high_ax.set_xlim(x_lower - x_margin, x_upper + x_margin)
            high_ax.set_ylim(y_lower - y_margin, y_upper + y_margin)
            
            high_ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
            high_ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
            # high_ax.set_title('(b) High Entropy Segments (t-SNE)', fontsize=14)
            high_ax.tick_params(labelsize=12)
            plt.tight_layout()
            
            high_plot_path = os.path.join(output_dir, 'high_entropy_tsne_vertical.svg')
            high_fig.savefig(high_plot_path, dpi=300, bbox_inches='tight')
            plt.close(high_fig)
            print(f"✅ High entropy subplot (vertical) saved to: {high_plot_path}")
    
    return fig


def create_two_panel_plot_vertical_simple(tsne_coords, entropy_groups, table_ids,
                                          title_suffix="", show_table_hulls=True, output_dir=None):
    """
    Create a simplified two-panel plot with vertical stacking (rollback version):
    (a) Low entropy segments (t-SNE scatter plot) - top, simple circles
    (b) High entropy segments (t-SNE scatter plot) - bottom
    
    This version does NOT include:
    - Semantic category grouping
    - Different shapes/markers
    - Legends
    - Header name labels
    
    Args:
        tsne_coords: 2D t-SNE coordinates (N, 2)
        entropy_groups: List of entropy group labels
        table_ids: List of table IDs
        title_suffix: Additional title text
        show_table_hulls: Whether to show convex hulls around tables
        output_dir: Output directory for saving the plot
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    print("Creating two-panel plot (vertical, simple)...")
    
    # Create figure with two subplots vertically stacked
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 10))
    
    # === TOP PLOT (a): Low Entropy Segments (t-SNE) ===
    # Filter to low entropy only
    low_mask_tsne = np.array(entropy_groups) == 'low'
    if not np.any(low_mask_tsne):
        print("Warning: No low-entropy points found for t-SNE plot")
        ax1.text(0.5, 0.5, 'No low-entropy points found', 
                transform=ax1.transAxes, ha='center', va='center', fontsize=12)
        ax1.set_title('(a) Low Entropy Segments (t-SNE)', 
                     fontsize=14)
    else:
        low_tsne_coords = tsne_coords[low_mask_tsne]
        low_table_ids = np.array(table_ids)[low_mask_tsne]
        low_entropy_groups = np.array(entropy_groups)[low_mask_tsne]
        
        # Draw convex hulls first (so they appear behind points)
        if show_table_hulls:
            draw_table_convex_hulls(ax1, low_tsne_coords, low_table_ids, 
                                   low_entropy_groups, include_mid=False)
        
        # Plot low entropy points (simple circles, no categorization)
        color = ENTROPY_COLORS['low']
        alpha = ENTROPY_ALPHAS['low']
        
        ax1.scatter(low_tsne_coords[:, 0], low_tsne_coords[:, 1], 
                   c=color, marker='o', alpha=alpha, s=50, 
                   edgecolors='none')
        
        ax1.set_xlabel('t-SNE Dimension 1', fontsize=14)
        ax1.set_ylabel('t-SNE Dimension 2', fontsize=14)
        ax1.set_title('(a) Low Entropy Segments (t-SNE)', 
                     fontsize=14)
        # Show ticks for t-SNE dimensions
        ax1.tick_params(labelsize=12)
    
    # === BOTTOM PLOT (b): High Entropy Segments (t-SNE) ===
    # Filter to high entropy only
    high_mask_tsne = np.array(entropy_groups) == 'high'
    if not np.any(high_mask_tsne):
        print("Warning: No high-entropy points found for t-SNE plot")
        ax2.text(0.5, 0.5, 'No high-entropy points found', 
                transform=ax2.transAxes, ha='center', va='center', fontsize=12)
        ax2.set_title('(b) High Entropy Segments (t-SNE)', 
                     fontsize=14)
    else:
        high_tsne_coords = tsne_coords[high_mask_tsne]
        high_table_ids = np.array(table_ids)[high_mask_tsne]
        high_entropy_groups = np.array(entropy_groups)[high_mask_tsne]
        
        # Filter outliers using percentile-based method (keep 4th to 96th percentile)
        # This removes extreme outliers while keeping the main clusters
        x_coords = high_tsne_coords[:, 0]
        y_coords = high_tsne_coords[:, 1]
        
        x_lower = np.percentile(x_coords, 4)
        x_upper = np.percentile(x_coords, 96)
        y_lower = np.percentile(y_coords, 4)
        y_upper = np.percentile(y_coords, 96)
        
        # Create mask to filter outliers
        outlier_mask = (
            (x_coords >= x_lower) & (x_coords <= x_upper) &
            (y_coords >= y_lower) & (y_coords <= y_upper)
        )
        
        filtered_coords = high_tsne_coords[outlier_mask]
        filtered_table_ids = np.array(high_table_ids)[outlier_mask]
        filtered_entropy_groups = np.array(high_entropy_groups)[outlier_mask]
        
        print(f"Filtered {np.sum(~outlier_mask)} outliers from {len(high_tsne_coords)} high-entropy points")
        
        # Draw convex hulls first (so they appear behind points)
        if show_table_hulls:
            draw_table_convex_hulls(ax2, filtered_coords, filtered_table_ids, 
                                   filtered_entropy_groups, include_mid=False)
        
        # Plot high entropy points
        color = ENTROPY_COLORS['high']
        alpha = ENTROPY_ALPHAS['high']
        
        ax2.scatter(filtered_coords[:, 0], filtered_coords[:, 1], 
                   c=color, marker='o', alpha=alpha, s=50, 
                   edgecolors='none')
        
        # Set axis limits to zoom in on the main cluster area
        # Use a slightly tighter range than the filtered data for better focus
        x_margin = (x_upper - x_lower) * 0.05
        y_margin = (y_upper - y_lower) * 0.05
        ax2.set_xlim(x_lower - x_margin, x_upper + x_margin)
        ax2.set_ylim(y_lower - y_margin, y_upper + y_margin)
        
        ax2.set_xlabel('t-SNE Dimension 1', fontsize=14)
        ax2.set_ylabel('t-SNE Dimension 2', fontsize=14)
        ax2.set_title('(b) High Entropy Segments (t-SNE)', 
                     fontsize=14)
        # Show ticks for t-SNE dimensions
        ax2.tick_params(labelsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if output directory is provided
    if output_dir:
        plot_path = os.path.join(output_dir, 'two_panel_plot_vertical_simple.png')
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        fig.savefig(plot_path.replace('.png', '.svg'), bbox_inches='tight')
        print(f"✅ Two-panel plot (vertical, simple) saved to: {plot_path}")
    
    return fig


def save_coordinates_csv(coords, entropy_groups, table_ids, row_ids, headers, 
                        method, output_dir, model_labels=None):
    """
    Save 2D coordinates and metadata to CSV.
    
    Args:
        coords: 2D coordinates (N, 2)
        entropy_groups: List of entropy group labels
        table_ids: List of table IDs
        row_ids: List of row IDs
        headers: List of header names
        method: Projection method
        output_dir: Output directory
        model_labels: Optional list of model labels (for cross-model plots)
    """
    print(f"Saving {method} coordinates to CSV...")
    
    data_dict = {
        'table_id': table_ids,
        'row_id': row_ids,
        'header': headers,
        'entropy_group': entropy_groups,
        'proj_x': coords[:, 0],
        'proj_y': coords[:, 1],
        'method': method
    }
    
    # Add model labels if provided
    if model_labels is not None:
        data_dict['model'] = model_labels
    
    df = pd.DataFrame(data_dict)
    
    csv_path = os.path.join(output_dir, f'coords_{method}.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Coordinates saved to: {csv_path}")


def draw_model_convex_hulls(ax, coords, model_labels):
    """
    Draw convex hulls around segments from each model (dark gray dashed, no fill).
    Also add model name labels near the hulls.
    
    Args:
        ax: Matplotlib axis
        coords: 2D coordinates (N, 2)
        model_labels: List of model labels
    """
    print("Drawing convex hulls for each model...")
    
    # Group coordinates by model
    model_coords = {}
    for i, model_label in enumerate(model_labels):
        if model_label not in model_coords:
            model_coords[model_label] = []
        model_coords[model_label].append(coords[i])
    
    # Draw convex hull for each model
    for model_label, coords_list in model_coords.items():
        if len(coords_list) < 3:  # Need at least 3 points for convex hull
            continue
            
        coords_array = np.array(coords_list)
        
        try:
            # Compute convex hull
            hull = ConvexHull(coords_array)
            
            # Get hull vertices
            hull_points = coords_array[hull.vertices]
            
            # Create polygon (no fill, dark gray dashed border)
            hull_polygon = plt.Polygon(hull_points, 
                                     fill=False,
                                     edgecolor=MODEL_HULL_COLOR,
                                     linestyle=MODEL_HULL_LINESTYLE,
                                     linewidth=MODEL_HULL_LINEWIDTH)
            
            ax.add_patch(hull_polygon)
            
            # Add model name label near the hull
            # Use the centroid of the hull as label position
            centroid = coords_array.mean(axis=0)
            
            # Find a point on the hull boundary that's furthest from centroid for better label placement
            distances = np.linalg.norm(hull_points - centroid, axis=1)
            furthest_idx = np.argmax(distances)
            label_pos = hull_points[furthest_idx]
            
            # Format model label for display
            display_label = 'NAVI' if model_label == 'segment' else 'BERT'
            
            # ax.text(label_pos[0], label_pos[1], display_label, 
            #        fontsize=13,
            #        color=MODEL_HULL_COLOR,
            #        bbox=dict(boxstyle='round,pad=0.5', 
            #                facecolor='none', 
            #                edgecolor=MODEL_HULL_COLOR,
            #                linestyle=MODEL_HULL_LINESTYLE))
            
        except Exception as e:
            print(f"Warning: Could not create convex hull for model {model_label}: {e}")
            continue


def create_cross_model_plot(coords, entropy_groups, model_labels, table_ids, method, 
                            title_suffix="", include_mid=True, figsize=(6, 5), 
                            show_table_hulls=True):
    """
    Create cross-model visualization plot showing segments from two models.
    
    Args:
        coords: 2D coordinates (N, 2)
        entropy_groups: List of entropy group labels
        model_labels: List of model labels ('segment' or 'segment_bert')
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
        model_labels = np.array(model_labels)[mask]
        table_ids = np.array(table_ids)[mask]
    
    # Get unique entropy groups and model labels
    unique_entropy_groups = sorted(set(entropy_groups))
    unique_models = sorted(set(model_labels))
    
    MODEL_MARKERS = {
        'segment': 'o',      # Circle
        'segment_bert': 'x'  # Cross
    }
    
    # Draw convex hulls first (so they appear behind points), separated by model
    if show_table_hulls:
        # Draw table hulls for each model separately
        for model_label in unique_models:
            # Filter data for this model
            model_mask = np.array(model_labels) == model_label
            model_coords = coords[model_mask]
            model_table_ids = np.array(table_ids)[model_mask]
            model_entropy_groups = np.array(entropy_groups)[model_mask]
            
            # Draw table hulls for this model
            draw_table_convex_hulls(ax, model_coords, model_table_ids, model_entropy_groups, include_mid)
    
    # Draw model convex hulls (after table hulls, before points)
    draw_model_convex_hulls(ax, coords, model_labels)
    
    # Create plots for each combination of entropy group and model
    legend_elements = []
    
    for entropy_group in unique_entropy_groups:
        if entropy_group == 'mid' and not include_mid:
            continue
        
        for model_label in unique_models:
            # Filter by both entropy group and model
            mask = (np.array(entropy_groups) == entropy_group) & (np.array(model_labels) == model_label)
            if not np.any(mask):
                continue
                
            group_coords = coords[mask]
            
            # Get color and alpha based on entropy group
            base_color = ENTROPY_COLORS.get(entropy_group, '#000000')
            alpha = ENTROPY_ALPHAS.get(entropy_group, 0.5)
            
            # Use model-specific marker
            marker = MODEL_MARKERS.get(model_label, 'o')
            
            # Create label
            label = f"{entropy_group.title()} Entropy ({model_label})"
            
            # For segment model (circle), use hollow markers with colored border
            # For segment_bert model (x), use filled markers
            if model_label == 'segment':
                # Hollow circle with colored border
                ax.scatter(group_coords[:, 0], group_coords[:, 1], 
                          facecolors='none', edgecolors=base_color, marker=marker, 
                          alpha=alpha, s=40, linewidths=1.5,
                          label=label)
                
                # Add to legend elements (hollow circle)
                legend_elements.append(Line2D([0], [0], marker=marker, color='w', 
                                            markerfacecolor='none', markeredgecolor=base_color,
                                            markersize=8, markeredgewidth=1.5,
                                            alpha=alpha, label=label, linestyle='None'))
            else:
                # Filled marker for segment_bert
                ax.scatter(group_coords[:, 0], group_coords[:, 1], 
                          c=base_color, marker=marker, alpha=alpha, s=40, 
                          label=label, edgecolors='none')
                
                # Add to legend elements (filled marker)
                legend_elements.append(Line2D([0], [0], marker=marker, color='w', 
                                            markerfacecolor=base_color, markersize=8, 
                                            alpha=alpha, label=label, linestyle='None'))
    
    # Customize plot
    # ax.set_title(f'{method.upper()} — Cross-Model Segment Space{title_suffix}', 
    #             fontsize=16, fontweight='bold')
    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=14)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=14)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create legend
    # if legend_elements:
    #     ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def load_and_combine_cross_model_segments(input_segment, input_segment_bert, domain):
    """
    Load segments from both models and combine them with model labels.
    
    Args:
        input_segment: Path to segments.json for segment model
        input_segment_bert: Path to segments.json for segment_bert model
        domain: Domain name (e.g., 'movie' or 'product')
        
    Returns:
        tuple: (combined_segments_data, metadata)
    """
    print(f"\nLoading segment embeddings from both models for domain: {domain}")
    
    # Load segment model data
    print(f"Loading from segment model: {input_segment}")
    segment_data, segment_metadata = load_segments_data(input_segment)
    
    # Load segment_bert model data
    print(f"Loading from segment_bert model: {input_segment_bert}")
    segment_bert_data, segment_bert_metadata = load_segments_data(input_segment_bert)
    
    # Add model labels
    combined_segments = []
    
    for seg in segment_data:
        seg_copy = seg.copy()
        seg_copy['model'] = 'segment'
        combined_segments.append(seg_copy)
    
    for seg in segment_bert_data:
        seg_copy = seg.copy()
        seg_copy['model'] = 'segment_bert'
        combined_segments.append(seg_copy)
    
    # Combine metadata
    combined_metadata = {
        'domain': domain,
        'segment_model': segment_metadata,
        'segment_bert_model': segment_bert_metadata
    }
    
    print(f"\nCombined segments:")
    print(f"  Segment model: {len(segment_data)} segments")
    print(f"  Segment_BERT model: {len(segment_bert_data)} segments")
    print(f"  Total: {len(combined_segments)} segments")
    
    return combined_segments, combined_metadata


def save_metrics(segments_data, umap_analysis, tsne_analysis, original_analysis, output_dir):
    """
    Save Core-Periphery analysis metrics to text file.
    
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
    parser.add_argument('--input', type=str, default=None,
                       help='Path to segments.json file (required unless --cross-model)')
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
    parser.add_argument('--ablation-mode', action='store_true', default=False,
                       help='Enable ablation mode for ablation model naming')
    
    # Cross-model arguments
    parser.add_argument('--cross-model', action='store_true', default=False,
                       help='Enable cross-model visualization (combines segment and segment_bert)')
    parser.add_argument('--input-segment', type=str, default=None,
                       help='Path to segments.json for segment model (required if --cross-model)')
    parser.add_argument('--input-segment-bert', type=str, default=None,
                       help='Path to segments.json for segment_bert model (required if --cross-model)')
    parser.add_argument('--domain', type=str, default=None,
                       choices=['movie', 'product'],
                       help='Domain name (required if --cross-model)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.cross_model:
        if not args.input_segment or not args.input_segment_bert or not args.domain:
            parser.error("--cross-model requires --input-segment, --input-segment-bert, and --domain")
    else:
        if not args.input:
            parser.error("--input is required unless --cross-model is specified")
    
    if args.cross_model:
        print(f"Cross-model visualization for domain: {args.domain}")
        print(f"  Segment model: {args.input_segment}")
        print(f"  Segment_BERT model: {args.input_segment_bert}")
    else:
        print(f"Visualizing segment embeddings from {args.input}")
    print(f"Model: {args.model_name}")
    print(f"Ablation mode: {args.ablation_mode}")
    print(f"Include mid-entropy: {args.include_mid}")
    print(f"Preprocessing: {args.preprocessing}")
    print(f"Show table hulls: {not args.no_table_hulls}")
    
    # Adjust output directory for ablation mode
    if args.ablation_mode:
        # Create ablation subdirectory if needed
        if 'ablation' not in args.outdir:
            base_outdir = os.path.dirname(args.outdir) if os.path.dirname(args.outdir) else args.outdir
            outdir_name = os.path.basename(args.outdir)
            outdir = os.path.join(base_outdir, 'ablation', outdir_name)
        else:
            outdir = args.outdir
    else:
        outdir = args.outdir
    
    # Create output directory
    os.makedirs(outdir, exist_ok=True)
    
    # Load data
    if args.cross_model:
        # Cross-model mode: load from both models
        segments_data, metadata = load_and_combine_cross_model_segments(
            args.input_segment, args.input_segment_bert, args.domain
        )
    else:
        # Single model mode
        segments_data, metadata = load_segments_data(args.input)
    
    # Extract arrays
    embeddings = np.array([seg['segment_embedding'] for seg in segments_data])
    entropy_groups = [seg['entropy_group'] for seg in segments_data]
    table_ids = [seg['table_id'] for seg in segments_data]
    row_ids = [seg['row_id'] for seg in segments_data]
    headers = [seg['header'] for seg in segments_data]
    
    # Extract model labels if in cross-model mode
    if args.cross_model:
        model_labels = [seg['model'] for seg in segments_data]
    else:
        model_labels = None
    
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Entropy groups: {set(entropy_groups)}")
    print(f"Unique tables: {len(set(table_ids))}")
    if args.cross_model:
        print(f"Models: {set(model_labels)}")
    
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
    model_title = f"Ablation Model {args.model_name}" if args.ablation_mode else args.model_name
    umap_fig = create_entropy_plot(
        umap_coords, entropy_groups, table_ids, 'umap',
        title_suffix=f" ({model_title}, {len(segments_data)} segments)",
        include_mid=args.include_mid,
        show_table_hulls=not args.no_table_hulls
    )
    
    umap_png_path = os.path.join(outdir, 'umap_entropy.png')
    umap_svg_path = os.path.join(outdir, 'umap_entropy.svg')
    umap_fig.savefig(umap_png_path, dpi=300, bbox_inches='tight')
    umap_fig.savefig(umap_svg_path, bbox_inches='tight')
    plt.close(umap_fig)
    print(f"✅ UMAP plot saved to: {umap_png_path}")
    
    # Create t-SNE plot
    print("Creating t-SNE plot...")
    tsne_fig = create_entropy_plot(
        tsne_coords, entropy_groups, table_ids, 'tsne',
        title_suffix=f" ({model_title}, {len(segments_data)} segments)",
        include_mid=args.include_mid,
        show_table_hulls=not args.no_table_hulls
    )
    
    tsne_png_path = os.path.join(outdir, 'tsne_entropy.png')
    tsne_svg_path = os.path.join(outdir, 'tsne_entropy.svg')
    tsne_fig.savefig(tsne_png_path, dpi=300, bbox_inches='tight')
    tsne_fig.savefig(tsne_svg_path, bbox_inches='tight')
    plt.close(tsne_fig)
    print(f"✅ t-SNE plot saved to: {tsne_png_path}")
    
    # Create cross-model plots if in cross-model mode
    if args.cross_model:
        print("\nCreating cross-model plots...")
        
        # Cross-model UMAP plot
        print("Creating cross-model UMAP plot...")
        cross_umap_fig = create_cross_model_plot(
            umap_coords, entropy_groups, model_labels, table_ids, 'umap',
            title_suffix=f" ({args.domain.title()} domain, {len(segments_data)} segments)",
            include_mid=args.include_mid,
            show_table_hulls=not args.no_table_hulls
        )
        
        cross_umap_png_path = os.path.join(outdir, 'cross_model_umap.png')
        cross_umap_svg_path = os.path.join(outdir, 'cross_model_umap.svg')
        cross_umap_fig.savefig(cross_umap_png_path, dpi=300, bbox_inches='tight')
        cross_umap_fig.savefig(cross_umap_svg_path, bbox_inches='tight')
        plt.close(cross_umap_fig)
        print(f"✅ Cross-model UMAP plot saved to: {cross_umap_png_path}")
        
        # Cross-model t-SNE plot
        print("Creating cross-model t-SNE plot...")
        cross_tsne_fig = create_cross_model_plot(
            tsne_coords, entropy_groups, model_labels, table_ids, 'tsne',
            title_suffix=f" ({args.domain.title()} domain, {len(segments_data)} segments)",
            include_mid=args.include_mid,
            show_table_hulls=not args.no_table_hulls
        )
        
        cross_tsne_png_path = os.path.join(outdir, 'cross_model_tsne.png')
        cross_tsne_svg_path = os.path.join(outdir, 'cross_model_tsne.svg')
        cross_tsne_fig.savefig(cross_tsne_png_path, dpi=300, bbox_inches='tight')
        cross_tsne_fig.savefig(cross_tsne_svg_path, bbox_inches='tight')
        plt.close(cross_tsne_fig)
        print(f"✅ Cross-model t-SNE plot saved to: {cross_tsne_png_path}")
        
        # Save cross-model coordinates
        save_coordinates_csv(umap_coords, entropy_groups, table_ids, row_ids, headers, 
                           'cross_model_umap', outdir, model_labels=model_labels)
        save_coordinates_csv(tsne_coords, entropy_groups, table_ids, row_ids, headers, 
                           'cross_model_tsne', outdir, model_labels=model_labels)
    
    # Core-periphery analysis using original embeddings
    original_analysis = compute_core_periphery_analysis_original(embeddings, entropy_groups, 'original')
    
    # Create density plot using original embeddings
    create_density_plot_original(embeddings, entropy_groups, 'original', outdir)
    
    # Core-periphery analysis using projected coordinates (keep for comparison)
    umap_analysis = compute_core_periphery_analysis(umap_coords, entropy_groups, 'umap')
    tsne_analysis = compute_core_periphery_analysis(tsne_coords, entropy_groups, 'tsne')
    
    # Create density plots using projected coordinates (keep for comparison)
    create_density_plot(umap_coords, entropy_groups, 'umap', outdir)
    create_density_plot(tsne_coords, entropy_groups, 'tsne', outdir)

    # Create combined t-SNE and density plot
    print("Creating combined t-SNE and density plot...")
    combined_fig = create_combined_tsne_density_plot(
        tsne_coords, processed_embeddings, entropy_groups, table_ids,
        title_suffix=f" ({model_title}, {len(segments_data)} segments)",
        include_mid=args.include_mid,
        show_table_hulls=not args.no_table_hulls,
        output_dir=outdir
    )
    plt.close(combined_fig)
    
    # Create three-panel plot
    print("Creating three-panel plot...")
    three_panel_fig = create_three_panel_plot(
        tsne_coords, embeddings, entropy_groups, table_ids,
        title_suffix=f" ({model_title}, {len(segments_data)} segments)",
        show_table_hulls=not args.no_table_hulls,
        output_dir=outdir
    )
    plt.close(three_panel_fig)
    
    # Create two-panel plot
    print("Creating two-panel plot...")
    two_panel_fig = create_two_panel_plot(
        tsne_coords, entropy_groups, table_ids,
        title_suffix=f" ({model_title}, {len(segments_data)} segments)",
        show_table_hulls=not args.no_table_hulls,
        output_dir=outdir
    )
    plt.close(two_panel_fig)
    
    # Create two-panel plot (vertical)
    print("Creating two-panel plot (vertical)...")
    two_panel_vertical_fig = create_two_panel_plot_vertical(
        tsne_coords, entropy_groups, table_ids, headers,
        title_suffix=f" ({model_title}, {len(segments_data)} segments)",
        show_table_hulls=not args.no_table_hulls,
        output_dir=outdir
    )
    plt.close(two_panel_vertical_fig)
    
    # Create two-panel plot (vertical, simple - rollback version)
    print("Creating two-panel plot (vertical, simple)...")
    two_panel_vertical_simple_fig = create_two_panel_plot_vertical_simple(
        tsne_coords, entropy_groups, table_ids,
        title_suffix=f" ({model_title}, {len(segments_data)} segments)",
        show_table_hulls=not args.no_table_hulls,
        output_dir=outdir
    )
    plt.close(two_panel_vertical_simple_fig)
    
    # Save coordinates
    save_coordinates_csv(umap_coords, entropy_groups, table_ids, row_ids, headers, 'umap', outdir)
    save_coordinates_csv(tsne_coords, entropy_groups, table_ids, row_ids, headers, 'tsne', outdir)
    
    # Save metrics (updated to include original analysis)
    save_metrics(segments_data, umap_analysis, tsne_analysis, original_analysis, outdir)
    
    print("✅ Visualization complete!")


if __name__ == "__main__":
    main()
