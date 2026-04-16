#!/usr/bin/env python3
"""
Visualize segment embeddings from multiple domains in a combined embedding space.

This script creates UMAP and t-SNE visualizations showing:
- Low-entropy segments (blue) and high-entropy segments (red) from multiple domains
- Individual table convex hulls (gray)
- Domain-level convex hulls (dark gray dashed, with domain labels)
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
    'high': '#d62728'    # Red
}

# Entropy group alphas
ENTROPY_ALPHAS = {
    'low': 0.5,
    'high': 0.5
}

# Table hull colors (gray)
TABLE_HULL_COLORS = ['#808080']  # Gray

# Domain hull style
DOMAIN_HULL_COLOR = '#404040'  # Dark gray
DOMAIN_HULL_LINESTYLE = '--'   # Dashed
DOMAIN_HULL_LINEWIDTH = 1.5


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


def draw_table_convex_hulls(ax, coords, table_ids):
    """
    Draw convex hulls around segments from the same table (gray).
    
    Args:
        ax: Matplotlib axis
        coords: 2D coordinates (N, 2)
        table_ids: List of table IDs
    """
    print("Drawing convex hulls for each table...")
    
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
            
            # Create polygon (gray fill)
            hull_polygon = plt.Polygon(hull_points, 
                                     alpha=0.15,
                                     facecolor=TABLE_HULL_COLORS[table_idx % len(TABLE_HULL_COLORS)],
                                     edgecolor=TABLE_HULL_COLORS[table_idx % len(TABLE_HULL_COLORS)],
                                     linewidth=1)
            
            ax.add_patch(hull_polygon)
            
        except Exception as e:
            print(f"Warning: Could not create convex hull for table {table_id}: {e}")
            continue


def draw_domain_convex_hulls(ax, coords, domains):
    """
    Draw convex hulls around segments from each domain (dark gray dashed, no fill).
    Also add domain name labels near the hulls.
    
    Args:
        ax: Matplotlib axis
        coords: 2D coordinates (N, 2)
        domains: List of domain labels
    """
    print("Drawing convex hulls for each domain...")
    
    # Group coordinates by domain
    domain_coords = {}
    for i, domain in enumerate(domains):
        if domain not in domain_coords:
            domain_coords[domain] = []
        domain_coords[domain].append(coords[i])
    
    # Draw convex hull for each domain
    for domain, coords_list in domain_coords.items():
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
                                     edgecolor=DOMAIN_HULL_COLOR,
                                     linestyle=DOMAIN_HULL_LINESTYLE,
                                     linewidth=DOMAIN_HULL_LINEWIDTH)
            
            ax.add_patch(hull_polygon)
            
            # Add domain name label near the hull
            # Use the centroid of the hull as label position
            centroid = coords_array.mean(axis=0)
            
            # Find a point on the hull boundary that's furthest from centroid for better label placement
            distances = np.linalg.norm(hull_points - centroid, axis=1)
            furthest_idx = np.argmax(distances)
            label_pos = hull_points[furthest_idx]
            
            ax.text(label_pos[0], label_pos[1], domain, 
                   fontsize=14, fontweight='bold',
                   color=DOMAIN_HULL_COLOR,
                   bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor='white', 
                           edgecolor=DOMAIN_HULL_COLOR,
                           alpha=0.8, linestyle=DOMAIN_HULL_LINESTYLE))
            
        except Exception as e:
            print(f"Warning: Could not create convex hull for domain {domain}: {e}")
            continue


def create_cross_domain_plot(coords, entropy_groups, table_ids, domains, method, 
                             title_suffix="", figsize=(6, 5)):
    """
    Create cross-domain visualization plot.
    
    Args:
        coords: 2D coordinates (N, 2)
        entropy_groups: List of entropy group labels
        table_ids: List of table IDs
        domains: List of domain labels
        method: Projection method ('umap' or 'tsne')
        title_suffix: Additional title text
        figsize: Figure size
        
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter to only low and high entropy (exclude mid)
    mask = np.isin(entropy_groups, ['low', 'high'])
    filtered_coords = coords[mask]
    filtered_entropy_groups = np.array(entropy_groups)[mask]
    filtered_table_ids = np.array(table_ids)[mask]
    filtered_domains = np.array(domains)[mask]
    
    # Draw table convex hulls first (so they appear behind points)
    draw_table_convex_hulls(ax, filtered_coords, filtered_table_ids)
    
    # Draw domain convex hulls (before points, so they appear behind)
    draw_domain_convex_hulls(ax, filtered_coords, filtered_domains)
    
    # Plot points for each entropy group
    for entropy_group in ['low', 'high']:
        mask = np.array(filtered_entropy_groups) == entropy_group
        group_coords = filtered_coords[mask]
        
        if len(group_coords) == 0:
            continue
        
        color = ENTROPY_COLORS.get(entropy_group, '#000000')
        alpha = ENTROPY_ALPHAS.get(entropy_group, 0.5)
        
        label = f"{entropy_group.title()} Entropy"
        
        ax.scatter(group_coords[:, 0], group_coords[:, 1], 
                  c=color, marker='o', alpha=alpha, s=50, 
                  label=label, edgecolors='none')
    
    # Customize plot
    ax.set_title(f'{method.upper()} — Cross-Domain Segment Space{title_suffix}', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel(f'{method.upper()} Dimension 1', fontsize=14)
    ax.set_ylabel(f'{method.upper()} Dimension 2', fontsize=14)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Create legend
    entropy_patches = []
    for group in ['low', 'high']:
        color = ENTROPY_COLORS.get(group, '#000000')
        alpha = ENTROPY_ALPHAS.get(group, 0.5)
        patch = mpatches.Patch(color=color, alpha=alpha, label=f'{group.title()} Entropy')
        entropy_patches.append(patch)
    
    if entropy_patches:
        ax.legend(handles=entropy_patches, loc='best')
    
    # Tight layout
    plt.tight_layout()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Visualize segment embeddings from multiple domains in combined embedding space'
    )
    parser.add_argument('--input_movie', type=str, required=True,
                       help='Path to Movie domain segments.json file')
    parser.add_argument('--input_product', type=str, required=True,
                       help='Path to Product domain segments.json file')
    parser.add_argument('--outdir', type=str, required=True,
                       help='Output directory for plots and data')
    parser.add_argument('--model-name', type=str, default='Cross-Domain',
                       help='Model name for titles/filenames')
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
    
    args = parser.parse_args()
    
    print(f"Visualizing cross-domain segment embeddings")
    print(f"Movie domain: {args.input_movie}")
    print(f"Product domain: {args.input_product}")
    print(f"Model: {args.model_name}")
    print(f"Preprocessing: {args.preprocessing}")
    
    # Create output directory
    os.makedirs(args.outdir, exist_ok=True)
    
    # Load data from both domains
    print("\nLoading Movie domain segments...")
    movie_segments, movie_metadata = load_segments_data(args.input_movie)
    
    print("\nLoading Product domain segments...")
    product_segments, product_metadata = load_segments_data(args.input_product)
    
    # Combine segments and add domain labels
    all_segments = []
    for seg in movie_segments:
        seg_copy = seg.copy()
        seg_copy['domain'] = 'Movie'
        all_segments.append(seg_copy)
    
    for seg in product_segments:
        seg_copy = seg.copy()
        seg_copy['domain'] = 'Product'
        all_segments.append(seg_copy)
    
    print(f"\nTotal segments: {len(all_segments)}")
    print(f"  Movie: {len(movie_segments)}")
    print(f"  Product: {len(product_segments)}")
    
    # Extract arrays
    embeddings = np.array([seg['segment_embedding'] for seg in all_segments])
    entropy_groups = [seg['entropy_group'] for seg in all_segments]
    table_ids = [seg['table_id'] for seg in all_segments]
    domains = [seg['domain'] for seg in all_segments]
    
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Entropy groups: {set(entropy_groups)}")
    print(f"Domains: {set(domains)}")
    print(f"Unique tables: {len(set(table_ids))}")
    
    # Filter to only low and high entropy
    mask = np.isin(entropy_groups, ['low', 'high'])
    filtered_embeddings = embeddings[mask]
    filtered_entropy_groups = np.array(entropy_groups)[mask]
    filtered_table_ids = np.array(table_ids)[mask]
    filtered_domains = np.array(domains)[mask]
    
    print(f"\nFiltered segments (low + high entropy): {len(filtered_embeddings)}")
    print(f"  Low entropy: {np.sum(filtered_entropy_groups == 'low')}")
    print(f"  High entropy: {np.sum(filtered_entropy_groups == 'high')}")
    
    # Preprocess embeddings
    print(f"\nPreprocessing embeddings with {args.preprocessing}...")
    processed_embeddings = preprocess_embeddings(filtered_embeddings, method=args.preprocessing)
    
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
    print("\nCreating UMAP plot...")
    model_title = args.model_name
    umap_fig = create_cross_domain_plot(
        umap_coords, filtered_entropy_groups, filtered_table_ids, filtered_domains, 'umap',
        title_suffix=f" ({model_title}, {len(filtered_embeddings)} segments)",
        figsize=(12, 10)
    )
    
    umap_png_path = os.path.join(args.outdir, 'umap_cross_domain.png')
    umap_svg_path = os.path.join(args.outdir, 'umap_cross_domain.svg')
    umap_fig.savefig(umap_png_path, dpi=300, bbox_inches='tight')
    umap_fig.savefig(umap_svg_path, bbox_inches='tight')
    plt.close(umap_fig)
    print(f"✅ UMAP plot saved to: {umap_png_path}")
    
    # Create t-SNE plot
    print("\nCreating t-SNE plot...")
    tsne_fig = create_cross_domain_plot(
        tsne_coords, filtered_entropy_groups, filtered_table_ids, filtered_domains, 'tsne',
        title_suffix=f" ({model_title}, {len(filtered_embeddings)} segments)",
        figsize=(12, 10)
    )
    
    tsne_png_path = os.path.join(args.outdir, 'tsne_cross_domain.png')
    tsne_svg_path = os.path.join(args.outdir, 'tsne_cross_domain.svg')
    tsne_fig.savefig(tsne_png_path, dpi=300, bbox_inches='tight')
    tsne_fig.savefig(tsne_svg_path, bbox_inches='tight')
    plt.close(tsne_fig)
    print(f"✅ t-SNE plot saved to: {tsne_png_path}")
    
    # Save coordinates
    print("\nSaving coordinates to CSV...")
    df = pd.DataFrame({
        'domain': filtered_domains,
        'table_id': filtered_table_ids,
        'entropy_group': filtered_entropy_groups,
        'umap_x': umap_coords[:, 0],
        'umap_y': umap_coords[:, 1],
        'tsne_x': tsne_coords[:, 0],
        'tsne_y': tsne_coords[:, 1],
    })
    
    csv_path = os.path.join(args.outdir, 'coords_cross_domain.csv')
    df.to_csv(csv_path, index=False)
    print(f"✅ Coordinates saved to: {csv_path}")
    
    print("\n✅ Cross-domain visualization complete!")


if __name__ == "__main__":
    main()

