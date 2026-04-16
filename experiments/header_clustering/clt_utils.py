"""
Utility functions for header clustering experiments.
"""

import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import umap


def extract_header_value_embeddings_from_positions(embeddings, header_positions, value_positions):
    """Extracts header and value embeddings from contextualized embeddings."""
    hidden_size = embeddings.size(-1)
    embeddings = embeddings.squeeze(0)
    
    header_pos_dict = header_positions[0] if isinstance(header_positions, list) else header_positions
    value_pos_dict = value_positions[0] if isinstance(value_positions, list) else value_positions

    header_keys = list(header_pos_dict.keys())
    num_headers = len(header_keys)
    
    header_embeds = torch.zeros((1, num_headers, hidden_size), device=embeddings.device)
    val_embeds = torch.zeros((1, num_headers, hidden_size), device=embeddings.device)
    
    for i, header_name in enumerate(header_keys):
        h_token_indices = header_pos_dict.get(header_name, [])
        if h_token_indices:
            header_tokens = embeddings[h_token_indices]
            header_embeds[0, i] = header_tokens.mean(dim=0)
            
        v_token_indices = value_pos_dict.get(header_name, [])
        if v_token_indices:
            value_tokens = embeddings[v_token_indices]
            val_embeds[0, i] = value_tokens.mean(dim=0)
            
    return header_embeds, val_embeds


def b_cubed_score(y_true, y_pred):
    """
    Calculate B-cubed precision, recall, and F1 score for clustering evaluation.
    
    Args:
        y_true: Ground truth cluster labels
        y_pred: Predicted cluster labels
        
    Returns:
        dict: Dictionary containing precision, recall, and f1 scores
    """
    # Convert to numpy arrays if not already
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Get unique labels
    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    
    # Calculate B-cubed precision and recall
    precision_sum = 0
    recall_sum = 0
    total_items = len(y_true)
    
    for i in range(total_items):
        # Get the true and predicted clusters for this item
        true_cluster = y_true[i]
        pred_cluster = y_pred[i]
        
        # Count items in the same true cluster
        true_cluster_size = np.sum(y_true == true_cluster)
        # Count items in the same predicted cluster
        pred_cluster_size = np.sum(y_pred == pred_cluster)
        # Count items in both clusters (intersection)
        intersection = np.sum((y_true == true_cluster) & (y_pred == pred_cluster))
        
        # B-cubed precision for this item
        if pred_cluster_size > 0:
            precision_sum += intersection / pred_cluster_size
        
        # B-cubed recall for this item
        if true_cluster_size > 0:
            recall_sum += intersection / true_cluster_size
    
    # Average precision and recall
    precision = precision_sum / total_items
    recall = recall_sum / total_items
    
    # Calculate F1 score
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def flatten_nested_canonical_map(nested_map: dict) -> dict[str, str]:
    """
    Flattens a nested dictionary of canonical classes into a flat map
    from raw header to a dot-separated canonical path.
    """
    raw_to_canonical = {}

    def recurse(d, path_prefix):
        for key, value in d.items():
            current_path = f"{path_prefix}.{key}" if path_prefix else key
            if isinstance(value, list):
                for header in value:
                    raw_to_canonical[header] = current_path
            elif isinstance(value, dict):
                recurse(value, current_path)

    recurse(nested_map, '')
    return raw_to_canonical


def get_canonical_group_lengths(nested_map: dict) -> dict[str, int]:
    """
    Calculates the length (number of elements) in each canonical group's list.
    Returns a dictionary mapping canonical path to its list length.
    """
    canonical_lengths = {}

    def recurse(d, path_prefix):
        for key, value in d.items():
            current_path = f"{path_prefix}.{key}" if path_prefix else key
            if isinstance(value, list):
                canonical_lengths[current_path] = len(value)
            elif isinstance(value, dict):
                recurse(value, current_path)

    recurse(nested_map, '')
    return canonical_lengths


def get_embedding_path(artifacts_dir: Path, model_name: str, domain: str, ablation_mode: bool = False) -> Path:
    """
    Determines the correct path for embeddings based on model type.
    """
    if ablation_mode:
        # For ablation models, embeddings are in artifacts_dir/emb/ablation/navi_{domain}_{model_name}/domain/
        domain_artifact = domain.split('/')[-1].lower()  # 'cleaned/Movie' -> 'movie'
        return artifacts_dir / 'emb' / 'ablation' / model_name / domain_artifact / 'header_embeds.parquet'
    else:
        # For regular models, embeddings are in artifacts_dir/emb/model_name/domain/
        domain_artifact = domain.split('/')[-1].lower()  # 'cleaned/Movie' -> 'movie'
        return artifacts_dir / 'emb' / model_name / domain_artifact / 'header_embeds.parquet'


def create_actor_clustering_visualization(df, mapper, model_name, domain, artifacts_dir, ablation_mode):
    """
    Creates a visualization showing actor-related headers clustering.
    Only for movie domain.
    """
    domain_artifact = domain.split('/')[-1].lower()
    if domain_artifact != 'movie':
        return
    
    # Get actor-related headers from canonical map
    # This assumes the canonical map is available - we'll need to load it
    # For now, we'll use a simple heuristic: headers containing 'actor' or 'cast'
    actor_headers = df[df['header'].str.contains('actor|cast', case=False, na=False)]['header'].unique()
    
    if len(actor_headers) == 0:
        return
    
    actor_mask = df['header'].isin(actor_headers)
    actor_embeddings = mapper.embedding_[actor_mask]
    actor_labels = df.loc[actor_mask, 'header']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(actor_embeddings[:, 0], actor_embeddings[:, 1], c='blue', s=50, alpha=0.6)
    
    for i, label in enumerate(actor_labels):
        ax.annotate(label, (actor_embeddings[i, 0], actor_embeddings[i, 1]), 
                   fontsize=8, alpha=0.7)
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(f'Actor-Related Headers Clustering for {model_name}')
    plt.tight_layout()
    
    # Save plot
    domain_display = domain.replace('Quarter_', '').replace('_top100_cleaned', '')
    clean_domain_display = domain_display.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    if ablation_mode:
        plot_dir = artifacts_dir / 'plots' / 'ablation'
        plot_dir.mkdir(parents=True, exist_ok=True)
        clean_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        plot_path = plot_dir / f'ablation_{clean_model_name}_{clean_domain_display}_actor_clustering.png'
    else:
        plot_dir = artifacts_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / f'{model_name}_{clean_domain_display}_actor_clustering.png'
    
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
