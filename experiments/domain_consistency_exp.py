import argparse
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.cluster import AgglomerativeClustering
import umap.plot
from experiments.experiment_utils import b_cubed_score
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['axes.titlesize'] = 19
plt.rcParams['legend.fontsize'] = 19
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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


def get_actor_related_headers(nested_map: dict) -> list[str]:
    """
    Extracts all actor-related headers that should be semantically identical.
    Returns a list of all headers from actors.collections, actors.names, and actors.given_names.
    """
    actor_headers = []
    
    # Get actor-related headers from the nested structure
    if 'actors' in nested_map:
        actors_section = nested_map['actors']
        
        # Add headers from collections
        if 'collections' in actors_section:
            actor_headers.extend(actors_section['collections'])
        
        # Add headers from names
        if 'names' in actors_section:
            actor_headers.extend(actors_section['names'])
        
        # Add headers from given_names
        if 'given_names' in actors_section:
            actor_headers.extend(actors_section['given_names'])
    
    logging.info(f"Found {len(actor_headers)} actor-related headers: {actor_headers}")
    return actor_headers


def get_embedding_path(artifacts_dir: Path, model_name: str, domain: str, ablation_mode: bool = False) -> Path:
    """
    Determines the correct path for embeddings based on model type.
    """
    if ablation_mode:
        # For ablation models, embeddings are in artifacts_dir/emb/ablation/model_name/domain/
        return artifacts_dir / 'emb' / 'ablation' / model_name / domain / 'header_embeds.parquet'
    else:
        # For regular models, embeddings are in artifacts_dir/emb/model_name/domain/
        return artifacts_dir / 'emb' / model_name / domain / 'header_embeds.parquet'


def create_actor_clustering_visualization(df: pd.DataFrame, mapper, model_name: str, domain: str, artifacts_dir: Path, ablation_mode: bool = False):
    """
    Creates a UMAP visualization focusing on actor-related headers for movie domain.
    """
    # Only run for movie domain
    if 'Movie' not in domain:
        logging.info(f"Skipping actor clustering visualization for non-movie domain: {domain}")
        return
    
    # Load the canonical map to get actor headers
    canonical_path = artifacts_dir / f"canonical_proposals_{domain}.json"
    with open(canonical_path, 'r') as f:
        nested_map = json.load(f)
    
    actor_headers = get_actor_related_headers(nested_map)
    
    # Filter dataframe to only include actor-related headers that exist in the data
    actor_mask = df['header'].isin(actor_headers)
    actor_df = df[actor_mask]
    
    if len(actor_df) == 0:
        logging.warning(f"No actor-related headers found in data for {model_name} on {domain}")
        return
    
    logging.info(f"Creating actor clustering visualization with {len(actor_df)} headers")
    
    # Create a custom color mapping for actor header types
    def get_actor_type(header):
        if header in nested_map.get('actors', {}).get('collections', []):
            return 'actor'
        elif header in nested_map.get('actors', {}).get('names', []):
            return 'actor names'
        elif header in nested_map.get('actors', {}).get('given_names', []):
            return 'actor given names'
        else:
            return 'other'
    
    actor_df = actor_df.copy()
    actor_df['actor_type'] = actor_df['header'].apply(get_actor_type)
    
    # Get indices in the original dataframe for embedding access
    actor_indices = actor_df.index.tolist()
    actor_embeddings = mapper.embedding_[actor_indices]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Create scatter plot with different colors for each actor type
    scatter = sns.scatterplot(
        x=actor_embeddings[:, 0],
        y=actor_embeddings[:, 1],
        hue=actor_df['actor_type'],
        style=actor_df['actor_type'],
        ax=ax,
        legend='full',
        s=100
    )
    
    # Add header labels
    # for i, (idx, row) in enumerate(actor_df.iterrows()):
    #     ax.annotate(row['header'], 
    #                (actor_embeddings[i, 0], actor_embeddings[i, 1]),
    #                xytext=(5, 5), textcoords='offset points',
    #                fontsize=8, alpha=0.7)
    
    domain_display = domain.replace('Quarter_', '').replace('_top100_cleaned', '')
    
    # Create appropriate title and filename
    if ablation_mode:
        plt.title(f"Actor Header Clustering for Ablation Model {model_name} on {domain_display}")
        plot_dir = artifacts_dir / 'plots' / 'ablation'
        plot_dir.mkdir(parents=True, exist_ok=True)
        clean_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        plot_path = plot_dir / f'ablation_{clean_model_name}_{domain_display}_actor_clustering.png'
    else:
        plt.title(f"Actor Header Clustering for {model_name} on {domain_display}")
        plot_dir = artifacts_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / f'{model_name}_{domain_display}_actor_clustering.png'
    
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved actor clustering visualization to {plot_path}")
    plt.close(fig)


def run_clustering_experiment(domain: str, model_name: str, artifacts_dir: Path, ablation_mode: bool = False):
    logging.info(f"Starting Clustering experiment for domain='{domain}', model='{model_name}', ablation_mode={ablation_mode}")
    
    emb_path = get_embedding_path(artifacts_dir, model_name, domain, ablation_mode)
    if not emb_path.exists():
        logging.error(f"Embedding file not found: {emb_path}")
        return

    df = pd.read_parquet(emb_path)
    
    # Load and process the manually curated nested canonical proposals.
    canonical_path = artifacts_dir / f"canonical_proposals_{domain}.json"
    if not canonical_path.exists():
        logging.error(f"Manually curated canonical file not found: {canonical_path}")
        logging.error("Please ensure you have a curated nested JSON for your domain.")
        return
    with open(canonical_path, 'r') as f:
        nested_map = json.load(f)
    
    raw_to_canonical = flatten_nested_canonical_map(nested_map)
    df['canonical'] = df['header'].map(raw_to_canonical)
    
    df = df.dropna(subset=['canonical'])

    if len(df) < 2:
        logging.warning("Not enough data for clustering.")
        return

    X = np.stack(df['embed'].values)
    
    le = LabelEncoder()
    y_true = le.fit_transform(df['canonical'])
    
    n_clusters = len(np.unique(y_true))
    if n_clusters < 2:
        logging.warning("Not enough clusters to evaluate.")
        return

    clustering_model = AgglomerativeClustering(n_clusters=n_clusters)
    y_pred = clustering_model.fit_predict(X)
    
    nmi = normalized_mutual_info_score(y_true, y_pred)
    b3 = b_cubed_score(y_true, y_pred)
    
    silhouette = silhouette_score(X, y_pred)
    calinski = calinski_harabasz_score(X, y_pred)
    davies = davies_bouldin_score(X, y_pred)

    results = {
        'model': model_name,
        'domain': domain,
        'n_clusters': n_clusters,
        'NMI': nmi,
        'B3_F1': b3['f1'],
        'B3_Precision': b3['precision'],
        'B3_Recall': b3['recall'],
        'Silhouette': silhouette,
        'CalinskiHarabasz': calinski,
        'DaviesBouldin': davies
    }

    logging.info(f"Clustering Results for {model_name} on {domain}: {results}")
    
    # Save results with appropriate naming for ablation models
    if ablation_mode:
        output_path = artifacts_dir / f'results_clustering_ablation_{model_name}_{domain}.csv'
    else:
        output_path = artifacts_dir / f'results_clustering_{model_name}_{domain}.csv'
    
    pd.DataFrame([results]).to_csv(output_path, index=False)
    logging.info(f"Saved clustering results to {output_path}")

    # Fit UMAP mapper
    mapper = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42).fit(X)
    
    # 1. Create top 5 canonical groups visualization
    canonical_lengths = get_canonical_group_lengths(nested_map)
    
    # Get unique canonicals present in the data
    available_canonicals = df['canonical'].unique()
    
    # Filter lengths to only include canonicals present in data
    available_lengths = {canonical: canonical_lengths.get(canonical, 0) 
                        for canonical in available_canonicals}
    
    # Sort by length (descending) and get top 5
    top_5_canonicals = sorted(available_lengths.items(), key=lambda x: x[1], reverse=True)[:5]
    top_5_canonicals = [canonical for canonical, length in top_5_canonicals]
    
    logging.info(f"Top 5 canonicals by list length: {top_5_canonicals}")
    logging.info(f"Canonical lengths: {[(c, available_lengths[c]) for c in top_5_canonicals]}")
    
    top_5_mask = df['canonical'].isin(top_5_canonicals)

    embedding_to_plot = mapper.embedding_[top_5_mask]
    labels_to_plot = df.loc[top_5_mask, 'canonical']

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        x=embedding_to_plot[:, 0],
        y=embedding_to_plot[:, 1],
        hue=labels_to_plot,
        ax=ax,
        legend='full'
    )
    domain_display = domain.replace('Quarter_', '').replace('_top100_cleaned', '')
    
    # Create appropriate title and filename based on mode
    if ablation_mode:
        plt.title(f"UMAP of Top 5 Canonicals for Ablation Model {model_name} on {domain_display}")
        plot_dir = artifacts_dir / 'plots' / 'ablation'
        plot_dir.mkdir(parents=True, exist_ok=True)
        # Clean up model name for filename (remove special characters)
        clean_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        plot_path = plot_dir / f'ablation_{clean_model_name}_{domain_display}_umap_top5.png'
    else:
        plt.title(f"UMAP of Top 5 Canonicals for {model_name} on {domain_display}")
        plot_dir = artifacts_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / f'{model_name}_{domain_display}_umap_top5.png'
    
    plt.savefig(plot_path)
    logging.info(f"Saved UMAP plot of top 5 canonicals to {plot_path}")
    plt.close(fig)
    
    # 2. Create actor-related headers clustering visualization (only for movie domain)
    create_actor_clustering_visualization(df, mapper, model_name, domain, artifacts_dir, ablation_mode)


def main():
    parser = argparse.ArgumentParser(description="Run lexical consistency experiments.")
    parser.add_argument('--artifacts_dir', type=Path, default='artifacts/lexvar', help="Directory to save artifacts.")
    parser.add_argument('--domains', type=str, nargs='+', default=['Quarter_Movie_top100_cleaned', 'Quarter_Product_top100_cleaned'], help="List of domains to process.")
    parser.add_argument('--models', type=str, nargs='+', default=['bert', 'tapas', 'haetae', 'navi'], help="Models to run.")
    parser.add_argument('--ablation_mode', action='store_true', help="Enable ablation mode for complex model names.")
    parser.add_argument('--ablation_models', type=str, nargs='+', help="List of ablation model names to run (only used when --ablation_mode is set).")
    
    args = parser.parse_args()

    if args.ablation_mode:
        if not args.ablation_models:
            logging.error("--ablation_models must be specified when --ablation_mode is set")
            return
        
        logging.info(f"Running in ablation mode with models: {args.ablation_models}")
        for domain in args.domains:
            for model_name in args.ablation_models:
                run_clustering_experiment(domain, model_name, args.artifacts_dir, ablation_mode=True)
    else:
        logging.info(f"Running in regular mode with models: {args.models}")
        for domain in args.domains:
            for model_name in args.models:
                run_clustering_experiment(domain, model_name, args.artifacts_dir, ablation_mode=False)
    

if __name__ == '__main__':
    main()