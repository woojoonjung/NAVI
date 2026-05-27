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
import umap
from experiments.header_clustering.clt_utils import b_cubed_score, flatten_nested_canonical_map, get_canonical_group_lengths, get_embedding_path
from experiments.downstream_tasks.clt_utils import (
    centroid_cosine_distance,
    mean_pairwise_centroid_cosine_dist,
    mean_pairwise_centroid_l2,
)
import matplotlib.pyplot as plt

plt.rcParams['axes.titlesize'] = 20
plt.rcParams['legend.fontsize'] = 19
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')




def get_actor_type(header: str, nested_map: dict) -> str:
    """Figure 3 legend groups: collections -> actor, names -> actor names, given_names -> actor given names."""
    actors = nested_map.get('actors', {})
    if header in actors.get('collections', []):
        return 'actor'
    if header in actors.get('names', []):
        return 'actor names'
    if header in actors.get('given_names', []):
        return 'actor given names'
    return 'other'


def actor_fig3_metrics_placeholder() -> dict:
    """Column-consistent dict when Movie actor metrics are not computed."""
    return {
        'ActorFig3_InterCentroid_mean_L2': np.nan,
        'ActorFig3_InterCentroid_mean_cosine_dist': np.nan,
        'ActorFig3_n_actor': 0,
        'ActorFig3_n_actor_names': 0,
        'ActorFig3_n_actor_given_names': 0,
        'ActorFig3_L2_actor_vs_actor_names': np.nan,
        'ActorFig3_L2_actor_vs_actor_given_names': np.nan,
        'ActorFig3_L2_actor_names_vs_actor_given_names': np.nan,
        'ActorFig3_cos_actor_vs_actor_names': np.nan,
        'ActorFig3_cos_actor_vs_actor_given_names': np.nan,
        'ActorFig3_cos_actor_names_vs_actor_given_names': np.nan,
    }


def compute_actor_fig3_metrics(df: pd.DataFrame, nested_map: dict, domain: str) -> dict:
    """
    Inter-group centroid distances in original embedding space for Figure 3 actor groups
    (same subset/labels as actor UMAP plot). Non-Movie domains return NaN/zeros.
    """
    if 'Movie' not in domain:
        return actor_fig3_metrics_placeholder()

    actor_headers = get_actor_related_headers(nested_map)
    actor_mask = df['header'].isin(actor_headers)
    actor_df = df.loc[actor_mask].copy()
    out = actor_fig3_metrics_placeholder()
    if len(actor_df) == 0:
        return out

    actor_df['actor_type'] = actor_df['header'].apply(lambda h: get_actor_type(h, nested_map))
    actor_df = actor_df[actor_df['actor_type'] != 'other']

    out['ActorFig3_n_actor'] = int((actor_df['actor_type'] == 'actor').sum())
    out['ActorFig3_n_actor_names'] = int((actor_df['actor_type'] == 'actor names').sum())
    out['ActorFig3_n_actor_given_names'] = int((actor_df['actor_type'] == 'actor given names').sum())

    if len(actor_df) < 2 or actor_df['actor_type'].nunique() < 2:
        return out

    X_actor = np.stack(actor_df['embed'].values)
    le = LabelEncoder()
    y = le.fit_transform(actor_df['actor_type'])
    mp = mean_pairwise_centroid_l2(X_actor, y, le.classes_)
    out['ActorFig3_InterCentroid_mean_L2'] = mp['mean_l2']
    mpc = mean_pairwise_centroid_cosine_dist(X_actor, y, le.classes_)
    out['ActorFig3_InterCentroid_mean_cosine_dist'] = mpc['mean_cosine_dist']

    def _centroids(a: str, b: str) -> tuple[np.ndarray, np.ndarray] | None:
        if a not in le.classes_ or b not in le.classes_:
            return None
        code_a = le.transform([a])[0]
        code_b = le.transform([b])[0]
        cta = X_actor[y == code_a].mean(axis=0)
        ctb = X_actor[y == code_b].mean(axis=0)
        return cta, ctb

    def _pair_l2(a: str, b: str) -> float:
        cc = _centroids(a, b)
        if cc is None:
            return float(np.nan)
        cta, ctb = cc
        return float(np.linalg.norm(cta - ctb))

    def _pair_cos(a: str, b: str) -> float:
        cc = _centroids(a, b)
        if cc is None:
            return float(np.nan)
        return centroid_cosine_distance(cc[0], cc[1])

    out['ActorFig3_L2_actor_vs_actor_names'] = _pair_l2('actor', 'actor names')
    out['ActorFig3_L2_actor_vs_actor_given_names'] = _pair_l2('actor', 'actor given names')
    out['ActorFig3_L2_actor_names_vs_actor_given_names'] = _pair_l2('actor names', 'actor given names')
    out['ActorFig3_cos_actor_vs_actor_names'] = _pair_cos('actor', 'actor names')
    out['ActorFig3_cos_actor_vs_actor_given_names'] = _pair_cos('actor', 'actor given names')
    out['ActorFig3_cos_actor_names_vs_actor_given_names'] = _pair_cos('actor names', 'actor given names')
    return out


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


def create_actor_clustering_visualization(df: pd.DataFrame, mapper, model_name: str, domain: str, artifacts_dir: Path, ablation_mode: bool = False):
    """
    Creates a UMAP visualization focusing on actor-related headers for movie domain.
    """
    # Only run for movie domain
    if 'Movie' not in domain:
        logging.info(f"Skipping actor clustering visualization for non-movie domain: {domain}")
        return
    
    # Load the human-curated canonical sets file
    domain_artifact = domain.split('/')[-1].lower()  # 'cleaned/Movie' -> 'movie'
    canonical_path = artifacts_dir / f"canonical_sets_{domain_artifact}.json"
    if not canonical_path.exists():
        logging.error(f"Canonical sets file not found at {canonical_path}. Please ensure the curated canonical sets file exists.")
        return
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
    
    actor_df = actor_df.copy()
    actor_df['actor_type'] = actor_df['header'].apply(lambda h: get_actor_type(h, nested_map))
    
    # Get indices in the original dataframe for embedding access
    actor_indices = actor_df.index.tolist()
    actor_embeddings = mapper.embedding_[actor_indices]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Create scatter plot with different colors for each actor type
    actor_types = actor_df['actor_type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(actor_types)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for i, actor_type in enumerate(actor_types):
        mask = actor_df['actor_type'] == actor_type
        ax.scatter(
            actor_embeddings[mask, 0],
            actor_embeddings[mask, 1],
            c=colors[i],
            marker=markers[i % len(markers)],
            label=actor_type,
            s=100
        )
    
    ax.legend(loc='best')
    
    # Add header labels
    # for i, (idx, row) in enumerate(actor_df.iterrows()):
    #     ax.annotate(row['header'], 
    #                (actor_embeddings[i, 0], actor_embeddings[i, 1]),
    #                xytext=(5, 5), textcoords='offset points',
    #                fontsize=8, alpha=0.7)
    
    domain_display = domain.replace('Quarter_', '').replace('_top100_cleaned', '')
    # Sanitize domain_display for use in filenames
    clean_domain_display = domain_display.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    # Create appropriate title and filename
    if ablation_mode:
        plt.title(f"Actor Header Clustering for Ablation Model {model_name} on {domain_display}")
        plot_dir = artifacts_dir / 'plots' / 'ablation'
        plot_dir.mkdir(parents=True, exist_ok=True)
        clean_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        plot_path = plot_dir / f'ablation_{clean_model_name}_{clean_domain_display}_actor_clustering.png'
    else:
        plt.title(f"Actor Header Clustering for {model_name} on {domain_display}")
        plot_dir = artifacts_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / f'{model_name}_{clean_domain_display}_actor_clustering.png'
    
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved actor clustering visualization to {plot_path}")
    plt.close(fig)


def compute_intra_group_distances(df: pd.DataFrame, nested_map: dict) -> dict:
    """
    Average pairwise L2 and cosine distance within each canonical group across all canonical sets.

    For each canonical group with ≥2 members present in df, compute all pairwise distances,
    average within the group, then macro-average across groups.
    Returns keys: mean_intra_L2, mean_intra_cosine_dist.
    """
    from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

    raw_to_canonical = flatten_nested_canonical_map(nested_map)
    df = df.copy()
    df["canonical"] = df["header"].map(raw_to_canonical)
    df = df.dropna(subset=["canonical"])

    group_l2: list[float] = []
    group_cos: list[float] = []

    for canonical, group_df in df.groupby("canonical"):
        if len(group_df) < 2:
            continue
        X = np.stack(group_df["embed"].values)
        l2_mat = euclidean_distances(X)
        cos_mat = cosine_distances(X)
        n = len(X)
        triu_idx = np.triu_indices(n, k=1)
        group_l2.append(float(l2_mat[triu_idx].mean()))
        group_cos.append(float(cos_mat[triu_idx].mean()))

    if not group_l2:
        return {"mean_intra_L2": float("nan"), "mean_intra_cosine_dist": float("nan")}

    return {
        "mean_intra_L2": float(np.mean(group_l2)),
        "mean_intra_cosine_dist": float(np.mean(group_cos)),
    }


def run_clustering_experiment(domain: str, model_name: str, artifacts_dir: Path, ablation_mode: bool = False):
    logging.info(f"Starting Clustering experiment for domain='{domain}', model='{model_name}', ablation_mode={ablation_mode}")
    
    emb_path = get_embedding_path(artifacts_dir, model_name, domain, ablation_mode)
    if not emb_path.exists():
        logging.error(f"Embedding file not found: {emb_path}")
        return

    df = pd.read_parquet(emb_path)
    
    # Load and process the human-curated canonical sets file
    domain_artifact = domain.split('/')[-1].lower()  # 'cleaned/Movie' -> 'movie'
    canonical_path = artifacts_dir / f"canonical_sets_{domain_artifact}.json"
    if not canonical_path.exists():
        logging.error(f"Canonical sets file not found at {canonical_path}. Please ensure the curated canonical sets file exists.")
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
    results.update(compute_actor_fig3_metrics(df, nested_map, domain))
    results.update(compute_intra_group_distances(df, nested_map))

    logging.info(f"Clustering Results for {model_name} on {domain}: {results}")
    
    # Save results with appropriate naming for ablation models
    # Sanitize domain name to avoid path issues (replace / with _)
    clean_domain = domain.replace('/', '_').replace('\\', '_').replace(':', '_')
    if ablation_mode:
        output_path = artifacts_dir / f'results_clustering_ablation_{model_name}_{clean_domain}.csv'
    else:
        output_path = artifacts_dir / f'results_clustering_{model_name}_{clean_domain}.csv'
    
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
    
    # Create scatter plot with different colors for each canonical group
    unique_labels = labels_to_plot.unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels_to_plot == label
        ax.scatter(
            embedding_to_plot[mask, 0],
            embedding_to_plot[mask, 1],
            c=colors[i],
            label=label,
            s=50,
            alpha=0.7
        )
    
    ax.legend(loc='best')
    domain_display = domain.replace('Quarter_', '').replace('_top100_cleaned', '')
    # Sanitize domain_display for use in filenames
    clean_domain_display = domain_display.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    # Create appropriate title and filename based on mode
    if ablation_mode:
        plt.title(f"UMAP of Top 5 Canonicals for Ablation Model {model_name} on {domain_display}")
        plot_dir = artifacts_dir / 'plots' / 'ablation'
        plot_dir.mkdir(parents=True, exist_ok=True)
        # Clean up model name for filename (remove special characters)
        clean_model_name = model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        plot_path = plot_dir / f'ablation_{clean_model_name}_{clean_domain_display}_umap_top5.png'
    else:
        plt.title(f"UMAP of Top 5 Canonicals for {model_name} on {domain_display}")
        plot_dir = artifacts_dir / 'plots'
        plot_dir.mkdir(exist_ok=True)
        plot_path = plot_dir / f'{model_name}_{clean_domain_display}_umap_top5.png'
    
    plt.savefig(plot_path)
    logging.info(f"Saved UMAP plot of top 5 canonicals to {plot_path}")
    plt.close(fig)
    
    # 2. Create actor-related headers clustering visualization (only for movie domain)
    create_actor_clustering_visualization(df, mapper, model_name, domain, artifacts_dir, ablation_mode)


def main():
    parser = argparse.ArgumentParser(
        description="Run lexical consistency experiments (header clustering vs canonical groups). "
        "For BERT vs NAVI on Figure 3 actor metrics, e.g.: "
        "--domains cleaned/Movie --models bert navi",
    )
    parser.add_argument('--artifacts_dir', type=Path, default='artifacts/lexvar', help="Directory to save artifacts.")
    parser.add_argument('--domains', type=str, nargs='+', default=['cleaned/Movie', 'cleaned/Product'], help="List of domains to process.")
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