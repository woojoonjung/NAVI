import argparse
import json
import logging
from collections import Counter
from pathlib import Path
import re
import string
from itertools import combinations
import torch
import numpy as np
from tqdm import tqdm
from unidecode import unidecode
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.metrics.distance import edit_distance
import jaro
import networkx as nx
from sklearn.preprocessing import normalize as l2_normalize
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertForMaskedLM, BertTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
lemmatizer = WordNetLemmatizer()

SYNONYM_MAP = {
    'publisher': ['distributor'],
    'actors': ['cast', 'starring'],
    'runtime': ['duration'],
    'date_published': ['release_date'],
}


def extract_header_frequencies(domain_path: Path) -> Counter:
    """
    Extracts header strings and their frequencies from all tables in a domain.
    """
    header_freqs = Counter()
    jsonl_files = sorted([p for p in domain_path.glob('*.jsonl')])
    logging.info(f"Found {len(jsonl_files)} files in {domain_path}")

    for file_path in tqdm(jsonl_files, desc=f"Processing {domain_path.name}"):
        if file_path.stat().st_size == 0:
            logging.warning(f"Skipping empty file: {file_path}")
            continue
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    row = json.loads(line)
                    header_freqs.update(row.keys())
                except json.JSONDecodeError:
                    logging.warning(f"Skipping malformed line in {file_path}")
    return header_freqs


def normalize_header(s: str, lemmatize: bool = True) -> str:
    """
    Normalizes a header string.
    """
    s = s.lower().strip()
    s = unidecode(s)
    s = re.sub(r'\[.*?\]', '', s)
    s = re.sub(r'[\s_.-]+', '_', s)
    s = s.translate(str.maketrans('', '', string.punctuation.replace('_', '')))
    if lemmatize:
        tokens = word_tokenize(s.replace('_', ' '))
        s = '_'.join([lemmatizer.lemmatize(token) for token in tokens])
    s = s.strip('_')
    return s


def jaccard_similarity(s1, s2):
    """
    Computes Jaccard similarity between two normalized header strings.
    """
    s1_tokens = set(s1.split('_'))
    s2_tokens = set(s2.split('_'))
    return len(s1_tokens.intersection(s2_tokens)) / len(s1_tokens.union(s2_tokens))


def get_header_embeddings_for_clustering(headers: list[str], model, tokenizer, batch_size: int = 32, domain: str = None) -> dict[str, np.ndarray]:
    """
    Generates non-contextualized embeddings for a list of header strings using a base BERT model.
    `model` should be a `BertModel` instance.
    If the domain contains 'movie', a weighted average is used for pooling to give more credit to the first word.
    """
    header_embeddings = {}
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(headers), batch_size), desc="Generating header embeddings"):
            batch_headers = headers[i:i+batch_size]
            inputs = tokenizer(
                [h.replace('_', ' ') for h in batch_headers], 
                return_tensors='pt', 
                padding=True, 
                truncation=True, 
                max_length=32
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            
            last_hidden = outputs.last_hidden_state
            is_movie_domain = domain is not None and 'movie' in domain.lower()

            if is_movie_domain:
                weight = 0.7 
                input_ids = inputs['input_ids']
                attention_mask = inputs['attention_mask']
                final_embeds = []

                for i, header in enumerate(batch_headers):
                    header_hidden_states = last_hidden[i]
                    first_word = header.split('_')[0]
                    num_first_word_tokens = len(tokenizer.tokenize(first_word))
                    actual_tokens_mask = attention_mask[i].bool()
                    actual_tokens_mask[0] = False
                    sep_token_index = (input_ids[i] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
                    if len(sep_token_index) > 0:
                        actual_tokens_mask[sep_token_index[0]] = False

                    if not actual_tokens_mask.any():
                        final_embeds.append(torch.zeros_like(header_hidden_states[0]))
                        continue

                    first_word_indices = torch.zeros_like(actual_tokens_mask)
                    end_index = min(1 + num_first_word_tokens, len(first_word_indices))
                    first_word_indices[1:end_index] = True
                    first_word_mask = first_word_indices & actual_tokens_mask
                    rest_mask = (~first_word_indices) & actual_tokens_mask

                    first_word_sum = torch.sum(header_hidden_states * first_word_mask.unsqueeze(-1), dim=0)
                    first_word_count = first_word_mask.sum()
                    rest_sum = torch.sum(header_hidden_states * rest_mask.unsqueeze(-1), dim=0)
                    rest_count = rest_mask.sum()

                    if first_word_count > 0 and rest_count > 0:
                        first_word_mean = first_word_sum / first_word_count
                        rest_mean = rest_sum / rest_count
                        weighted_avg = weight * first_word_mean + (1 - weight) * rest_mean
                    elif first_word_count > 0:
                        weighted_avg = first_word_sum / first_word_count
                    else:
                        weighted_avg = torch.sum(header_hidden_states * actual_tokens_mask.unsqueeze(-1), dim=0) / actual_tokens_mask.sum()
                    
                    final_embeds.append(weighted_avg)
                
                mean_pooled_embeds = torch.stack(final_embeds)
            else:
                attention_mask = inputs['attention_mask']
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeds = torch.sum(last_hidden * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                mean_pooled_embeds = sum_embeds / sum_mask
            
            normalized_embeds = l2_normalize(mean_pooled_embeds.cpu().numpy())
            for header, embed in zip(batch_headers, normalized_embeds):
                header_embeddings[header] = embed
    return header_embeddings


def enhanced_header_similarity(h1: str, h2: str) -> bool:
    """
    A more discerning similarity logic for headers.
    """
    if '_' in h1 or '_' in h2:
        h1_parts = h1.split('_')
        h2_parts = h2.split('_')
        base1, attrs1 = h1_parts[0], set([p for p in h1_parts[1:] if not p.isdigit()])
        base2, attrs2 = h2_parts[0], set([p for p in h2_parts[1:] if not p.isdigit()])
        if base1 == base2:
            return True
        if jaro.jaro_winkler_metric(base1, base2) >= 0.95:
            if attrs1 and attrs2:
                attr_jaccard = len(attrs1.intersection(attrs2)) / len(attrs1.union(attrs2))
                if attr_jaccard >= 0.5:
                    return True
            elif not attrs1 and not attrs2:
                return True
        return False
    else:
        if edit_distance(h1, h2) / (len(h1) + len(h2)) <= 0.15:
            return True
        if jaro.jaro_winkler_metric(h1, h2) >= 0.95:
            return True
    return False


def cluster_headers(normalized_headers: list[str], raw_header_freqs: Counter, raw_to_norm: dict[str, str], use_embeddings: bool = False, header_embeddings: dict = None, sim_threshold: float = 0.85) -> list[list[str]]:
    """
    Clusters normalized headers based on similarity metrics.
    """
    G = nx.Graph()
    G.add_nodes_from(normalized_headers)

    if use_embeddings:
        if header_embeddings is None:
            raise ValueError("header_embeddings must be provided when use_embeddings is True")
        header_list = normalized_headers
        embedding_matrix = np.array([header_embeddings[h] for h in header_list])
        logging.info("Computing cosine similarity matrix...")
        sim_matrix = cosine_similarity(embedding_matrix)
        for i, j in tqdm(combinations(range(len(header_list)), 2), desc="Building header graph from embeddings"):
            if sim_matrix[i, j] > sim_threshold:
                G.add_edge(header_list[i], header_list[j])
    else:
        for h1, h2 in tqdm(combinations(normalized_headers, 2), desc="Building header graph"):
            if enhanced_header_similarity(h1, h2):
                G.add_edge(h1, h2)

    for h1, synonyms in SYNONYM_MAP.items():
        if h1 in G:
            for s in synonyms:
                if s in G and not G.has_edge(h1, s):
                    G.add_edge(h1, s)

    clusters = list(nx.connected_components(G))
    norm_to_raw = {}
    for r, n in raw_to_norm.items():
        if n not in norm_to_raw:
            norm_to_raw[n] = []
        norm_to_raw[n].append(r)
    
    raw_clusters = [[header for norm_h in cluster for header in norm_to_raw.get(norm_h, [])] for cluster in clusters]
    
    def get_cluster_freq(cluster):
        return sum(raw_header_freqs.get(h, 0) for h in cluster)

    ranked_clusters = sorted(raw_clusters, key=get_cluster_freq, reverse=True)
    return ranked_clusters


def interactive_curation(proposals_path: Path, final_path: Path, top_n: int = 50):
    """
    A simple CLI tool to manually curate canonical header clusters.
    """
    with open(proposals_path, 'r') as f:
        proposals = json.load(f)

    final_canonical_classes = {}
    
    print("--- Starting Interactive Canonical Class Curation ---")
    print(f"Loaded {len(proposals)} proposed clusters from {proposals_path}")
    print(f"Reviewing top {top_n} clusters.")
    print("Commands: (a)ccept, (r)eject, (m)erge, (e)dit, (s)kip, (q)uit")

    for i, cluster in enumerate(proposals[:top_n]):
        print(f"\n--- Cluster {i+1}/{top_n} ---")
        print(f"Members: {cluster}")
        
        action = input("Action: ").lower()

        if action in ['a', 'accept']:
            name = input("Enter canonical name: ")
            final_canonical_classes[name] = sorted(list(set(cluster)))
        elif action in ['r', 'reject']:
            print("Cluster rejected.")
        elif action in ['s', 'skip']:
            print("Cluster skipped.")
        elif action in ['m', 'merge']:
            print("Current accepted classes:")
            for idx, name in enumerate(final_canonical_classes.keys()):
                print(f"  {idx}: {name}")
            try:
                target_idx = int(input("Merge with class index: "))
                target_name = list(final_canonical_classes.keys())[target_idx]
                final_canonical_classes[target_name].extend(cluster)
                final_canonical_classes[target_name] = sorted(list(set(final_canonical_classes[target_name])))
                print(f"Merged into '{target_name}'.")
            except (ValueError, IndexError):
                print("Invalid index.")
        elif action in ['e', 'edit']:
            print("Enter comma-separated headers to keep:")
            to_keep_str = input()
            to_keep = [h.strip() for h in to_keep_str.split(',')]
            name = input("Enter canonical name for edited list: ")
            final_canonical_classes[name] = sorted(list(set(to_keep)))
        elif action in ['q', 'quit']:
            break
        else:
            print("Unknown command.")

    with open(final_path, 'w') as f:
        json.dump(final_canonical_classes, f, indent=2, sort_keys=True)
    
    print(f"\n--- Curation Complete ---")
    print(f"Saved {len(final_canonical_classes)} canonical classes to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate and curate canonical header sets.")
    parser.add_argument('--data_dir', type=Path, default='data', help="Directory containing the domain data.")
    parser.add_argument('--artifacts_dir', type=Path, default='artifacts/lexvar', help="Directory to save artifacts.")
    parser.add_argument('--domains', type=str, nargs='+', default=['Quarter_Movie_top100_cleaned', 'Quarter_Product_top100_cleaned'], help="List of domains to process.")
    parser.add_argument('--run_curation', action='store_true', help="Run the interactive curation tool.")
    parser.add_argument('--cluster_method', type=str, default='semantic', choices=['lexical', 'semantic'], help="Clustering method for canonical proposals.")
    
    args = parser.parse_args()
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)

    if args.run_curation:
        for domain in args.domains:
            proposals_path = args.artifacts_dir / f"canonical_proposals_{domain}.json"
            final_path = args.artifacts_dir / f"canonical_final_{domain}.json"
            if not proposals_path.exists():
                print(f"Proposals file not found for {domain}. Please run the script without --run_curation first.")
                continue
            interactive_curation(proposals_path, final_path)
        return

    for domain in args.domains:
        logging.info(f"Processing domain: {domain}")
        domain_path = args.data_dir / domain
        
        header_freqs = extract_header_frequencies(domain_path)
        output_path = args.artifacts_dir / f"header_vocab_{domain}.json"
        with open(output_path, 'w') as f:
            json.dump(header_freqs, f, indent=2, sort_keys=True)
        logging.info(f"Saved header vocabulary for {domain} to {output_path}")

        raw_headers = list(header_freqs.keys())
        normalized_map = {h: normalize_header(h) for h in tqdm(raw_headers, desc="Normalizing headers")}
        output_path_norm = args.artifacts_dir / f"normalized_map_{domain}.json"
        with open(output_path_norm, 'w') as f:
            json.dump(normalized_map, f, indent=2, sort_keys=True)
        logging.info(f"Saved normalized header map for {domain} to {output_path_norm}")

        unique_normalized_headers = sorted(list(set(normalized_map.values())))
        
        raw_header_freqs = Counter()
        for file_path in (args.data_dir / domain).glob('*.jsonl'):
            if file_path.stat().st_size == 0: continue
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        row = json.loads(line)
                        raw_header_freqs.update(row.keys())
                    except json.JSONDecodeError:
                        pass

        if args.cluster_method == 'semantic':
            logging.info("Generating semantic embeddings for headers for clustering...")
            bert_for_clustering_mlm = BertForMaskedLM.from_pretrained('bert-base-uncased').to(device)
            tokenizer_for_clustering = BertTokenizer.from_pretrained('bert-base-uncased')
            header_embeddings = get_header_embeddings_for_clustering(
                unique_normalized_headers, bert_for_clustering_mlm.bert, tokenizer_for_clustering, domain=domain
            )
            proposals = cluster_headers(
                unique_normalized_headers, raw_header_freqs, normalized_map, 
                use_embeddings=True, header_embeddings=header_embeddings
            )
        else:
            proposals = cluster_headers(unique_normalized_headers, raw_header_freqs, normalized_map)

        output_path_proposals = args.artifacts_dir / f"canonical_proposals_{domain}.json"
        with open(output_path_proposals, 'w') as f:
            json.dump(proposals, f, indent=2)
        logging.info(f"Saved canonical proposals for {domain} to {output_path_proposals}")


if __name__ == '__main__':
    main()