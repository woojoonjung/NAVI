import json
import pandas as pd
from io import StringIO
import torch
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import f1_score

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, normalized_mutual_info_score

# === NAVI ===
from model.navi import NaviForMaskedLM

# === HAETAE ===
from baselines.haetae.model import HAETAE

# === TAPAS ===
from transformers import TapasForMaskedLM, BertForMaskedLM

# === TABBIE ===


############################ LOAD DATA ###################################

def load_data(path, path_is="jsonl"):
    data = []

    if path_is == "csv":
        if path is not None:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            table = pd.read_csv(StringIO(''.join(lines)))
            if "class" in table.columns:
                table.drop(columns=["class"], inplace=True)  # Drop 'class' column
            for _, row in table.iterrows():
                row_json = row.to_dict() 
                data.append(row_json)

    elif path_is == "jsonl":
        if path is not None:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))

    return data

###################### MASKED PREDICTION EXPERIMENT #########################

def mask_entry(example, tokenizer, collator):
    """
    Returns a batch with masked input, labels, and token-level representations.
    """
    batch = collator([example])

    # Decode tokens for visualization
    masked_input_tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"][0])
    label_tokens = tokenizer.convert_ids_to_tokens(batch["labels"][0])

    return batch, masked_input_tokens, label_tokens

def predict_masked_tokens(model, tokenizer, batch):
    """
    Runs the model and predicts masked tokens in the input.
    """
    device = next(model.parameters()).device

    # === ATLAS ===
    if isinstance(model, NaviForMaskedLM):
        input_kwargs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "position_ids": batch["position_ids"].to(device),
            "segment_ids": batch["segment_ids"].to(device) if batch.get("segment_ids") is not None else None,
            "header_strings": batch.get("header_strings"),
        }
        with torch.no_grad():
            outputs = model(**input_kwargs)
        logits = outputs[1]

    # === HAETAE ===
    elif isinstance(model, HAETAE):
        input_kwargs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "key_positions": batch.get("key_positions")
        }
        with torch.no_grad():
            outputs = model(**input_kwargs)
        logits = outputs["logits"]

    # === TAPAS ===
    elif isinstance(model, TapasForMaskedLM):
        input_kwargs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "token_type_ids": batch["token_type_ids"].to(device) if batch.get("token_type_ids") is not None else None
        }
        with torch.no_grad():
            outputs = model(**input_kwargs)
        logits = outputs.logits

    # === BERT ===
    else:
        input_kwargs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device)
        }
        with torch.no_grad():
            outputs = model(**input_kwargs)
        logits = outputs.logits
    

    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids[0].tolist())
    return predicted_tokens

def evaluate_masked_prediction(dataset, model, tokenizer, collator, epoch):
    """
    Evaluate how well the model recovers masked tokens using HAETAE's masking logic.
    Adapts behavior dynamically via collator.set_epoch(epoch).
    """
    if collator is not None:
        collator.set_epoch(epoch)
    
    correct = 0
    total = 0

    for i, example in enumerate(dataset):
        batch, masked_tokens, label_tokens = mask_entry(example, tokenizer, collator)
        predicted_tokens = predict_masked_tokens(model, tokenizer, batch)

        # Evaluate prediction quality
        for label, pred in zip(label_tokens, predicted_tokens):
            if label in ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]:
                continue
            if label == pred:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"✅ Accuracy: {correct}/{total} = {accuracy:.4f}")

    return accuracy
    

######################## GET EMBEDDING ############################
# === ATLAS, HAETAE, BERT ===
def get_meanpooled_embedding(dataset, idx, model):
    # Create a copy of the data item to avoid modifying the original dataset
    if hasattr(dataset, '__getitem__'):
        # Handle both regular lists and dataset objects
        data_item = dataset[idx]
        if hasattr(data_item, 'copy'):
            # If it's a regular dict, copy it
            data_item = data_item.copy()
        else:
            # If it's a dataset item, convert to dict
            data_item = dict(data_item)
    else:
        data_item = dataset[idx].copy()
    
    if isinstance(model, NaviForMaskedLM):
        allowed_keys = ['input_ids', 'attention_mask', 'position_ids', 'segment_ids', 'header_strings']
        if 'header_positions' in data_item and isinstance(data_item['header_positions'], dict):
            data_item['header_positions'] = [data_item['header_positions']]  # wrap as batched input
        if 'header_strings' in data_item and isinstance(data_item['header_strings'], list):
            if all(isinstance(k, str) for k in data_item['header_strings']):
                data_item['header_strings'] = [data_item['header_strings']]  # wrap as batched input
    elif isinstance(model, HAETAE):
        allowed_keys = ['input_ids', 'attention_mask', 'key_positions']
        if 'key_positions' in data_item and isinstance(data_item['key_positions'], dict):
            data_item['key_positions'] = [data_item['key_positions']]  # wrap as batched input
    elif isinstance(model, TapasForMaskedLM):
        allowed_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        if 'token_type_ids' in data_item and isinstance(data_item['token_type_ids'], dict):
            data_item['token_type_ids'] = [data_item['token_type_ids']]  # wrap as batched input
    else:
        allowed_keys = ['input_ids', 'attention_mask']
        
    device = next(model.parameters()).device

    # Prepare inputs
    inputs = {
        k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in data_item.items()
        if k in allowed_keys
    }

    with torch.no_grad():
        if isinstance(model, NaviForMaskedLM):
            outputs = model(**inputs)
            last_hidden = outputs[0]  # (1, seq_len, hidden)
        elif isinstance(model, HAETAE):
            outputs = model(**inputs)
            last_hidden = outputs["hidden_states"]  # (1, seq_len, hidden)
        elif isinstance(model, BertForMaskedLM):
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden)
        elif isinstance(model, TapasForMaskedLM):
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden)
    attention_mask = inputs['attention_mask'].unsqueeze(-1)  # (1, seq_len, 1)
    masked_hidden = last_hidden * attention_mask  # zero out paddings
    sum_hidden = masked_hidden.sum(dim=1)  # (1, hidden)
    valid_token_counts = attention_mask.sum(dim=1)  # (1, 1)
    mean_pooled = (sum_hidden / valid_token_counts).squeeze().to("cpu").numpy()  # (hidden,)

    return mean_pooled

def get_cls_embedding(dataset, idx, model):
    # Handle different dataset types
    if hasattr(dataset, '__getitem__'):
        if hasattr(dataset[idx], 'copy'):
            # If it's a regular dict, copy it
            data_item = dataset[idx].copy()
        else:
            # If it's a dataset item, convert to dict
            data_item = dict(dataset[idx])
    else:
        data_item = dataset[idx].copy()
    
    if isinstance(model, NaviForMaskedLM):
        allowed_keys = ['input_ids', 'attention_mask', 'position_ids', 'segment_ids', 'header_strings']
        if 'header_positions' in data_item and isinstance(data_item['header_positions'], dict):
            data_item['header_positions'] = [data_item['header_positions']]  # wrap as batched input
        if 'header_strings' in data_item and isinstance(data_item['header_strings'], list):
            if all(isinstance(k, str) for k in data_item['header_strings']):
                data_item['header_strings'] = [data_item['header_strings']]  # wrap as batched input
    elif isinstance(model, HAETAE):
        allowed_keys = ['input_ids', 'attention_mask', 'key_positions']
        if 'key_positions' in data_item and isinstance(data_item['key_positions'], dict):
            data_item['key_positions'] = [data_item['key_positions']]  # wrap as batched input
    elif isinstance(model, TapasForMaskedLM):
        allowed_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        if 'token_type_ids' in data_item and isinstance(data_item['token_type_ids'], dict):
            data_item['token_type_ids'] = [data_item['token_type_ids']]  # wrap as batched input
    else:
        allowed_keys = ['input_ids', 'attention_mask']
        
    device = next(model.parameters()).device

    # Prepare inputs
    inputs = {
        k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in data_item.items()
        if k in allowed_keys
    }

    with torch.no_grad():
        if isinstance(model, NaviForMaskedLM):
            outputs = model(**inputs)
            cls_embedding = outputs[0][:, 0, :].squeeze().to("cpu").numpy()
        elif isinstance(model, HAETAE):
            outputs = model(**inputs)
            cls_embedding = outputs["hidden_states"][:, 0, :].squeeze().to("cpu").numpy()
        elif isinstance(model, BertForMaskedLM):
            outputs = model(**inputs, output_hidden_states=True)
            cls_embedding = outputs.hidden_states[-1][:, 0, :].squeeze().to("cpu").numpy()
        elif isinstance(model, TapasForMaskedLM):
            outputs = model(**inputs, output_hidden_states=True)
            cls_embedding = outputs.hidden_states[-1][:, 0, :].squeeze().to("cpu").numpy()
    return cls_embedding

def extract_header_value_embeddings_from_positions(embeddings, header_positions, value_positions):
    """
    Extracts header and value embeddings from contextualized embeddings based on their token positions.
    This is a utility function adapted for inference, assuming a batch size of 1.
    """
    hidden_size = embeddings.size(-1)
    
    # Assuming batch size is 1, so we access the first element
    embeddings = embeddings.squeeze(0) # From (1, seq_len, hidden) to (seq_len, hidden)
    
    header_pos_dict = header_positions[0] if isinstance(header_positions, list) else header_positions
    value_pos_dict = value_positions[0] if isinstance(value_positions, list) else value_positions

    header_keys = list(header_pos_dict.keys())
    num_headers = len(header_keys)
    
    header_embeds = torch.zeros((1, num_headers, hidden_size), device=embeddings.device)
    val_embeds = torch.zeros((1, num_headers, hidden_size), device=embeddings.device)
    
    for i, header_name in enumerate(header_keys):
        # Extract header embedding
        h_token_indices = header_pos_dict.get(header_name, [])
        if h_token_indices:
            header_tokens = embeddings[h_token_indices]
            header_embeds[0, i] = header_tokens.mean(dim=0)
            
        # Extract value embedding
        v_token_indices = value_pos_dict.get(header_name, [])
        if v_token_indices:
            value_tokens = embeddings[v_token_indices]
            val_embeds[0, i] = value_tokens.mean(dim=0)
            
    return header_embeds, val_embeds


def get_meanpooled_segment_embedding(dataset, idx, model):
    """
    Extracts mean-pooled segment embeddings for Navi models.
    """
    if not isinstance(model, NaviForMaskedLM):
        raise TypeError("This function is only for NaviForMaskedLM models.")

    # --- 1. Get data and prepare inputs ---
    if hasattr(dataset, '__getitem__'):
        data_item = dict(dataset[idx])
    else:
        data_item = dataset[idx].copy()

    allowed_keys = ['input_ids', 'attention_mask', 'position_ids', 'segment_ids', 'header_strings']
    device = next(model.parameters()).device
    inputs = {
        k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in data_item.items()
        if k in allowed_keys
    }
    if 'header_strings' in inputs and isinstance(inputs['header_strings'], list):
         if all(isinstance(k, str) for k in inputs['header_strings']):
                inputs['header_strings'] = [inputs['header_strings']]

    # --- 2. Get contextualized embeddings ---
    with torch.no_grad():
        outputs = model(**inputs)
        contextualized_embeddings = outputs[0]  # (1, seq_len, hidden)

    # --- 3. Extract segment components ---
    header_positions = data_item.get('header_positions')
    value_positions = data_item.get('value_positions')
    header_strings = data_item.get('header_strings')

    if not all([header_positions, value_positions, header_strings]):
        raise ValueError("Missing header_positions, value_positions, or header_strings for segment embedding.")
    
    # Ensure header_strings is in the right format for the header encoder
    if isinstance(header_strings, list) and all(isinstance(k, str) for k in header_strings):
        header_strings_for_encoder = [header_strings]
    else:
        header_strings_for_encoder = header_strings

    with torch.no_grad():
        E_univ, _ = model.bert.embeddings.header_encoder(header_strings_for_encoder)
        H_ctx, V_ctx = extract_header_value_embeddings_from_positions(
            contextualized_embeddings,
            header_positions,
            value_positions
        )

        # --- 4. Create and mean-pool segment embeddings ---
        segment_embeddings = model.create_segment_embeddings(E_univ, H_ctx, V_ctx) # (1, num_segments, hidden)
        
    mean_pooled = segment_embeddings.mean(dim=1).squeeze().cpu().numpy() # (hidden,)
    return mean_pooled


# === TABBIE ===
def extract_cls_embeddings_tabbie(dataset, model):
    """
    Extract CLS embeddings for each row in the dataset using Tabbie.
    Returns: np.ndarray of shape (num_rows, hidden_dim)
    """
    device = next(model.parameters()).device
    embeddings = []
    for i in range(len(dataset)):
        batch = dataset[i]
        with torch.no_grad():
            output = model(
                table_info=[{'id': batch['id'], 'num_rows': batch['num_rows'], 'num_cols': batch['num_cols']}],
                indexed_headers={'input_ids': batch['header_input_ids'].unsqueeze(0).to(device)},
                indexed_cells={'input_ids': batch['table_input_ids'].unsqueeze(0).to(device)}
            )
            # Assume output['cls_embedding'] is (1, hidden_dim)
            cls_emb = output['cls_embedding'].squeeze(0).cpu().numpy()
            embeddings.append(cls_emb)
    return np.stack(embeddings)

###################### CLASSIFICATION EXPERIMENT ##########################

def run_row_classification(X, y, model_type="rf", test_size=0.2):
    # ✅ Encode string labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, stratify=y_encoded)

    if model_type == "rf":
        clf = RandomForestClassifier()
    elif model_type == "xgboost":
        clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', tree_method='gpu_hist')
    elif model_type == "lr":
        clf = LogisticRegression(max_iter=1000)
    elif model_type == "svm":
        clf = SVC()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\n✅ Macro-F1 ({model_type.upper()} row classification): {f1:.4f}")
    return f1

###################### CLUSTERING EXPERIMENT ##########################

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



def run_row_clustering(X, y):
    """
    Run row clustering using various algorithms and evaluate with internal metrics.
    
    Args:
        X (np.ndarray): The embedding vectors for the dataset.
        y (np.ndarray): The ground truth labels, used to determine n_clusters.
        
    Returns:
        dict: A dictionary containing the scores for each clustering algorithm.
    """
    results = {}
    # Determine the number of clusters from the ground truth labels
    n_clusters = len(np.unique(y))
    print(f"  Target number of clusters: {n_clusters}")
    
    # Define clustering models
    models = {
        "KMeans": KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        "Agglomerative": AgglomerativeClustering(n_clusters=n_clusters),
    }
    
    for name, model in models.items():
        print(f"  Running {name}...")
        try:
            cluster_labels = model.fit_predict(X)
            
            # Check if clustering produced more than one cluster (a requirement for metrics)
            n_labels = len(set(cluster_labels))
            if -1 in cluster_labels: # Exclude noise point label from DBSCAN
                n_labels -= 1

            if n_labels < 2:
                print(f"    - {name} found fewer than 2 clusters. Metrics cannot be calculated.")
                results[name] = {
                    "Silhouette": None,
                    "Calinski-Harabasz": None,
                    "Davies-Bouldin": None,
                    "NMI": None,
                    "B-cubed Precision": None,
                    "B-cubed Recall": None,
                    "B-cubed F1": None,
                }
                continue

            # Internal metrics (no ground truth needed)
            silhouette = silhouette_score(X, cluster_labels)
            calinski = calinski_harabasz_score(X, cluster_labels)
            davies = davies_bouldin_score(X, cluster_labels)
            
            # External metrics (require ground truth)
            nmi = normalized_mutual_info_score(y, cluster_labels)
            b_cubed = b_cubed_score(y, cluster_labels)
            
            results[name] = {
                "Silhouette": float(silhouette),
                "Calinski-Harabasz": float(calinski),
                "Davies-Bouldin": float(davies),
                "NMI": float(nmi),
                "B-cubed Precision": b_cubed['precision'],
                "B-cubed Recall": b_cubed['recall'],
                "B-cubed F1": b_cubed['f1'],
            }
            print(f"    - Silhouette: {silhouette:.4f}, Calinski-Harabasz: {calinski:.4f}, Davies-Bouldin: {davies:.4f}")
            print(f"    - NMI: {nmi:.4f}, B-cubed F1: {b_cubed['f1']:.4f}")

        except Exception as e:
            print(f"    - An error occurred while running {name}: {e}")
            results[name] = {
                "Silhouette": None,
                "Calinski-Harabasz": None,
                "Davies-Bouldin": None,
                "NMI": None,
                "B-cubed Precision": None,
                "B-cubed Recall": None,
                "B-cubed F1": None,
            }
            
    return results