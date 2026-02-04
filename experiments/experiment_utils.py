import json
import pandas as pd
from io import StringIO
import torch
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier  # Commented out - not used
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC  # Removed - not used
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
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

    # === NAVI ===
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
# === NAVI, HAETAE, BERT ===
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

###################### RAW FEATURE EXTRACTION ##########################

def extract_raw_features(dataset, target_col):
    """
    Extract raw features from JSON dataset for end-to-end baselines.
    
    Converts JSON rows to numerical feature matrix by:
    - Keeping numeric fields as-is (converting to float)
    - Label encoding categorical fields
    - Excluding text fields or using simple features (length, word count)
    - Handling missing values
    
    Args:
        dataset: List of JSON dictionaries (rows)
        target_col: Name of target column to exclude from features
        
    Returns:
        tuple: (X, y) where X is numpy array of features and y is array of labels
    """
    import re
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(dataset)
    
    # Extract labels
    y = df[target_col].values
    
    # Remove target column from features
    feature_df = df.drop(columns=[target_col])
    
    # Process each column
    processed_features = []
    feature_names = []
    
    for col in feature_df.columns:
        col_data = feature_df[col]
        
        # Check if column is numeric - values must be directly convertible to float or int
        numeric_values = []
        is_numeric = True
        
        for val in col_data:
            if pd.isna(val) or val == '' or val is None or val == 'None':
                numeric_values.append(np.nan)
                continue
            
            # Try to convert directly to float or int
            try:
                val_str = str(val).strip()
                # Try direct conversion (handles scientific notation like "3.4e0", "5", "22.00")
                try:
                    float_val = float(val_str)
                    numeric_values.append(float_val)
                except ValueError:
                    # If direct conversion fails, column is not numeric
                    is_numeric = False
                    break
            except (ValueError, TypeError):
                is_numeric = False
                break
        
        if is_numeric and len([v for v in numeric_values if not pd.isna(v)]) > 0:
            # Numeric column - keep as float
            processed_features.append(numeric_values)
            feature_names.append(f"{col}_numeric")
        else:
            # Categorical or text column
            # Check if it's categorical (limited unique values) or text
            unique_ratio = col_data.nunique() / len(col_data) if len(col_data) > 0 else 1.0
            
            if unique_ratio < 0.5 and col_data.nunique() < 100:
                # Treat as categorical - use label encoding
                le = LabelEncoder()
                # Handle missing values by using a special category
                col_filled = col_data.fillna('__MISSING__')
                encoded = le.fit_transform(col_filled.astype(str))
                processed_features.append(encoded.tolist())
                feature_names.append(f"{col}_categorical")
            else:
                # Treat as text - extract simple features
                text_features = []
                for val in col_data:
                    if pd.isna(val) or val == '' or val is None:
                        text_features.append([0, 0])  # length=0, word_count=0
                    else:
                        val_str = str(val)
                        text_features.append([len(val_str), len(val_str.split())])
                
                # Add as two separate features
                text_lengths = [f[0] for f in text_features]
                text_word_counts = [f[1] for f in text_features]
                processed_features.append(text_lengths)
                feature_names.append(f"{col}_text_length")
                processed_features.append(text_word_counts)
                feature_names.append(f"{col}_text_word_count")
    
    # Convert to numpy array
    if processed_features:
        X = np.array(processed_features).T  # Transpose to get (n_samples, n_features)
        
        # Handle missing values with imputation
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        
        # Ensure all values are finite
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        # Fallback: create dummy features if no features extracted
        X = np.zeros((len(dataset), 1))
    
    return X, y

def extract_features_tablevectorizer(dataset, target_col):
    """
    Extract features using TableVectorizer from skrub.
    
    Args:
        dataset: List of JSON dictionaries (rows)
        target_col: Name of target column to exclude from features
        
    Returns:
        tuple: (X, y) where X is feature matrix and y is labels
    """
    from skrub import TableVectorizer
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset)
    
    # Extract labels
    y = df[target_col].values
    
    # Remove target column from features
    feature_df = df.drop(columns=[target_col])
    
    # Try to use TextEncoder for high-cardinality text columns, fall back to default if it fails
    vectorizer = None
    try:
        from skrub import TextEncoder
        # Use TableVectorizer with TextEncoder for high-cardinality text columns
        vectorizer = TableVectorizer(high_cardinality=TextEncoder())
        # Try to fit_transform - if this fails, we'll fall back to default
        X = vectorizer.fit_transform(feature_df)
    except (ImportError, ValueError, Exception) as e:
        # If TextEncoder fails (e.g., due to dependency issues), use default TableVectorizer
        print(f"⚠️  Warning: TextEncoder failed ({type(e).__name__}: {e}), using default TableVectorizer")
        vectorizer = TableVectorizer()
        X = vectorizer.fit_transform(feature_df)
    
    # Convert to numpy array if it's a sparse matrix
    if hasattr(X, 'toarray'):
        X = X.toarray()
    else:
        X = np.array(X)
    
    return X, y

def extract_features_textencoder(dataset, target_col):
    """
    Extract features using TextEncoder from skrub.
    Linearizes table rows using "{key} is {value}" format similar to BertDataset.
    
    Args:
        dataset: List of JSON dictionaries (rows)
        target_col: Name of target column to exclude from features
        
    Returns:
        tuple: (X, y) where X is feature matrix and y is labels
    """
    from skrub import TextEncoder
    
    # Linearize each row into text format: "{key} is {value}"
    linearized_texts = []
    y = []
    
    for row_dict in dataset:
        # Remove target column
        feature_dict = {k: v for k, v in row_dict.items() if k != target_col}
        
        # Linearize: "{key} is {value}"
        text_parts = []
        for key, value in feature_dict.items():
            if value is not None and value != '':
                text_parts.append(f"{key} is {value}")
        linearized_text = " ".join(text_parts)
        linearized_texts.append(linearized_text)
        
        y.append(row_dict[target_col])
    
    # Use TextEncoder to encode the linearized texts
    text_encoder = TextEncoder()
    
    # Convert to DataFrame with single column
    text_df = pd.DataFrame({'text': linearized_texts})
    
    # Fit and transform
    X = text_encoder.fit_transform(text_df)
    
    # Convert to numpy array if it's a sparse matrix
    if hasattr(X, 'toarray'):
        X = X.toarray()
    else:
        X = np.array(X)
    
    y = np.array(y)
    
    return X, y

def extract_concatenated_navi_features(dataset, target_col, model, dataset_X):
    """
    Extract concatenated features: NAVI embeddings + numerical attributes.
    
    Args:
        dataset: List of JSON dictionaries (rows) with target column
        target_col: Name of target column
        model: NAVI model instance
        dataset_X: Preprocessed dataset (NaviDataset) without target column
        
    Returns:
        tuple: (X, y) where X has shape (n_samples, 768 + n_numerical)
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    
    # Extract NAVI embeddings
    navi_embeddings = []
    labels = []
    numerical_features_list = []
    
    # Convert to DataFrame to identify numerical columns
    df = pd.DataFrame(dataset)
    feature_df = df.drop(columns=[target_col])
    
    # Identify numerical columns - values must be directly convertible to float or int
    numerical_cols = []
    for col in feature_df.columns:
        # First check if already numeric dtype
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            numerical_cols.append(col)
        else:
            # Check if all non-null values can be directly converted to float or int
            sample_values = feature_df[col].dropna().head(100)  # Sample first 100 non-null values
            if len(sample_values) == 0:
                continue
            
            is_numeric = True
            for val in sample_values:
                if pd.isna(val) or val == '' or val is None or val == 'None':
                    continue
                val_str = str(val).strip()
                # Try direct conversion to float (handles "5", "22.00", "3.4e0", etc.)
                try:
                    float(val_str)
                except (ValueError, TypeError):
                    # If any value can't be converted directly, column is not numeric
                    is_numeric = False
                    break
            
            if is_numeric:
                numerical_cols.append(col)
    
    # Extract features for each row
    for i, row in enumerate(dataset):
        # Get NAVI embedding (768-dim)
        navi_emb = get_cls_embedding(dataset_X, i, model)
        navi_embeddings.append(navi_emb)
        
        # Extract numerical attributes
        numerical_vals = []
        for col in numerical_cols:
            val = row.get(col)
            if pd.isna(val) or val is None or val == '' or val == 'None':
                numerical_vals.append(0.0)
            else:
                try:
                    val_str = str(val).strip()
                    # Direct conversion to float (handles "5", "22.00", "3.4e0", etc.)
                    numerical_vals.append(float(val_str))
                except (ValueError, TypeError):
                    numerical_vals.append(0.0)
        numerical_features_list.append(numerical_vals)
        
        labels.append(row[target_col])
    
    # Stack NAVI embeddings
    navi_X = np.stack(navi_embeddings)  # (n_samples, 768)
    
    # Stack numerical features
    if numerical_cols and len(numerical_cols) > 0:
        numerical_X = np.array(numerical_features_list)  # (n_samples, n_numerical)
        
        # Check if we actually have numerical features (not all empty)
        if numerical_X.shape[1] > 0:
            # Scale numerical features
            scaler = StandardScaler()
            numerical_X = scaler.fit_transform(numerical_X)
            
            # Concatenate
            X = np.hstack([navi_X, numerical_X])  # (n_samples, 768 + n_numerical)
        else:
            # No numerical columns, just use NAVI embeddings
            X = navi_X
    else:
        # No numerical columns, just use NAVI embeddings
        X = navi_X
    
    y = np.array(labels)
    
    return X, y

def run_classification_with_features(X, y, model_type="xgboost", test_size=0.2):
    """
    Generic function to run classification on features with XGBoost, CatBoost, LogisticRegression, or TabPFN.
    
    Args:
        X: Feature matrix (numpy array)
        y: Labels (array-like)
        model_type: "xgboost", "catboost", "tabpfn", or "lr"
        test_size: Proportion of data to use for testing
        
    Returns:
        float: Macro-F1 score
    """
    if model_type == "tabpfn":
        return run_row_classification_tabpfn(X, y, test_size=test_size)
    elif model_type in ["xgboost", "catboost", "lr"]:
        return run_row_classification(X, y, model_type=model_type, test_size=test_size)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

###################### CLASSIFICATION EXPERIMENT ##########################

def run_row_classification(X, y, model_type="xgboost", test_size=0.2):
    # ✅ Encode string labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Check for NaN values and handle them for models that don't support NaN
    # Convert to numpy array if needed (handles sparse matrices)
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X)
    
    has_nan = np.isnan(X).any()
    
    # Models that don't support NaN: lr
    # Models that can handle NaN: xgboost, catboost (but we'll impute for consistency)
    if has_nan:
        # Use SimpleImputer to fill NaN values with constant value 0
        imputer = SimpleImputer(strategy='constant', fill_value=0)
        X = imputer.fit_transform(X)

    # Use random_state=None to respect np.random.seed() set by caller
    # This ensures reproducibility when the same seed is used
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=None
    )

    # Scale features for models that are sensitive to feature scales
    if model_type == "lr":
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Use random_state=None to respect np.random.seed() set by caller
    # This ensures reproducibility when the same seed is used
    # if model_type == "rf":
    #     clf = RandomForestClassifier(random_state=None)  # Commented out - not used
    if model_type == "xgboost":
        # XGBoost >= 2.0 removed/changed GPU tree_method strings like "gpu_hist".
        # Use "hist" for broad compatibility; GPU usage (if available) is handled
        # by XGBoost configuration/installation rather than this string.
        clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', tree_method='hist', random_state=None)
    elif model_type == "catboost":
        clf = CatBoostClassifier(random_state=None, verbose=False, iterations=100)
    elif model_type == "lr":
        clf = LogisticRegression(max_iter=1000, random_state=None)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\n✅ Macro-F1 ({model_type.upper()} row classification): {f1:.4f}")
    return f1

def run_row_classification_tabpfn(X, y, test_size=0.2):
    """
    Run classification using TabPFN on raw features.
    
    Args:
        X: Feature matrix (numpy array)
        y: Labels (array-like)
        test_size: Proportion of data to use for testing
        
    Returns:
        float: Macro-F1 score
        
    Raises:
        RuntimeError: If TabPFN authentication fails or model cannot be loaded
    """
    # TabPFN depends on numba/llvmlite. On some systems the dynamic loader may pick up
    # an older system libstdc++ first, causing errors like:
    #   CXXABI_1.3.15 not found (required by .../libllvmlite.so)
    #
    # Make the runtime more robust by:
    # - Prepending $CONDA_PREFIX/lib to LD_LIBRARY_PATH
    # - Preloading conda's libstdc++/libgcc_s with RTLD_GLOBAL
    def _ensure_conda_runtime_for_llvmlite():
        import os
        import ctypes

        prefix = os.environ.get("CONDA_PREFIX")
        if not prefix:
            return

        libdir = os.path.join(prefix, "lib")

        # Ensure LD_LIBRARY_PATH prefers conda libs.
        ld = os.environ.get("LD_LIBRARY_PATH", "")
        ld_parts = [p for p in ld.split(":") if p]
        if libdir not in ld_parts:
            os.environ["LD_LIBRARY_PATH"] = f"{libdir}:{ld}" if ld else libdir

        # Preload runtime libs if present.
        candidates = [
            os.path.join(libdir, "libstdc++.so.6"),
            os.path.join(prefix, "libstdc++.so.6"),
            os.path.join(libdir, "libgcc_s.so.1"),
            os.path.join(prefix, "libgcc_s.so.1"),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    # Best-effort: if it fails, the subsequent import will raise with details.
                    pass

    _ensure_conda_runtime_for_llvmlite()

    try:
        from tabpfn import TabPFNClassifier
    except ImportError:
        raise ImportError("TabPFN is not installed. Please install it with: pip install tabpfn")
    except OSError as e:
        # Commonly triggered by llvmlite/numba binary/runtime issues.
        raise RuntimeError(
            "TabPFN import failed due to a native dependency issue (likely numba/llvmlite + libstdc++).\n"
            "If you're using conda, try re-activating the env so LD_LIBRARY_PATH points to $CONDA_PREFIX/lib.\n"
            f"Original error: {e}"
        )
    
    # Encode string labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # TabPFN requires specific data types
    X = X.astype(np.float32)
    y_encoded = y_encoded.astype(np.int64)
    
    # Split data (use random_state=None to respect np.random.seed() set by caller)
    # This ensures reproducibility when the same seed is used
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=None
    )
    
    try:
        # Initialize and train TabPFN
        # TabPFN is a prior-data fitted network, so it doesn't need training
        # but we still need to fit it to get predictions
        # Use GPU if available, otherwise CPU
        import torch
        tabpfn_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        clf = TabPFNClassifier(device=tabpfn_device)
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = clf.predict(X_test)
        
        # Calculate F1 score
        f1 = f1_score(y_test, y_pred, average='macro')
        print(f"\n✅ Macro-F1 (TabPFN row classification): {f1:.4f}")
        return f1
    except (RuntimeError, Exception) as e:
        error_msg = str(e)
        if "Authentication error" in error_msg or "gated" in error_msg.lower():
            raise RuntimeError(
                "TabPFN requires HuggingFace authentication for gated models.\n"
                "Options:\n"
                "1. Run: hf auth login\n"
                "2. Set environment variable: export HF_TOKEN=your_token_here\n"
                "   (Get token from https://huggingface.co/settings/tokens)\n"
                "3. Visit https://huggingface.co/Prior-Labs/tabpfn_2_5 to accept terms first.\n"
                f"Original error: {error_msg}"
            )
        else:
            raise RuntimeError(f"TabPFN initialization failed: {error_msg}")

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
    # Use random_state=None to respect np.random.seed() set by caller
    # This ensures reproducibility when the same seed is used
    models = {
        "KMeans": KMeans(n_clusters=n_clusters, random_state=None, n_init=10),
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