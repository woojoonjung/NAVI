import torch
import json
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoTokenizer
import random
import math

# -----------------------------
# Dataset classes for Navi
# Ablation for "woSI" processed here
# -----------------------------
class Segment:
    """
    Represents a header-value pair as a segment.
    """
    def __init__(self, header: str, value: str):
        self.header = header
        self.value = value
    
    def serialize(self):
        """Converts the segment into a 'header : value' string representation."""
        return f"{self.header} : {self.value}"


class Row:
    """
    Represents a row of data, consisting of multiple Segments and a CLS token.
    """
    def __init__(self, segments, ablation_mode):
        self.cls_token = "[CLS]"
        self.segments = []
        for segment in segments:
            self.segments.append(segment)
        self.ablation_mode = ablation_mode
    def serialize(self):
        """Converts the Row into a single structured string."""
        return f"{self.cls_token} " + " ".join([seg.serialize() for seg in self.segments])

    def preprocess_for_input(self, tokenizer):
        """
        Builds token ids, position ids, and segment ids with [SEP] as delimiter.
        Returns:
            - input_ids: List[int]
            - pos_ids: List[int]
            - segment_ids: List[int]
            - header_positions: Dict[str, List[int]]
            - value_positions: Dict[str, List[int]]
        """
        input_ids = [tokenizer.cls_token_id]
        pos_ids = [0]
        segment_ids = [0]
        current_token_index = 1
        current_segment_id = 0

        header_positions = {}
        value_positions = {}

        for segment in self.segments:
            current_segment_id += 1 # Increment segment id
            if self.ablation_mode != "woSI" or current_segment_id == 1:
                segment_pos = 1  # Reset inside each header-value segment

            header_tokens = tokenizer.tokenize(str(segment.header))
            value_tokens = tokenizer.tokenize(str(segment.value))
            header_token_ids = tokenizer.convert_tokens_to_ids(header_tokens)
            value_token_ids = tokenizer.convert_tokens_to_ids(value_tokens)

            header_pos = []
            for tid in header_token_ids:
                input_ids.append(tid)
                pos_ids.append(segment_pos)
                segment_ids.append(current_segment_id)
                header_pos.append(current_token_index)
                current_token_index += 1
                segment_pos += 1
            header_positions[segment.header] = header_pos

            # Colon between header and value
            colon_id = tokenizer.convert_tokens_to_ids([":"])[0]
            input_ids.append(colon_id)
            pos_ids.append(segment_pos)
            segment_ids.append(current_segment_id)
            current_token_index += 1
            segment_pos += 1

            value_pos = []
            for tid in value_token_ids:
                input_ids.append(tid)
                pos_ids.append(segment_pos)
                segment_ids.append(current_segment_id)
                value_pos.append(current_token_index)
                current_token_index += 1
                segment_pos += 1
            value_positions[segment.header] = value_pos

            # Add [SEP]
            sep_id = tokenizer.sep_token_id
            input_ids.append(sep_id)
            pos_ids.append(segment_pos)
            segment_pos += 1
            segment_ids.append(current_segment_id)
            current_token_index += 1

        input_ids = input_ids[:512]
        pos_ids = pos_ids[:512]
        segment_ids = segment_ids[:512]
        header_positions = {
            k: [i for i in v if i < 512]
            for k, v in header_positions.items()
        }
        value_positions = {
            k: [i for i in v if i < 512]
            for k, v in value_positions.items()
        }
        header_positions = {k: v for k, v in header_positions.items() if v}
        value_positions = {k: v for k, v in value_positions.items() if v}

        return input_ids, pos_ids, segment_ids, header_positions, value_positions
    
    def collect_header_strings(self):
        header_strings = []
        for segment in self.segments:
            header_strings.append(f"{segment.header}")
        return header_strings
        

class NaviDataset(Dataset):
    """
    Dataset class for Navi, consisting of multiple SegmentSets.
    """
    def __init__(self, json_data, tokenizer_name="bert-base-uncased", max_length=512, 
                 ablation_mode="full", compute_field_entropy=False):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.ablation_mode = ablation_mode
        self.field_analyzer = None

        # Handle tuple format: (table_id, json_object)
        if json_data and isinstance(json_data[0], tuple):
            self.table_ids = [item[0] for item in json_data]
            self.data = [Row([Segment(k, v) for k, v in item[1].items()], ablation_mode) for item in json_data]
        else:
            # Fallback for original format
            self.table_ids = [0] * len(json_data)  # Default table_id
            self.data = [Row([Segment(k, v) for k, v in row.items()], ablation_mode) for row in json_data]

        if compute_field_entropy:
            print("Initializing field entropy analysis...")
            self.field_analyzer = FieldEntropyAnalyzer()
            self.field_analyzer.compute_field_entropy(self)

    def get_field_categories(self):
        """
        Get field categories if entropy analysis was performed.
        
        Returns:
            dict or None: Field categories if available, None otherwise
        """
        if self.field_analyzer:
            return self.field_analyzer.get_field_categories()
        return None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Tokenizes and returns a single example from the dataset."""
        row = self.data[idx]
        table_id = self.table_ids[idx]
        
        input_ids, pos_ids, segment_ids, header_positions, value_positions = row.preprocess_for_input(self.tokenizer)
        
        # Retrieve list of header names
        header_strings = row.collect_header_strings()
        
        # Truncate to max length
        input_ids = input_ids[:self.max_length]
        pos_ids = pos_ids[:self.max_length]
        segment_ids = segment_ids[:self.max_length]
        
        header_positions = {
            k: [i for i in v if i < self.max_length]
            for k, v in header_positions.items()
        }
        value_positions = {
            k: [i for i in v if i < self.max_length]
            for k, v in value_positions.items()
        }
        header_positions = {k: v for k, v in header_positions.items() if v}
        value_positions = {k: v for k, v in value_positions.items() if v}
        header_strings = [header for header in header_strings if header in header_positions]
        
        # Convert to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        position_ids = torch.tensor(pos_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "segment_ids": segment_ids,
            "header_strings": header_strings,
            "header_positions": header_positions,
            "value_positions": value_positions,
            "table_id": table_id,
            "raw_segments": row.segments
        }

        
# -----------------------------
# Field entropy analyzer
# -----------------------------

class FieldEntropyAnalyzer:
    def __init__(self, delta_threshold=1.5):
        self.delta = delta_threshold
        # Store all metrics on a per-table basis
        self.field_entropy = {} # {table_id: {field: entropy, ...}}
        self.low_entropy_fields = {} # {table_id: set_of_fields}
        self.high_entropy_fields = {} # {table_id: set_of_fields}
        self.field_value_counts = {} # {table_id: {field: {value: count}}}
        self.field_total_counts = {} # {table_id: {field: count}}
        self.total_rows_per_table = {} # {table_id: count}

    def compute_field_entropy(self, dataset):
        print("🔍 Computing field entropy on a per-table basis...")
        
        # Collect value counts for each field within each table
        for idx in range(len(dataset)):
            row_data = dataset.data[idx]
            table_id = dataset.table_ids[idx]

            if table_id not in self.field_value_counts:
                self.field_value_counts[table_id] = {}
                self.field_total_counts[table_id] = {}
                self.total_rows_per_table[table_id] = 0
                self.field_entropy[table_id] = {}
            
            self.total_rows_per_table[table_id] += 1

            for segment in row_data.segments:
                field_name = segment.header
                field_value = str(segment.value)
                
                if field_name not in self.field_value_counts[table_id]:
                    self.field_value_counts[table_id][field_name] = {}
                    self.field_total_counts[table_id][field_name] = 0
                
                self.field_total_counts[table_id][field_name] += 1
                
                if field_value not in self.field_value_counts[table_id][field_name]:
                    self.field_value_counts[table_id][field_name][field_value] = 0
                
                self.field_value_counts[table_id][field_name][field_value] += 1
        
        # Calculate entropy for each field in each table
        for table_id, field_counts in self.field_value_counts.items():
            total_table_rows = self.total_rows_per_table[table_id]
            for field_name, value_counts in field_counts.items():
                entropy = self._calculate_entropy(value_counts, total_table_rows)
                self.field_entropy[table_id][field_name] = entropy
        
        print(f"✅ Computed entropy for {len(self.field_entropy)} tables.")
        
        self.categorize_fields()
        return self.field_entropy
    
    def _calculate_entropy(self, value_counts, total_count):
        """Calculates Shannon entropy."""
        if total_count == 0:
            return 0.0
        
        # Calculate Shannon entropy
        entropy = 0.0
        for count in value_counts.values():
            if count > 0:
                probability = count / total_count
                entropy -= probability * math.log2(probability)
        
        # Normalize by maximum possible entropy for this field
        unique_values = len(value_counts)
        if unique_values <= 1:
            return 0.0
        
        max_entropy = math.log2(unique_values)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return normalized_entropy
    
    def categorize_fields(self):
        """Categorize fields as low/high entropy for each table."""
        if not self.field_entropy:
            print("⚠️ No field entropy computed. Call compute_field_entropy() first.")
            return

        print("📊 Categorizing fields for each table using improved thresholds...")
        for table_id, entropies in self.field_entropy.items():
            self.low_entropy_fields[table_id] = set()
            self.high_entropy_fields[table_id] = set()
            
            entropy_values = sorted(list(entropies.values()))
            
            if len(entropy_values) < 4:
                print(f"   - Table {table_id}: Skipped (only {len(entropy_values)} fields).")
                continue
            
            # Use quartiles for more stable categorization
            n = len(entropy_values)
            q1_idx = max(0, (n) // 4 - 1)
            q3_idx = min(n - 1, (3 * n) // 4)
            low_threshold = entropy_values[q1_idx]
            high_threshold = entropy_values[q3_idx]

            for field_name, entropy in entropies.items():
                if entropy <= low_threshold:
                    self.low_entropy_fields[table_id].add(field_name)
                elif entropy >= high_threshold:
                    self.high_entropy_fields[table_id].add(field_name)
            
            # Ensure minimum number of fields in each category
            if len(self.low_entropy_fields[table_id]) == 0 and len(entropy_values) >= 2:
                # Add the lowest entropy field
                min_field = min(entropies.items(), key=lambda x: x[1])[0]
                self.low_entropy_fields[table_id].add(min_field)
                
            if len(self.high_entropy_fields[table_id]) == 0 and len(entropy_values) >= 2:
                # Add the highest entropy field
                max_field = max(entropies.items(), key=lambda x: x[1])[0]
                self.high_entropy_fields[table_id].add(max_field)
            
            print(f"   - Table {table_id}: Low={len(self.low_entropy_fields[table_id])}, High={len(self.high_entropy_fields[table_id])} fields.-> Total {len(entropies)} fields.")
            print(f"     - Low entropy fields: {self.low_entropy_fields[table_id]}")
            print(f"     - High entropy fields: {self.high_entropy_fields[table_id]}")

    def get_field_categories(self):
        """
        Get the categorized fields for all tables.
        
        Returns:
            dict: {table_id: {'low_entropy': set, 'high_entropy': set}}
        """
        all_categories = {}
        for table_id in self.field_entropy.keys():
            all_categories[table_id] = {
                'low_entropy': self.low_entropy_fields.get(table_id, set()),
                'high_entropy': self.high_entropy_fields.get(table_id, set()),
                'field_entropy': self.field_entropy.get(table_id, {}).copy()
            }
        return all_categories
    
    def get_field_entropy_stats(self):
        """
        Get statistical information about field entropy.
        
        Returns:
            dict: Entropy statistics
        """
        if not self.field_entropy:
            return {}
        
        entropy_values = list(self.field_entropy.values())
        return {
            'min_entropy': min(entropy_values),
            'max_entropy': max(entropy_values),
            'mean_entropy': sum(entropy_values) / len(entropy_values),
            'median_entropy': sorted(entropy_values)[len(entropy_values) // 2],
            'total_fields': len(self.field_entropy),
            'total_rows': self.total_rows
        }
    
    def print_detailed_analysis(self):
        """
        Print detailed analysis of field entropy.
        """
        if not self.field_entropy:
            print("⚠️ No field entropy computed yet.")
            return
        
        print("\nDetailed Field Entropy Analysis")
        print("=" * 50)
        
        # Sort fields by entropy
        sorted_fields = sorted(self.field_entropy.items(), key=lambda x: x[1], reverse=True)
        
        print(f"{'Field Name':<25} {'Entropy':<10} {'Category':<15} {'Unique Values':<15}")
        print("-" * 70)
        
        for field_name, entropy in sorted_fields:
            if field_name in self.low_entropy_fields:
                category = "Low (Schema)"
            elif field_name in self.high_entropy_fields:
                category = "High (Entity)"
            else:
                category = "Medium"
            
            unique_values = len(self.field_value_counts.get(field_name, {}))
            print(f"{field_name:<25} {entropy:<10.4f} {category:<15} {unique_values:<15}")
        
        # Print statistics
        stats = self.get_field_entropy_stats()
        print("\nEntropy Statistics")
        print(f"   Min: {stats['min_entropy']:.4f}")
        print(f"   Max: {stats['max_entropy']:.4f}")
        print(f"   Mean: {stats['mean_entropy']:.4f}")
        print(f"   Median: {stats['median_entropy']:.4f}")
    

# -----------------------------
# Custom Sampler for stratified sampling
# -----------------------------

class StratifiedBatchSampler:
    """Ensures each batch contains proportional representation from all table_ids based on actual row counts."""
    
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.table_ids = dataset.table_ids
        
        # Group indices by table_id
        self.table_indices = {}
        for i, table_id in enumerate(self.table_ids):
            if table_id not in self.table_indices:
                self.table_indices[table_id] = []
            self.table_indices[table_id].append(i)
        
        # Calculate proportional ratios for each table_id
        total_rows = len(self.dataset)
        self.table_ratios = {}
        for table_id, indices in self.table_indices.items():
            self.table_ratios[table_id] = len(indices) / total_rows
        
        # Calculate how many rows each table_id should contribute to a batch
        self.table_batch_counts = {}
        for table_id, ratio in self.table_ratios.items():
            count = max(1, round(ratio * batch_size))
            self.table_batch_counts[table_id] = min(count, len(self.table_indices[table_id]))
        
        # Adjust batch counts to ensure they sum to batch_size
        total_batch_count = sum(self.table_batch_counts.values())
        if total_batch_count != batch_size:
            largest_table = max(self.table_batch_counts.items(), key=lambda x: len(self.table_indices[x[0]]))
            largest_table_id = largest_table[0]
            adjustment = batch_size - total_batch_count
            new_count = self.table_batch_counts[largest_table_id] + adjustment
            if new_count >= 1 and new_count <= len(self.table_indices[largest_table_id]):
                self.table_batch_counts[largest_table_id] = new_count
        
        # Verify the sum equals batch_size
        final_sum = sum(self.table_batch_counts.values())
        assert final_sum == batch_size, f"Batch counts sum to {final_sum}, expected {batch_size}"

        # Shuffle within each table_id
        for table_id in self.table_indices:
            random.shuffle(self.table_indices[table_id])
        
        print(f" Batch composition: {self.table_batch_counts}")
        print(f" Sum verification: {final_sum}")

    def __iter__(self):
        """Yield individual indices for the DataLoader to batch together."""
        # Create a list of all indices, maintaining proportional representation
        all_indices = []
        
        # For each table_id, add indices in proportion to their batch_count
        for table_id, target_count in self.table_batch_counts.items():
            indices = self.table_indices[table_id].copy()
            random.shuffle(indices)  # Shuffle within each table_id
            
            # Add indices in the proportion they should appear in each batch
            for i in range(0, len(indices), target_count):
                batch_indices = indices[i:i + target_count]
                all_indices.extend(batch_indices)
        
        # Shuffle the entire list to randomize batch order
        random.shuffle(all_indices)
        
        # Yield individual indices (DataLoader will batch them)
        for idx in all_indices:
            yield idx

    def __len__(self):
        """Return the total number of samples."""
        return len(self.dataset)
        

# -----------------------------
# Dataset classes for baseline models
# -----------------------------
class BertDataset(Dataset):
    def __init__(self, json_data, tokenizer_name='bert-base-uncased', max_length=512, mode="training"):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.data = json_data
        self.mode = mode

    def _serialize_vanilla(self, json_obj, parent_key="", sep="."):
        """
        Serialize a JSON object into a string format suitable for tokenization, handling nested structures.
        """
        serialized = []
        for key, value in json_obj.items():
            full_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict):
                serialized.append(self._serialize_vanilla(value, parent_key=full_key, sep=sep))
            elif isinstance(value, list):
                list_content = ", ".join([str(v) if not isinstance(v, dict) else self._serialize_vanilla(v, parent_key=full_key, sep=sep) for v in value])
                serialized.append(f"{full_key} : [{list_content}]")
                serialized.append(",")
            else:
                serialized.append(f"{full_key} : {value}")
                serialized.append(",")
        return " ".join(serialized)

    def _extract_key_value_positions(self, json_obj, tokenized_text):
        """
        Extract positions of keys and values in the tokenized text for masking purposes.
        Returns dictionaries mapping field names to their token positions.
        """
        key_positions = {}
        value_positions = {}
        
        # Get the tokenized sequence as text to find positions
        tokens = self.tokenizer.convert_ids_to_tokens(tokenized_text)
        
        def extract_positions_recursive(obj, parent_key="", token_idx=0):
            nonlocal key_positions, value_positions
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{parent_key}.{key}" if parent_key else key
                    
                    # Find key positions in tokens
                    key_tokens = self.tokenizer.tokenize(key)
                    key_start_idx = self._find_token_sequence(tokens, key_tokens, token_idx)
                    if key_start_idx != -1:
                        key_positions[full_key] = list(range(key_start_idx, key_start_idx + len(key_tokens)))
                        token_idx = key_start_idx + len(key_tokens)
                    
                    # Find value positions
                    if isinstance(value, (str, int, float)):
                        value_tokens = self.tokenizer.tokenize(str(value))
                        value_start_idx = self._find_token_sequence(tokens, value_tokens, token_idx)
                        if value_start_idx != -1:
                            value_positions[full_key] = list(range(value_start_idx, value_start_idx + len(value_tokens)))
                            token_idx = value_start_idx + len(value_tokens)
                    elif isinstance(value, dict):
                        token_idx = extract_positions_recursive(value, full_key, token_idx)
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                token_idx = extract_positions_recursive(item, full_key, token_idx)
                            else:
                                item_tokens = self.tokenizer.tokenize(str(item))
                                item_start_idx = self._find_token_sequence(tokens, item_tokens, token_idx)
                                if item_start_idx != -1:
                                    if full_key not in value_positions:
                                        value_positions[full_key] = []
                                    value_positions[full_key].extend(list(range(item_start_idx, item_start_idx + len(item_tokens))))
                                    token_idx = item_start_idx + len(item_tokens)
            
            return token_idx
        
        extract_positions_recursive(json_obj)
        return key_positions, value_positions
    
    def _find_token_sequence(self, tokens, target_tokens, start_idx=0):
        """
        Find the starting index of a token sequence within the tokens list.
        """
        if not target_tokens:
            return -1
            
        for i in range(start_idx, len(tokens) - len(target_tokens) + 1):
            if tokens[i:i + len(target_tokens)] == target_tokens:
                return i
        return -1
    
    def __getitem__(self, idx):
        """
        Return tokenized inputs and metadata for the collator.
        """
        json_obj = self.data[idx]

        serialized_vanilla = self._serialize_vanilla(json_obj)
        tokenized = self.tokenizer(
            serialized_vanilla, 
            max_length=self.max_length, 
            truncation=True, 
            padding="max_length",
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        
        # Extract key and value positions
        key_positions, value_positions = self._extract_key_value_positions(json_obj, tokenized["input_ids"].squeeze(0))
        if self.mode == "training":
            return {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "token_type_ids": tokenized["token_type_ids"].squeeze(0),
                "special_tokens_mask": tokenized["special_tokens_mask"].squeeze(0),
            }
        elif self.mode == "masked_prediction":
            return {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "token_type_ids": tokenized["token_type_ids"].squeeze(0),
                "special_tokens_mask": tokenized["special_tokens_mask"].squeeze(0),
                "header_positions": key_positions,
                "value_positions": value_positions,
            }

    def __len__(self):
        return len(self.data)


class HaetaeDataset(Dataset):
    def __init__(self, json_data, tokenizer_name='bert-base-uncased', max_length=512):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.data = json_data

    def _serialize(self, json_obj):
        """
        Serialize the JSON object with clear hierarchical key representation.
        """
        def serialize_recursive(obj, parent_key=""):
            parts = []
            if isinstance(obj, dict):
                parts.append("{")
                for k, v in obj.items():
                    full_key = f"{parent_key}.{k}" if parent_key else k
                    parts.append(f"{k}: {serialize_recursive(v, full_key)}")
                    parts.append(",")
                parts.append("}")
            elif isinstance(obj, list):
                parts.append("[")
                parts.append(", ".join([serialize_recursive(item, parent_key) for item in obj]))
                parts.append("]")
            else:
                parts.append(str(obj))
            return " ".join(parts)

        serialized = serialize_recursive(json_obj)
        return serialized

    def _find_key_value_positions(self, serialized, json_obj, parent_key=""):
        tokenized = self.tokenizer(
            serialized,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].squeeze(0).tolist()
        tokenized_serialized = self.tokenizer.convert_ids_to_tokens(input_ids)

        key_positions = {}
        value_positions = {}
        current_position = 1  # After [CLS]

        def recurse_json(obj, parent_key=""):
            nonlocal current_position
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{parent_key}.{key}" if parent_key else key
                    tokenized_key = self.tokenizer.tokenize(key)

                    key_start_pos = _find_token_indices(tokenized_serialized, tokenized_key, current_position)
                    if key_start_pos is not None:
                        key_positions[full_key] = key_start_pos
                    current_position = key_start_pos[-1] + 2 if key_start_pos else current_position + len(tokenized_key) + 2
                    
                    # Find value positions
                    if not isinstance(value, (dict, list)):
                        tokenized_value = self.tokenizer.tokenize(str(value))
                        value_start_pos = _find_token_indices(tokenized_serialized, tokenized_value, current_position)
                        if value_start_pos is not None:
                            value_positions[full_key] = value_start_pos
                        current_position = value_start_pos[-1] + 1 if value_start_pos else current_position + len(tokenized_value)
                    else:
                        recurse_json(value, full_key)
                    
            elif isinstance(obj, list):
                for i, element in enumerate(obj):
                    recurse_json(element, f"{parent_key}[{i}]")
                    
            else:
                current_position += len(self.tokenizer.tokenize(str(obj)))

        def _find_token_indices(sequence, tokens, start_index):
            for i in range(start_index, len(sequence) - len(tokens) + 1):
                if sequence[i:i+len(tokens)] == tokens:
                    return list(range(i, i+len(tokens)))
            return None

        recurse_json(json_obj)
        return key_positions, value_positions

    def __getitem__(self, idx):
        json_obj = self.data[idx]

        serialized = self._serialize(json_obj)
        
        tokenized = self.tokenizer(
            serialized, 
            max_length=self.max_length, 
            truncation=True, 
            padding="max_length",
            return_tensors="pt",
            return_special_tokens_mask=True,
        )

        key_positions, value_positions = self._find_key_value_positions(serialized, json_obj)

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "token_type_ids": tokenized["token_type_ids"].squeeze(0),
            "special_tokens_mask": tokenized["special_tokens_mask"].squeeze(0),
            "header_positions": key_positions,
            "value_positions": value_positions
        }

    def __len__(self):
        return len(self.data)
        

class TapasDataset(Dataset):
    def __init__(self, json_data, tokenizer_name='google/tapas-base', max_length=512, mode="training"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.data = json_data
        self.mode = mode

    def _find_key_value_positions(self, tokenized_inputs, json_obj):
        """
        Find positions of keys and values in the tokenized TAPAS input.
        Uses the original JSON keys as headers and finds their positions.
        """
        key_positions = {}
        value_positions = {}
        
        input_ids = tokenized_inputs['input_ids'].squeeze(0)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        # Find table boundaries - TAPAS uses only one [SEP]
        sep_indices = [i for i, token in enumerate(tokens) if token == '[SEP]']
        
        if len(sep_indices) < 1:
            return key_positions, value_positions
        
        # For TAPAS: question ends at first [SEP], table starts after it
        table_start = sep_indices[0] + 1
        table_end = len(tokens)  # Table goes to the end (before padding)
        
        # Use the original JSON keys as headers
        headers = list(json_obj.keys())
        
        # Find positions for each header in the tokenized text
        for header in headers:
            # Tokenize the header (e.g., "offers.price" -> ["offers", ".", "price"])
            header_tokens = self.tokenizer.tokenize(header)
            
            # Find where this header appears in the tokenized text
            header_start = self._find_token_sequence(tokens, header_tokens, table_start)
            
            if header_start is not None:
                header_end = header_start + len(header_tokens)
                key_positions[header] = list(range(header_start, header_end))
                
                # Find the corresponding value position
                value = str(json_obj[header])
                value_tokens = self.tokenizer.tokenize(value)
                
                # Search for the value in the tokenized text
                value_start = self._find_token_sequence(tokens, value_tokens, table_start)
                
                if value_start is not None:
                    value_end = value_start + len(value_tokens)
                    value_positions[header] = list(range(value_start, value_end))
                else:
                    pass
            else:
                pass
        
        return key_positions, value_positions
    
    def _find_token_sequence(self, tokens, target_tokens, start_pos):
        """
        Find the starting position of a token sequence within the tokens list.
        
        Args:
            tokens: List of tokens to search in
            target_tokens: List of tokens to find
            start_pos: Position to start searching from
            
        Returns:
            Starting index if found, None otherwise
        """
        if not target_tokens:
            return None
            
        for i in range(start_pos, len(tokens) - len(target_tokens) + 1):
            if tokens[i:i+len(target_tokens)] == target_tokens:
                return i
        return None

    def __getitem__(self, idx):
        json_obj = self.data[idx]
        instance = {key: str(json_obj[key]) for key in json_obj.keys()}
        table = pd.DataFrame([instance])

        inputs = self.tokenizer(
            table=table, 
            queries=["What is the missing value?"], 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )

        try:
            key_positions, value_positions = self._find_key_value_positions(inputs, json_obj)
        except Exception as e:
            print(f"=== _find_key_value_positions failed with error: {e} ===")
            raise e

        if self.mode == "training":
            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "token_type_ids": inputs["token_type_ids"].squeeze(0),
            }
        elif self.mode == "masked_prediction":
            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "token_type_ids": inputs["token_type_ids"].squeeze(0),
                "header_positions": key_positions,
                "value_positions": value_positions
            }

    def __len__(self):
        return len(self.data)


class TabbieDataset(Dataset):
    """
    Dataset class for Tabbie MLM training, designed to work with tabular data.
    Converts JSON objects into the format expected by Tabbie's MLM model.
    """
    def __init__(self, json_data, tokenizer_name='bert-base-uncased', max_length=512, mode="training"):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.mode = mode
        
        # Handle tuple format: (table_id, json_object)
        if json_data and isinstance(json_data[0], tuple):
            self.table_ids = [item[0] for item in json_data]
            self.data = [item[1] for item in json_data]
        else:
            # Fallback for original format
            self.table_ids = [0] * len(json_data)  # Default table_id
            self.data = json_data

    def _serialize_for_tabbie(self, json_obj):
        """
        Serialize JSON object into a format suitable for Tabbie MLM training.
        Creates a structured representation with clear header-value pairs.
        """
        serialized_parts = []
        
        for key, value in json_obj.items():
            # Handle different value types
            if isinstance(value, dict):
                # For nested objects, flatten them
                nested_str = self._serialize_nested_dict(value, key)
                serialized_parts.append(nested_str)
            elif isinstance(value, list):
                # For lists, join with commas
                list_str = f"{key} : {', '.join(str(v) for v in value)}"
                serialized_parts.append(list_str)
            else:
                # Simple key-value pair
                serialized_parts.append(f"{key} : {value}")
        
        return " [SEP] ".join(serialized_parts)

    def _serialize_nested_dict(self, obj, parent_key=""):
        """Helper to serialize nested dictionaries."""
        parts = []
        for k, v in obj.items():
            full_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                parts.append(self._serialize_nested_dict(v, full_key))
            elif isinstance(v, list):
                parts.append(f"{full_key} : {', '.join(str(item) for item in v)}")
            else:
                parts.append(f"{full_key} : {v}")
        return " | ".join(parts)

    def _extract_header_value_positions(self, json_obj, tokenized_text):
        """
        Extract positions of headers and values in the tokenized text.
        Returns dictionaries mapping field names to their token positions.
        """
        header_positions = {}
        value_positions = {}
        
        # Get the tokenized sequence as text to find positions
        tokens = self.tokenizer.convert_ids_to_tokens(tokenized_text)
        
        def find_positions_recursive(obj, parent_key="", token_idx=0):
            nonlocal header_positions, value_positions
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{parent_key}.{key}" if parent_key else key
                    
                    # Find header positions in tokens
                    header_tokens = self.tokenizer.tokenize(key)
                    header_start_idx = self._find_token_sequence(tokens, header_tokens, token_idx)
                    if header_start_idx != -1:
                        header_positions[full_key] = list(range(header_start_idx, header_start_idx + len(header_tokens)))
                        token_idx = header_start_idx + len(header_tokens)
                    
                    # Find value positions
                    if isinstance(value, (str, int, float)):
                        value_tokens = self.tokenizer.tokenize(str(value))
                        value_start_idx = self._find_token_sequence(tokens, value_tokens, token_idx)
                        if value_start_idx != -1:
                            value_positions[full_key] = list(range(value_start_idx, value_start_idx + len(value_tokens)))
                            token_idx = value_start_idx + len(value_tokens)
                    elif isinstance(value, dict):
                        token_idx = find_positions_recursive(value, full_key, token_idx)
                    elif isinstance(value, list):
                        for i, item in enumerate(value):
                            if isinstance(item, dict):
                                token_idx = find_positions_recursive(item, f"{full_key}[{i}]", token_idx)
                            else:
                                item_tokens = self.tokenizer.tokenize(str(item))
                                item_start_idx = self._find_token_sequence(tokens, item_tokens, token_idx)
                                if item_start_idx != -1:
                                    if full_key not in value_positions:
                                        value_positions[full_key] = []
                                    value_positions[full_key].extend(list(range(item_start_idx, item_start_idx + len(item_tokens))))
                                    token_idx = item_start_idx + len(item_tokens)
            
            return token_idx
        
        find_positions_recursive(json_obj)
        return header_positions, value_positions

    def _find_token_sequence(self, tokens, target_tokens, start_idx=0):
        """Find the starting index of a token sequence within the tokens list."""
        if not target_tokens:
            return -1
            
        for i in range(start_idx, len(tokens) - len(target_tokens) + 1):
            if tokens[i:i + len(target_tokens)] == target_tokens:
                return i
        return -1

    def __getitem__(self, idx):
        """
        Return tokenized inputs and metadata for Tabbie MLM training.
        """
        json_obj = self.data[idx]
        table_id = self.table_ids[idx]
        
        # Serialize the JSON object for Tabbie
        serialized_text = self._serialize_for_tabbie(json_obj)
        
        # Tokenize the serialized text
        tokenized = self.tokenizer(
            serialized_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        
        # Extract header and value positions for masking
        header_positions, value_positions = self._extract_header_value_positions(
            json_obj, tokenized["input_ids"].squeeze(0)
        )
        
        # Prepare the output based on mode
        if self.mode == "training":
            return {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "token_type_ids": tokenized["token_type_ids"].squeeze(0),
                "special_tokens_mask": tokenized["special_tokens_mask"].squeeze(0),
                "table_id": table_id,
                "raw_data": json_obj
            }
        elif self.mode == "mlm_training":
            return {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "token_type_ids": tokenized["token_type_ids"].squeeze(0),
                "special_tokens_mask": tokenized["special_tokens_mask"].squeeze(0),
                "header_positions": header_positions,
                "value_positions": value_positions,
                "table_id": table_id,
                "raw_data": json_obj
            }
        elif self.mode == "masked_prediction":
            return {
                "input_ids": tokenized["input_ids"].squeeze(0),
                "attention_mask": tokenized["attention_mask"].squeeze(0),
                "token_type_ids": tokenized["token_type_ids"].squeeze(0),
                "special_tokens_mask": tokenized["special_tokens_mask"].squeeze(0),
                "header_positions": header_positions,
                "value_positions": value_positions,
                "table_id": table_id,
                "raw_data": json_obj
            }

    def __len__(self):
        return len(self.data)