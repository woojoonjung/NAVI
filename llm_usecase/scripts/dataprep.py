import torch
import json
import pandas as pd
from torch.utils.data import Dataset
from transformers import BertTokenizer


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
                 ablation_mode="full"):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.ablation_mode = ablation_mode

        # Handle tuple format: (table_id, json_object)
        if json_data and isinstance(json_data[0], tuple):
            self.table_ids = [item[0] for item in json_data]
            self.data = [Row([Segment(k, v) for k, v in item[1].items()], ablation_mode) for item in json_data]
        else:
            # Fallback for original format
            self.table_ids = [0] * len(json_data)  # Default table_id
            self.data = [Row([Segment(k, v) for k, v in row.items()], ablation_mode) for row in json_data]

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


class BertDataset(Dataset):
    def __init__(self, json_data, tokenizer_name='bert-base-uncased', max_length=512):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.data = json_data

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
        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
            "token_type_ids": tokenized["token_type_ids"].squeeze(0),
            "special_tokens_mask": tokenized["special_tokens_mask"].squeeze(0),
        }

    def __len__(self):
        return len(self.data)