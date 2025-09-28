from dataclasses import dataclass
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import DataCollatorForLanguageModeling
import random
from itertools import chain
from dataset.dataset import Row


@dataclass
class NaviCollator(DataCollatorForLanguageModeling):
    """
    General collator for Navi pretraining that supports multiple masking strategies
    across different training stages with flexible stage configurations.
    Also supports two-view entity distinction augmentation.
    """
    stage_config: list = None  # List of tuples: [(strategy, num_epochs), ...]
    token_length_threshold: int = 8
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    ablation_mode: str = "full"
    hv_weight: float = 0.5
    value_ratio: float = 0.5   
    max_tokens_to_mask: int = 80 
    

    def __init__(self, tokenizer, mlm_probability=0.15, **kwargs):
        super().__init__(tokenizer, mlm_probability=mlm_probability)
        self.current_epoch = 0

        # Stage config controls only masking behavior (no augmentation anymore)
        if self.stage_config is None:
            self.stage_config = [('HV', 1), ('B', 1), ('HVB', 1)]

        # Validate stage configuration
        if not isinstance(self.stage_config, list) or len(self.stage_config) == 0:
            raise ValueError("stage_config must be a non-empty list of (strategy, epochs) tuples")
        for stage in self.stage_config:
            if not (isinstance(stage, tuple) and len(stage) == 2 and isinstance(stage[0], str) and isinstance(stage[1], int) and stage[1] > 0):
                raise ValueError("Each stage must be a tuple (strategy:str, epochs:int>0)")

        # Set additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    # -------- Stage helpers (kept for masking strategy scheduling) --------
    def set_epoch(self, current_epoch: int):
        self.current_epoch = current_epoch

    def get_current_strategy(self):
        cumulative_epochs = 0
        for strategy, num_epochs in self.stage_config:
            if self.current_epoch < cumulative_epochs + num_epochs:
                return strategy
            cumulative_epochs += num_epochs
        return self.stage_config[-1][0]

    def get_stage_info(self):
        cumulative_epochs = 0
        current_stage_idx = 0
        for i, (strategy, num_epochs) in enumerate(self.stage_config):
            if self.current_epoch < cumulative_epochs + num_epochs:
                current_stage_idx = i
                break
            cumulative_epochs += num_epochs
        return {
            'current_stage': current_stage_idx + 1,
            'total_stages': len(self.stage_config),
            'current_strategy': self.stage_config[current_stage_idx][0],
            'epochs_in_stage': self.current_epoch - cumulative_epochs + 1,
            'total_epochs_in_stage': self.stage_config[current_stage_idx][1]
        }

    # ------------------------------- Main call -------------------------------
    def __call__(self, examples):
        # Pack raw fields
        batch = {
            "input_ids": pad_sequence([e["input_ids"] for e in examples], batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "attention_mask": pad_sequence([e["attention_mask"] for e in examples], batch_first=True, padding_value=0),
            "position_ids": pad_sequence([e["position_ids"] for e in examples], batch_first=True, padding_value=0),
            "segment_ids": pad_sequence([e["segment_ids"] for e in examples], batch_first=True, padding_value=0),
            "header_positions": [e["header_positions"] for e in examples],
            "value_positions": [e["value_positions"] for e in examples],
            "header_strings": [e["header_strings"] for e in examples],
            "table_ids": [e["table_id"] for e in examples],
        }

        # Keep an unmasked copy for contextual embedding extraction
        batch["unmasked_input_ids"] = batch["input_ids"].clone()

        # Build special tokens mask for MLM masking
        special_tokens_mask = torch.stack([
            torch.tensor(self.tokenizer.get_special_tokens_mask(
                val.tolist(), already_has_special_tokens=True
            )) for val in batch["input_ids"]
        ]).bool()

        # Apply masking according to current stage strategy
        input_ids, labels = self.torch_mask_tokens(
            batch["input_ids"].clone(),
            special_tokens_mask=special_tokens_mask,
            header_positions=batch["header_positions"],
            value_positions=batch["value_positions"],
            segment_ids=batch["segment_ids"]
        )

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["loss_mask"] = (labels != -100).long()
        return batch

    # --------------------------- Masking strategies ---------------------------
    def torch_mask_tokens(self, inputs, special_tokens_mask=None, header_positions=None, value_positions=None, segment_ids=None):
        labels = inputs.clone()
        current_strategy = self.get_current_strategy()
        probability_matrix = torch.zeros_like(labels, dtype=torch.float)

        if current_strategy == 'B':
            self._apply_bert_masking(probability_matrix)
        elif current_strategy == 'HV':
            _, _ = self._apply_hv_masking(
                probability_matrix, header_positions, value_positions, self.max_tokens_to_mask
            )
        elif current_strategy == 'HVB':
            # Calculate token allocation based on hv_weight
            hv_tokens = int(self.max_tokens_to_mask * self.hv_weight)
            bert_tokens = self.max_tokens_to_mask - hv_tokens
            
            # Apply HV masking with allocated tokens
            selected_segments_per_batch, actual_hv_tokens_used = self._apply_hv_masking(
                probability_matrix, header_positions, value_positions, hv_tokens
            )
            
            # Calculate remaining budget for BERT masking
            remaining_budget_per_batch = [hv_tokens - used for used in actual_hv_tokens_used]
            
            # Apply BERT masking with remaining budget
            self._apply_bert_masking_with_remaining_budget(
                probability_matrix, special_tokens_mask, selected_segments_per_batch, 
                segment_ids, remaining_budget_per_batch
            )
        else:
            raise ValueError(f"Unknown strategy: {current_strategy}")

        # Sample and build labels
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 80% -> [MASK], 10% -> random, 10% -> original
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, self.mask_replace_prob)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, self.random_replace_prob)).bool()
            & masked_indices & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def _apply_bert_masking(self, probability_matrix):
        probability_matrix.fill_(self.mlm_probability)

    def _apply_hv_masking(self, probability_matrix, header_positions, value_positions, max_tokens):
        """Mask segments by selecting k segments for header masking and k segments for value masking.
        Returns the actual tokens used and remaining budget for BERT masking."""
        selected_segments_per_batch = []
        actual_tokens_used = []
        
        for batch_idx, (h_pos, v_pos) in enumerate(zip(header_positions, value_positions)):
            if not h_pos:
                selected_segments_per_batch.append(set())
                actual_tokens_used.append(0)
                continue

            # Calculate budget allocation for HV masking only
            # Each segment contributes at most token_length_threshold tokens
            max_segments = max_tokens // self.token_length_threshold
            
            # Split budget using configurable ratios
            k_value = int(max_segments * self.value_ratio)
            k_header = max_segments - k_value  # Remaining for headers
            
            # Get all available segments (header-value pairs)
            all_segments = list(h_pos.keys())
            
            # Randomly select k_value segments for value masking (prioritize values)
            num_value_segments = min(k_value, len(all_segments))
            selected_value_segments = random.sample(all_segments, num_value_segments)
            
            # Randomly select k_header segments for header masking (from remaining segments)
            remaining_segments = [seg for seg in all_segments if seg not in selected_value_segments]
            num_header_segments = min(k_header, len(remaining_segments))
            selected_header_segments = random.sample(remaining_segments, num_header_segments)
            
            # Track all selected segments for this batch
            all_selected_segments = set(selected_header_segments) | set(selected_value_segments)
            selected_segments_per_batch.append(all_selected_segments)
            
            # Apply header masking and count tokens
            header_tokens_used = 0
            for segment in selected_header_segments:
                token_positions = h_pos[segment]
                if token_positions:
                    # Apply token_length_threshold
                    if len(token_positions) <= self.token_length_threshold:
                        selected_tokens = token_positions
                    else:
                        selected_tokens = random.sample(token_positions, self.token_length_threshold)
                    
                    for pos in selected_tokens:
                        probability_matrix[batch_idx, pos] = 1.0
                        header_tokens_used += 1
            
            # Apply value masking and count tokens
            value_tokens_used = 0
            for segment in selected_value_segments:
                token_positions = v_pos.get(segment, [])
                if token_positions:
                    # Apply token_length_threshold
                    if len(token_positions) <= self.token_length_threshold:
                        selected_tokens = token_positions
                    else:
                        selected_tokens = random.sample(token_positions, self.token_length_threshold)
                    
                    for pos in selected_tokens:
                        probability_matrix[batch_idx, pos] = 1.0
                        value_tokens_used += 1
            
            # Track actual tokens used for this batch
            total_tokens_used = header_tokens_used + value_tokens_used
            actual_tokens_used.append(total_tokens_used)

        return selected_segments_per_batch, actual_tokens_used

    def _apply_bert_masking_with_remaining_budget(self, probability_matrix, special_tokens_mask, 
                                                   selected_segments_per_batch, segment_ids, remaining_budget_per_batch):
        """Apply BERT masking using the remaining budget from HV masking."""
        for batch_idx, remaining_budget in enumerate(remaining_budget_per_batch):
            if remaining_budget <= 0:
                continue
                
            # Get available positions (excluding special tokens and HV-masked segments)
            available_positions = ~special_tokens_mask[batch_idx]
            batch_segment_ids = segment_ids[batch_idx]
            
            # Exclude tokens from HV-masked segments
            for segment_id in selected_segments_per_batch[batch_idx]:
                hv_mask = (batch_segment_ids == segment_id)
                available_positions = available_positions & ~hv_mask
                
            available_indices = torch.where(available_positions)[0]
            
            # Select exactly remaining_budget tokens
            tokens_to_mask = min(remaining_budget, len(available_indices))
            if tokens_to_mask > 0:
                selected_indices = available_indices[torch.randperm(len(available_indices))[:tokens_to_mask]]
                probability_matrix[batch_idx, selected_indices] = 1.0


@dataclass
class CollatorForMaskedPrediction(DataCollatorForLanguageModeling):
    mask_keys_only: bool = True
    word_level_mask_fraction: float = 0.25
    token_length_threshold: int = 8
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    field_targeting_epochs: int = 8

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer)
        self.current_epoch = 0
        self.field_target_mode = True
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_epoch(self, current_epoch: int):
        self.current_epoch = current_epoch
        if current_epoch < self.field_targeting_epochs:
            self.field_target_mode = True
            self.mask_keys_only = (current_epoch < self.field_targeting_epochs // 2)
        else:
            self.field_target_mode = False
            self.mask_keys_only = False 

    def __call__(self, examples):
        batch = {
            "input_ids": pad_sequence([e["input_ids"] for e in examples], batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "attention_mask": pad_sequence([e["attention_mask"] for e in examples], batch_first=True, padding_value=0),
            "header_positions": [e.get("header_positions", None) for e in examples],
            "value_positions": [e.get("value_positions", None) for e in examples]
        }

        if any(e.get("header_strings") is not None for e in examples):
            batch["header_strings"] = [e.get("header_strings", None) for e in examples]

        # Only add position_ids and segment_ids if they exist in the examples
        if any(e.get("position_ids") is not None for e in examples):
            batch["position_ids"] = pad_sequence([e.get("position_ids", None) for e in examples], batch_first=True, padding_value=0)
        
        if any(e.get("segment_ids") is not None for e in examples):
            batch["segment_ids"] = pad_sequence([e.get("segment_ids", None) for e in examples], batch_first=True, padding_value=0)


        # Build special tokens mask if not already included
        special_tokens_mask = torch.stack([
            torch.tensor(self.tokenizer.get_special_tokens_mask(
                val.tolist(), already_has_special_tokens=True
            )) for val in batch["input_ids"]
        ]).bool()

        # Apply token masking
        input_ids, labels = self.torch_mask_tokens(
            batch["input_ids"].clone(),
            special_tokens_mask=special_tokens_mask,
            header_positions=batch["header_positions"],
            value_positions=batch["value_positions"]
        )

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["loss_mask"] = (labels != -100).long()

        return batch

    def torch_mask_tokens(self, inputs, special_tokens_mask=None, header_positions=None, value_positions=None):
        labels = inputs.clone()

        def apply_word_level_mask(batch_idx, field_dict, threshold):
            if not field_dict:
                return
            field_names = list(field_dict.keys())
            num_to_mask = max(1, int(len(field_names) * self.word_level_mask_fraction))
            selected_fields = random.sample(field_names, num_to_mask)

            for field in selected_fields:
                positions = field_dict[field]
                if not positions:
                    continue
                if len(positions) <= threshold:
                    for pos in positions:
                        probability_matrix[batch_idx, pos] = 1.0
                else:
                    sampled_positions = random.sample(positions, threshold)
                    for pos in sampled_positions:
                        probability_matrix[batch_idx, pos] = 1.0

        # Init masking matrix
        if self.field_target_mode and header_positions and value_positions:
            probability_matrix = torch.zeros_like(labels, dtype=torch.float)
            for batch_idx, (k_pos, v_pos) in enumerate(zip(header_positions, value_positions)):
                if self.mask_keys_only:
                    apply_word_level_mask(batch_idx, k_pos, self.token_length_threshold)
                else:
                    apply_word_level_mask(batch_idx, v_pos, self.token_length_threshold)
        else:
            probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # Final masking decisions
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Loss only on masked tokens

        # Replace with [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, self.mask_replace_prob)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # Replace with random tokens
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, self.random_replace_prob)).bool()
            & masked_indices & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

@dataclass
class TapasCollator(DataCollatorForLanguageModeling):
    """
    Collator specifically designed for TAPAS training that handles missing fields gracefully.
    Compatible with TapasDataset which only provides basic fields.
    """
    stage_config: list = None  # List of tuples: [(strategy, num_epochs), ...]
    token_length_threshold: int = 8
    mask_replace_prob: float = 0.8
    random_replace_prob: float = 0.1
    ablation_mode: str = "full"
    hv_weight: float = 0.5
    value_ratio: float = 0.5   
    max_tokens_to_mask: int = 80 
    

    def __init__(self, tokenizer, mlm_probability=0.15, **kwargs):
        super().__init__(tokenizer, mlm_probability=mlm_probability)
        self.current_epoch = 0

        # Stage config controls only masking behavior (no augmentation anymore)
        if self.stage_config is None:
            self.stage_config = [('HV', 1), ('B', 1), ('HVB', 1)]

        # Validate stage configuration
        if not isinstance(self.stage_config, list) or len(self.stage_config) == 0:
            raise ValueError("stage_config must be a non-empty list of (strategy, epochs) tuples")
        for stage in self.stage_config:
            if not (isinstance(stage, tuple) and len(stage) == 2 and isinstance(stage[0], str) and isinstance(stage[1], int) and stage[1] > 0):
                raise ValueError("Each stage must be a tuple (strategy:str, epochs:int>0)")

        # Set additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)

    # -------- Stage helpers (kept for masking strategy scheduling) --------
    def set_epoch(self, current_epoch: int):
        self.current_epoch = current_epoch

    def get_current_strategy(self):
        cumulative_epochs = 0
        for strategy, num_epochs in self.stage_config:
            if self.current_epoch < cumulative_epochs + num_epochs:
                return strategy
            cumulative_epochs += num_epochs
        return self.stage_config[-1][0]

    def get_stage_info(self):
        cumulative_epochs = 0
        current_stage_idx = 0
        for i, (strategy, num_epochs) in enumerate(self.stage_config):
            if self.current_epoch < cumulative_epochs + num_epochs:
                current_stage_idx = i
                break
            cumulative_epochs += num_epochs
        return {
            'current_stage': current_stage_idx + 1,
            'total_stages': len(self.stage_config),
            'current_strategy': self.stage_config[current_stage_idx][0],
            'epochs_in_stage': self.current_epoch - cumulative_epochs + 1,
            'total_epochs_in_stage': self.stage_config[current_stage_idx][1]
        }

    # ------------------------------- Main call -------------------------------
    def __call__(self, examples):
        # Pack basic fields that are always available
        batch = {
            "input_ids": pad_sequence([e["input_ids"] for e in examples], batch_first=True, padding_value=self.tokenizer.pad_token_id),
            "attention_mask": pad_sequence([e["attention_mask"] for e in examples], batch_first=True, padding_value=0),
        }
        
        # Add token_type_ids if available (TAPAS provides this)
        if any(e.get("token_type_ids") is not None for e in examples):
            batch["token_type_ids"] = pad_sequence([e.get("token_type_ids", None) for e in examples], batch_first=True, padding_value=0)
        
        # Only add position_ids and segment_ids if they exist in the examples
        if any(e.get("position_ids") is not None for e in examples):
            batch["position_ids"] = pad_sequence([e.get("position_ids", None) for e in examples], batch_first=True, padding_value=0)
        
        if any(e.get("segment_ids") is not None for e in examples):
            batch["segment_ids"] = pad_sequence([e.get("segment_ids", None) for e in examples], batch_first=True, padding_value=0)
        
        # Only add header_positions and value_positions if they exist in the examples
        if any(e.get("header_positions") is not None for e in examples):
            batch["header_positions"] = [e.get("header_positions", None) for e in examples]
        
        if any(e.get("value_positions") is not None for e in examples):
            batch["value_positions"] = [e.get("value_positions", None) for e in examples]
        
        # Only add header_strings if they exist in the examples
        if any(e.get("header_strings") is not None for e in examples):
            batch["header_strings"] = [e.get("header_strings", None) for e in examples]
        
        # Only add table_ids if they exist in the examples
        if any(e.get("table_id") is not None for e in examples):
            batch["table_ids"] = [e.get("table_id", None) for e in examples]

        # Keep an unmasked copy for contextual embedding extraction
        batch["unmasked_input_ids"] = batch["input_ids"].clone()

        # Build special tokens mask for MLM masking
        special_tokens_mask = torch.stack([
            torch.tensor(self.tokenizer.get_special_tokens_mask(
                val.tolist(), already_has_special_tokens=True
            )) for val in batch["input_ids"]
        ]).bool()

        # Apply masking according to current stage strategy
        input_ids, labels = self.torch_mask_tokens(
            batch["input_ids"].clone(),
            special_tokens_mask=special_tokens_mask,
            header_positions=batch.get("header_positions"),
            value_positions=batch.get("value_positions"),
            segment_ids=batch.get("segment_ids")
        )

        batch["input_ids"] = input_ids
        batch["labels"] = labels
        batch["loss_mask"] = (labels != -100).long()
        return batch

    # --------------------------- Masking strategies ---------------------------
    def torch_mask_tokens(self, inputs, special_tokens_mask=None, header_positions=None, value_positions=None, segment_ids=None):
        labels = inputs.clone()
        current_strategy = self.get_current_strategy()
        probability_matrix = torch.zeros_like(labels, dtype=torch.float)

        if current_strategy == 'B':
            self._apply_bert_masking(probability_matrix)
        elif current_strategy == 'HV':
            # Only apply HV masking if we have the required fields
            if header_positions and value_positions:
                _, _ = self._apply_hv_masking(
                    probability_matrix, header_positions, value_positions, self.max_tokens_to_mask
                )
            else:
                # Fallback to BERT masking if HV fields are not available
                self._apply_bert_masking(probability_matrix)
        elif current_strategy == 'HVB':
            # Only apply HVB masking if we have the required fields
            if header_positions and value_positions:
                # Calculate token allocation based on hv_weight
                hv_tokens = int(self.max_tokens_to_mask * self.hv_weight)
                bert_tokens = self.max_tokens_to_mask - hv_tokens
                
                # Apply HV masking with allocated tokens
                selected_segments_per_batch, actual_hv_tokens_used = self._apply_hv_masking(
                    probability_matrix, header_positions, value_positions, hv_tokens
                )
                
                # Calculate remaining budget for BERT masking
                remaining_budget_per_batch = [hv_tokens - used for used in actual_hv_tokens_used]
                
                # Apply BERT masking with remaining budget
                self._apply_bert_masking_with_remaining_budget(
                    probability_matrix, special_tokens_mask, selected_segments_per_batch, 
                    segment_ids, remaining_budget_per_batch
                )
            else:
                # Fallback to BERT masking if HV fields are not available
                self._apply_bert_masking(probability_matrix)
        else:
            raise ValueError(f"Unknown strategy: {current_strategy}")

        # Sample and build labels
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 80% -> [MASK], 10% -> random, 10% -> original
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, self.mask_replace_prob)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, self.random_replace_prob)).bool()
            & masked_indices & ~indices_replaced
        )
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def _apply_bert_masking(self, probability_matrix):
        probability_matrix.fill_(self.mlm_probability)

    def _apply_hv_masking(self, probability_matrix, header_positions, value_positions, max_tokens):
        """Mask segments by selecting k segments for header masking and k segments for value masking.
        Returns the actual tokens used and remaining budget for BERT masking."""
        selected_segments_per_batch = []
        actual_tokens_used = []
        
        for batch_idx, (h_pos, v_pos) in enumerate(zip(header_positions, value_positions)):
            if not h_pos:
                selected_segments_per_batch.append(set())
                actual_tokens_used.append(0)
                continue

            # Calculate budget allocation for HV masking only
            # Each segment contributes at most token_length_threshold tokens
            max_segments = max_tokens // self.token_length_threshold
            
            # Split budget using configurable ratios
            k_value = int(max_segments * self.value_ratio)
            k_header = max_segments - k_value  # Remaining for headers
            
            # Get all available segments (header-value pairs)
            all_segments = list(h_pos.keys())
            
            # Randomly select k_value segments for value masking (prioritize values)
            num_value_segments = min(k_value, len(all_segments))
            selected_value_segments = random.sample(all_segments, num_value_segments)
            
            # Randomly select k_header segments for header masking (from remaining segments)
            remaining_segments = [seg for seg in all_segments if seg not in selected_value_segments]
            num_header_segments = min(k_header, len(remaining_segments))
            selected_header_segments = random.sample(remaining_segments, num_header_segments)
            
            # Track all selected segments for this batch
            all_selected_segments = set(selected_header_segments) | set(selected_value_segments)
            selected_segments_per_batch.append(all_selected_segments)
            
            # Apply header masking and count tokens
            header_tokens_used = 0
            for segment in selected_header_segments:
                token_positions = h_pos[segment]
                if token_positions:
                    # Apply token_length_threshold
                    if len(token_positions) <= self.token_length_threshold:
                        selected_tokens = token_positions
                    else:
                        selected_tokens = random.sample(token_positions, self.token_length_threshold)
                    
                    for pos in selected_tokens:
                        probability_matrix[batch_idx, pos] = 1.0
                        header_tokens_used += 1
            
            # Apply value masking and count tokens
            value_tokens_used = 0
            for segment in selected_value_segments:
                token_positions = v_pos.get(segment, [])
                if token_positions:
                    # Apply token_length_threshold
                    if len(token_positions) <= self.token_length_threshold:
                        selected_tokens = token_positions
                    else:
                        selected_tokens = random.sample(token_positions, self.token_length_threshold)
                    
                    for pos in selected_tokens:
                        probability_matrix[batch_idx, pos] = 1.0
                        value_tokens_used += 1
            
            # Track actual tokens used for this batch
            total_tokens_used = header_tokens_used + value_tokens_used
            actual_tokens_used.append(total_tokens_used)

        return selected_segments_per_batch, actual_tokens_used

    def _apply_bert_masking_with_remaining_budget(self, probability_matrix, special_tokens_mask, 
                                                   selected_segments_per_batch, segment_ids, remaining_budget_per_batch):
        """Apply BERT masking using the remaining budget from HV masking."""
        for batch_idx, remaining_budget in enumerate(remaining_budget_per_batch):
            if remaining_budget <= 0:
                continue
                
            # Get available positions (excluding special tokens and HV-masked segments)
            available_positions = ~special_tokens_mask[batch_idx]
            
            # Only exclude HV-masked segments if segment_ids are available
            if segment_ids is not None:
                batch_segment_ids = segment_ids[batch_idx]
                # Exclude tokens from HV-masked segments
                for segment_id in selected_segments_per_batch[batch_idx]:
                    hv_mask = (batch_segment_ids == segment_id)
                    available_positions = available_positions & ~hv_mask
                
            available_indices = torch.where(available_positions)[0]
            
            # Select exactly remaining_budget tokens
            tokens_to_mask = min(remaining_budget, len(available_indices))
            if tokens_to_mask > 0:
                selected_indices = available_indices[torch.randperm(len(available_indices))[:tokens_to_mask]]
                probability_matrix[batch_idx, selected_indices] = 1.0