import os
import copy
import sys
import torch
import torch.nn as nn
from transformers import BertModel, BertForMaskedLM, BertTokenizer
from safetensors.torch import load_file
from config import Config

config = Config()

# -----------------------------
# Navi Model
# Ablation for "woSSI" processed here
# -----------------------------

class GlobalHeaderEncoder(nn.Module):
    """
    Encoder for generating  universal header embeddings.
    """
    def __init__(self, hidden_size=config.HIDDEN_SIZE, bert_name=config.BERT_NAME):
        super(GlobalHeaderEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_name)

        # Embedding layer
        self.embeddings = copy.deepcopy(self.bert.embeddings)

        # Encoding layers
        self.encoder_layer_1 = copy.deepcopy(self.bert.encoder.layer[7])  # Layer 8
        self.encoder_layer_2 = copy.deepcopy(self.bert.encoder.layer[8])   # Layer 9

    def forward(self, headers):
        """
        Encode header names into fixed embeddings.

        Args:
            headers (str | List[str] | List[List[str]]):
                - Single header string: "name"
                - Flat list of header strings: ["name", "age"]
                - Batched list of header strings: [["name", "age"], ["title"]]

        Returns:
            - If input is str: Tensor of shape (1, hidden_size)
            - If List[str]: Tensor of shape (num_headers, hidden_size)
            - If List[List[str]]: Tensor of shape (batch_size, max_headers, hidden_size), header_mask (batch_size, max_headers)
        """
        single_input = False
        flat_input = False

        if isinstance(headers, str):
            headers = [[headers]]
            single_input = True
        elif isinstance(headers, list) and all(isinstance(k, str) for k in headers):
            headers = [headers]
            flat_input = True
        elif isinstance(headers, list) and all(isinstance(k, list) for k in headers):
            pass  # batched input as usual
        else:
            raise ValueError("Invalid input format. Must be str, List[str], or List[List[str]].")

        flat_header_strings = [header for header_strings in headers for header in header_strings]
        lengths = [len(row) for row in headers]
        max_len = max(lengths)

        tokenized_headers = self.tokenizer(
            flat_header_strings,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        device = self.embeddings.word_embeddings.weight.device
        input_ids = tokenized_headers["input_ids"].to(device)
        attention_mask = tokenized_headers["attention_mask"].to(device)

        embedded_headers = self.embeddings(input_ids)
        extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        # Apply encoder layers sequentially
        hidden_states = embedded_headers
        hidden_states = self.encoder_layer_1(hidden_states, attention_mask=extended_attention_mask)[0]
        encoded_headers = self.encoder_layer_2(hidden_states, attention_mask=extended_attention_mask)[0]

        mean_pooled = (encoded_headers * attention_mask.unsqueeze(-1)).sum(dim=1) \
                    / attention_mask.sum(dim=1, keepdim=True)

        if single_input:
            return mean_pooled[0]  # shape: (hidden_size,)
        elif flat_input:
            return mean_pooled  # shape: (num_headers, hidden_size)
        else:
            # regroup into padded (B, max_len, H)
            grouped = []
            idx = 0
            for length in lengths:
                grouped.append(mean_pooled[idx:idx+length])
                idx += length

            hidden_size = mean_pooled.size(-1)
            batch_size = len(headers)
            padded_tensor = torch.zeros((batch_size, max_len, hidden_size), device=device)
            header_mask = torch.zeros((batch_size, max_len), device=device)

            for i, row in enumerate(grouped):
                padded_tensor[i, :row.size(0)] = row
                header_mask[i, :row.size(0)] = 1

            return padded_tensor, header_mask


class NaviEmbeddings(nn.Module):
    """
    Embedding module for Navi, positional embeddings & universalheader embeddings are segment-wise.
    """
    def __init__(self, embeddings, hidden_size=config.HIDDEN_SIZE, ablation_mode="full"):
        super(NaviEmbeddings, self).__init__()
        self.word_embeddings = embeddings.word_embeddings
        self.position_embeddings = embeddings.position_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = embeddings.dropout
        self.ablation_mode = ablation_mode
        self.header_encoder = GlobalHeaderEncoder()

    def forward(self, input_ids, position_ids=None, header_strings=None, segment_ids=None):
        """
        Compute embeddings.
        
        Input:
        - input_ids: Tensor of tokenized input IDs
        - position_ids: Segment-wise positional IDs
        - batched_header_strings (List[List[str]]): A batch of rows, each containing a list of header names from JSON tables.
        - segment_ids: Identifies segment-wise grouping of tokens
        """
        input_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        if header_strings is not None and segment_ids is not None:
            header_encoder_output = self.header_encoder(header_strings)

            if isinstance(header_encoder_output, tuple):
                header_embs, header_mask = header_encoder_output
            else:
                header_embs = header_encoder_output
                if header_embs.dim() == 3:
                    header_mask = torch.ones(header_embs.size()[:2], dtype=torch.bool, device=header_embs.device)
                else:
                    # Expand to (1, num_headers, H) to simulate batch dimension
                    header_embs = header_embs.unsqueeze(0)
                    header_mask = torch.ones((1, header_embs.size(1)), dtype=torch.bool, device=header_embs.device)

            batch_size, seq_len = segment_ids.size()
            max_num_headers = header_embs.size(1)
            hidden_size = header_embs.size(2)

            # Create header_embeddings lookup using segment_ids
            header_ids = segment_ids - 1  # offset for [CLS] = 0
            header_ids = header_ids.clamp(min=0)  # clamp invalid seg_id (e.g. CLS) to 0

            # Gather embeddings: (B, T, H)
            header_embeddings = torch.gather(
                header_embs.unsqueeze(1).expand(-1, seq_len, -1, -1),  # (B, T, max_num_headers, H)
                2,  # gather along header_id dim (size of max_num_headers) -- 0 dim : batch , 1 dim : header_ids[b, t]
                header_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, hidden_size)  # (B, T, 1, H)
            ).squeeze(2)

            # Mask out positions where seg_id == 0 or invalid
            valid_mask = (segment_ids > 0) & (header_ids < max_num_headers)
            valid_mask = valid_mask & header_mask.gather(1, header_ids.clamp(max=max_num_headers - 1)).bool()
            header_embeddings[~valid_mask] = 0.0
        else:
            header_embeddings = torch.zeros_like(input_embeddings)

        if self.ablation_mode == "woSSI":
            embeddings = input_embeddings + position_embeddings
        else:
            embeddings = input_embeddings + position_embeddings + header_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SegmentProjection(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size)
        )
    
    def forward(self, E_univ, H_ctx, V_ctx):
        B, H, D = E_univ.shape
        combined = torch.cat([E_univ, H_ctx, V_ctx], dim=-1)
        projected = self.projection(combined)
        return projected
        

class NaviForMaskedLM(BertForMaskedLM):
    """
    Navi Model - Header-Anchored pretraining for in-domain Tables Embedding
    Supports ablation via ablation_mode: 
    - "full": Complete model with all components
    - "woSSI": w/o schema induction
    """
    def __init__(self, model_path=None, hidden_size=config.HIDDEN_SIZE, bert_name=config.BERT_NAME, ablation_mode="full"):
        super().__init__(BertForMaskedLM.from_pretrained(bert_name).config)

        pretrained_bert = BertForMaskedLM.from_pretrained(bert_name)
        self.bert = pretrained_bert.bert
        self.cls = pretrained_bert.cls

        self.ablation_mode = ablation_mode
        self.bert.embeddings = NaviEmbeddings(pretrained_bert.bert.embeddings, hidden_size, ablation_mode)

        self.segment_projection = SegmentProjection(hidden_size)

        if model_path:
            state_dict = load_file(os.path.join(model_path, "model.safetensors"))
            self.load_state_dict(state_dict, strict=False)
            print(f"Pre-trained Navi loaded from {model_path}", file=sys.stderr)

    def create_segment_embeddings(self, E_univ, H_ctx, V_ctx):
        """
        Create unified segment embeddings using the projection layer.
        
        Args:
            E_univ: Universal header embeddings (B, H, D)
            H_ctx: Contextual header embeddings (B, H, D)  
            V_ctx: Contextual value embeddings (B, H, D)
        Returns:
            segment_embeddings: (B, H, D)
        """
        return self.segment_projection(E_univ, H_ctx, V_ctx)
    
    def forward(self, input_ids, attention_mask, position_ids=None, header_strings=None, segment_ids=None):
        """Forward pass of Navi - returns contextualized embeddings and logits"""
        
        # Get embeddings from Navi embeddings layer
        embedding_output = self.bert.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            header_strings=header_strings,
            segment_ids=segment_ids
        )
        
        # Prepare attention mask for encoder
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Pass through BERT encoder
        outputs = self.bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask
        )
        
        contextualized_embeddings = outputs[0]
        
        logits = self.cls(contextualized_embeddings)
        
        return (contextualized_embeddings, logits)