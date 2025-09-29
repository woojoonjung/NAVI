import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import Config

config = Config()

# -----------------------------
# Helper functions
# -----------------------------
def get_loss_weights(masking_strategy, ablation_type="full", alignment_weight=0.15):
    # [mlm, alignment]
    if masking_strategy == 'B':
        weigths = [1, alignment_weight]
        if ablation_type == "woESA":
            weigths[1] = 0

        return weigths
    else:
        weigths = [1, alignment_weight]
        # Apply ablation masks
        if ablation_type == "woESA":
            weigths[1] = 0 

        return weigths
    

def extract_header_embeddings(embeddings, header_positions):
    header_embeds = []
    for b in range(len(header_positions)):
        emb = embeddings[b]  # (seq_len, hidden)
        for header in header_positions[b]:
            h_pos = header_positions[b][header]
            # --- FIX: SAFEGUARD AGAINST EMPTY LIST ---
            if h_pos:
                h_emb = emb[h_pos].mean(dim=0)
                header_embeds.append(h_emb)
    # This check is also important to prevent error on empty batch
    return torch.stack(header_embeds) if header_embeds else torch.empty(0)

def extract_value_embeddings(embeddings, value_positions):
    val_embeds = []
    value_to_header = []
    for b in range(len(value_positions)):
        emb = embeddings[b]  # (seq_len, hidden)
        for header in value_positions[b]:
            v_pos = value_positions[b][header]
            # --- FIX: SAFEGUARD AGAINST EMPTY LIST ---
            if v_pos:
                v_emb = emb[v_pos].mean(dim=0)
                val_embeds.append(v_emb)
                value_to_header.append(header)
    # This check is also important to prevent error on empty batch
    if not val_embeds:
        return torch.empty(0), []
    return torch.stack(val_embeds), value_to_header

def extract_header_value_embeddings(embeddings, header_positions, value_positions):
    """
    Extract both header and value embeddings from contextualized embeddings.
    """
    batch_size = len(embeddings)
    max_headers = max(len(h_pos) for h_pos in header_positions) if header_positions else 0
    hidden_size = embeddings.size(-1)
    
    header_embeds = torch.zeros((batch_size, max_headers, hidden_size), device=embeddings.device)
    val_embeds = torch.zeros((batch_size, max_headers, hidden_size), device=embeddings.device)
    
    for b in range(batch_size):
        header_pos_dict = header_positions[b]
        value_pos_dict = value_positions[b]
        
        for k, header_name in enumerate(header_pos_dict.keys()):
            if k >= max_headers: continue

            # Extract header embedding
            h_token_indices = header_pos_dict.get(header_name, [])
            # --- FIX: SAFEGUARD AGAINST EMPTY LIST ---
            if h_token_indices and all(0 <= idx < embeddings.size(1) for idx in h_token_indices):
                header_tokens = embeddings[b, h_token_indices]
                header_embeds[b, k] = header_tokens.mean(dim=0)
            
            # Extract value embedding for this specific header
            if header_name in value_pos_dict:
                v_token_indices = value_pos_dict.get(header_name, [])
                # --- FIX: SAFEGUARD AGAINST EMPTY LIST ---
                if v_token_indices and all(0 <= idx < embeddings.size(1) for idx in v_token_indices):
                    value_tokens = embeddings[b, v_token_indices]
                    val_embeds[b, k] = value_tokens.mean(dim=0)
                    
    return header_embeds, val_embeds


# -----------------------------
# Loss terms
# -----------------------------

class MLMLoss(nn.Module):
    """
    Masked Language Modeling Loss
    """
    def __init__(self, ignore_index=-100):
        super(MLMLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, labels):
        """
        Args:
            logits: (batch_size, seq_len, vocab_size)
            labels: (batch_size, seq_len), where masked positions have label, others are -100
        Returns:
            Scalar loss
        """
        vocab_size = logits.size(-1)

        return self.loss_fn(logits.view(-1, vocab_size), labels.view(-1))

# ---------------------------------------------------------------------------------------
# Entropy-aware contrastive loss
# ---------------------------------------------------------------------------------------

class EntropyAwareContrastiveLoss(nn.Module):
    """
    Entropy-aware contrastive loss.
    
    - Low-entropy headers: Universal header embedding vs contextual instances
    - High-entropy headers: Row-conditioned anchor vs value embedding
    """
    
    def __init__(self, model, low_entropy_tau=0.13, high_entropy_tau=0.07):
        super().__init__()
        self.model = model
        self.low_entropy_tau = low_entropy_tau
        self.high_entropy_tau = high_entropy_tau
    
    def _norm(self, x, eps=1e-8):
        """L2 normalize embeddings."""
        return x / (x.norm(dim=-1, keepdim=True) + eps)
    
    def _sanitize(self, t):
        """Replace NaN values with 0."""
        return torch.nan_to_num(t, nan=0.0)

    def _create_segment_embeddings(self, E_univ, H_ctx, V_ctx):
        """
        Create unified segment embeddings using the projection layer.
        
        Args:
            E_univ, H_ctx, V_ctx: (B, H, D)
        Returns:
            segment_embeddings: (B, H, D)
        """
        return self.model.create_segment_embeddings(E_univ, H_ctx, V_ctx)
    
    def _contrastive_loss(self, queries, keys, pos_mask, current_loss_type):
        """
        Simple contrastive loss computation.
        
        Args:
            queries: (N, D) query embeddings
            keys: (N, D) key embeddings  
            pos_mask: (N, N) positive pair mask
        """
        if queries.size(0) == 0:
            return torch.tensor(0.0, device=queries.device)
        
        # Compute similarities
        # Use appropriate tau based on entropy type
        tau = self.low_entropy_tau if current_loss_type == 'low_entropy' else self.high_entropy_tau
        sim = (queries @ keys.t()) / tau
        sim = self._sanitize(sim)
        
        # Ensure diagonal is positive (self-similarity)
        pos_mask = pos_mask.clone()
        pos_mask.fill_diagonal_(True)
        
        # Check for valid positive pairs
        has_positives = pos_mask.any(dim=1)
        if not has_positives.any():
            return torch.tensor(0.0, device=queries.device)
        
        # Standard contrastive loss
        exp_sim = torch.exp(sim)
        pos_sim = exp_sim * pos_mask.float()
        
        pos_sum = pos_sim.sum(dim=1)
        all_sum = exp_sim.sum(dim=1)
        
        # Avoid division by zero
        valid_mask = (pos_sum > 0) & (all_sum > 0)
        if not valid_mask.any():
            return torch.tensor(0.0, device=queries.device)
        
        loss = -torch.log(pos_sum[valid_mask] / all_sum[valid_mask])
        return loss.mean()
    
    def _low_entropy_loss(self, E_univ, H_ctx, V_ctx, header_names, low_set):
        """
        Query: Segment_embeddings
        Keys: E_univ (universal header concepts) Positives: same header, Negatives: other headers
        """
        B, H, D = E_univ.shape
        
        # Create unified segment embeddings
        segment_embeddings = self._create_segment_embeddings(E_univ, H_ctx, V_ctx)
        
        # Select low-entropy fields
        valid_items = []
        header_labels = []
        
        for b in range(B):
            for h_idx, name in enumerate(header_names[b]):
                if name in low_set and E_univ[b, h_idx].abs().sum() > 0:
                    valid_items.append((b, h_idx))
                    header_labels.append(name)
        
        if len(valid_items) < 2:
            return torch.tensor(0.0, device=E_univ.device)
        
        # Get unique headers and their universal embeddings
        unique_headers = list(set(header_labels))
        if len(unique_headers) < 2:
            return torch.tensor(0.0, device=E_univ.device)
        
        # Build keys: unique universal header embeddings
        header_to_idx = {header: i for i, header in enumerate(unique_headers)}
        keys = []
        for header in unique_headers:
            # Find first occurrence of this header to get its E_univ
            for b, h_idx in valid_items:
                if header_names[b][h_idx] == header:
                    keys.append(E_univ[b, h_idx])
                    break
        
        keys = self._norm(torch.stack(keys))  # (K, D)
        
        # Build queries: segment embeddings (rich contextual representations)
        queries = []
        query_to_header = []
        
        for b, h_idx in valid_items:
            queries.append(segment_embeddings[b, h_idx])
            query_to_header.append(header_names[b][h_idx])
        
        queries = self._norm(torch.stack(queries))  # (N, D)
        
        # Create positive mask: (N, K)
        pos_mask = torch.zeros(len(queries), len(keys), dtype=torch.bool, device=queries.device)
        for i, query_header in enumerate(query_to_header):
            j = header_to_idx[query_header]
            pos_mask[i, j] = True
        
        return self._contrastive_loss(queries, keys, pos_mask, 'low_entropy')
    
    def _high_entropy_loss(self, E_univ, H_ctx, V_ctx, header_names, high_set):
        """
        Query: Segment embeddings, 
        Keys: V_ctx, Positives: same row, Negatives: other rows
        """
        B, H, D = E_univ.shape

        # Create unified segment embeddings
        segment_embeddings = self._create_segment_embeddings(E_univ, H_ctx, V_ctx)
        
        # Select high-entropy fields
        valid_items = []
        header_labels = []
        
        for b in range(B):
            for h_idx, name in enumerate(header_names[b]):
                if (name in high_set and 
                    E_univ[b, h_idx].abs().sum() > 0 and
                    V_ctx is not None and V_ctx[b, h_idx].abs().sum() > 0):
                    valid_items.append((b, h_idx))
                    header_labels.append(name)
        
        if len(valid_items) < 2:
            return torch.tensor(0.0, device=E_univ.device)
        
        # Extract embeddings
        queries = []
        keys = []
        
        for b, h_idx in valid_items:
            # Query: Segment embedding
            queries.append(segment_embeddings[b, h_idx])
            
            # Key: Value embedding
            keys.append(V_ctx[b, h_idx])
        
        queries = self._norm(torch.stack(queries))  # (N, D)
        keys = self._norm(torch.stack(keys))        # (N, D)
        
        # Group by header type for within-header contrastive learning
        from collections import defaultdict
        header_groups = defaultdict(list)
        for i, header in enumerate(header_labels):
            header_groups[header].append(i)
        
        total_loss = 0.0
        valid_groups = 0
        
        # Compute loss within each header group
        for header, indices in header_groups.items():
            if len(indices) < 2:
                continue
            
            group_queries = queries[indices]  # (G, D)
            group_keys = keys[indices]        # (G, D)
            
            # Positive mask: diagonal (same row)
            pos_mask = torch.eye(len(indices), dtype=torch.bool, device=queries.device)
            
            group_loss = self._contrastive_loss(group_queries, group_keys, pos_mask, 'high_entropy')
            total_loss += group_loss
            valid_groups += 1
        
        return total_loss / max(valid_groups, 1)
    
    def forward(self, E_univ, H_ctx, V_ctx, header_strings, table_ids, field_categories):
        """
        Simplified forward pass.
        """
        if field_categories is None:
            return torch.tensor(0.0, device=E_univ.device)
        
        total_loss = 0.0
        valid_tables = 0
        
        # Process each table
        for t in range(len(table_ids)):
            table_id = table_ids[t]
            if table_id not in field_categories:
                continue
            
            # Get field categories for this table
            categories = field_categories[table_id]
            low_set = categories.get('low_entropy', set())
            high_set = categories.get('high_entropy', set())
            
            if not low_set and not high_set:
                continue
            
            # Extract table data
            Eu_t = E_univ[t:t+1]
            Hc_t = H_ctx[t:t+1] if H_ctx is not None else None
            Vc_t = V_ctx[t:t+1] if V_ctx is not None else None
            names_t = header_strings[t:t+1]
            
            # Compute losses
            low_loss = self._low_entropy_loss(Eu_t, Hc_t, Vc_t, names_t, low_set)
            high_loss = self._high_entropy_loss(Eu_t, Hc_t, Vc_t, names_t, high_set)
            
            # Ensure they're tensors
            if not isinstance(low_loss, torch.Tensor):
                low_loss = torch.tensor(low_loss, device=E_univ.device)
            if not isinstance(high_loss, torch.Tensor):
                high_loss = torch.tensor(high_loss, device=E_univ.device)

            # Now safe to check for NaN
            if not torch.isnan(low_loss) and not torch.isnan(high_loss):
                total_loss += (low_loss + high_loss)
                valid_tables += 1
        
        return total_loss / max(valid_tables, 1)