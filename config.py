import os
import torch

class Config:
    """
    Configuration file
    """
    # Paths
    LOG_DIR = "logs/"
    
    # BERT (use local pretrained checkpoint under navi_private/pretrained)
    BERT_NAME = "pretrained/bert-base-uncased"
    TAPAS_NAME = "pretrained/tapas-base-masklm"
    MAX_SEQ_LENGTH = 512
    
    # Model Parameters
    HIDDEN_SIZE = 768
    NUM_ENCODER_LAYERS = 12
    NUM_ATTENTION_HEADS = 12
    DROPOUT_RATE = 0.1
    
    # Experiment settings
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Checkpoint epoch to load in experiments (must match training num_epochs / last saved epoch)
    CHECKPOINT_EPOCH = 2

    # ESA (Entropy-Aware Contrastive) internal routing ablation:
    # how eligible headers are routed to the low vs high entropy objectives.
    ESA_ROUTING_MODE = "entropy"

    def get_bert_name(self):
        return self.BERT_NAME

    def use_local_files_only(self):
        """Use local files for tokenizer/model loading (no HuggingFace Hub)."""
        return True

config = Config()