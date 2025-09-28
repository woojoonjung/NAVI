import os
import torch

class Config:
    """
    Configuration file for HAETAE project.
    """
    # Paths
    LOG_DIR = "logs/"
    
    # BERT
    BERT_NAME = "bert-base-uncased"
    MAX_SEQ_LENGTH = 512
    
    # Model Parameters
    HIDDEN_SIZE = 768
    NUM_ENCODER_LAYERS = 12
    NUM_ATTENTION_HEADS = 12
    DROPOUT_RATE = 0.1
    
    # Experiment settings
    SEED = 42
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()