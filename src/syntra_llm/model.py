import torch
from transformers import GPT2Config, GPT2LMHeadModel

def create_tiny_model():
    """Create a tiny GPT-2 style model for quick training."""
    config = GPT2Config(
        vocab_size=1000,  # Small vocab
        n_positions=128,  # Short sequences
        n_embd=64,        # Small embedding
        n_layer=2,        # Few layers
        n_head=2,         # Few heads
    )
    model = GPT2LMHeadModel(config)
    return model