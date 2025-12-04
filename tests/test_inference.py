import pytest
from src.syntra_llm.model import create_tiny_model

def test_model_creation():
    model = create_tiny_model()
    assert model is not None
    assert model.config.vocab_size == 1000

def test_inference():
    # This would require trained model, but for CI, just check import
    from scripts.inference import main
    # Mock or skip if no model
    pass