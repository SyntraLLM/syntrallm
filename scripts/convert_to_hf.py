#!/usr/bin/env python3
"""Convert trained model to Hugging Face format."""

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    model_path = "models/syntra_model"
    tokenizer_path = "models/tokenizer"
    hf_path = "models/syntra-hf"

    # Load
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Save in HF format
    model.save_pretrained(hf_path)
    tokenizer.save_pretrained(hf_path)
    print(f"Model and tokenizer saved to {hf_path}")

if __name__ == "__main__":
    main()