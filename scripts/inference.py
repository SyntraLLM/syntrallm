#!/usr/bin/env python3
"""Inference script for Syntra model."""

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

def main():
    model_path = "models/syntra_model"
    tokenizer_path = "models/tokenizer"

    # Load model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Generate
    prompt = "This is a test"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1, temperature=0.7)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")

if __name__ == "__main__":
    main()