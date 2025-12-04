#!/usr/bin/env python3
"""Train tokenizer on sample data."""

import os
from src.syntra_llm.tokenizer import train_tokenizer

def main():
    data_file = "data/sample_text.txt"
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found.")
        return

    tokenizer = train_tokenizer(data_file)
    tokenizer.save_model("models/tokenizer")
    print("Tokenizer trained and saved to models/tokenizer")

if __name__ == "__main__":
    main()