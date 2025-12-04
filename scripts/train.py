#!/usr/bin/env python3
"""Train a tiny model on synthetic data."""

import torch
from transformers import GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
from src.syntra_llm.model import create_tiny_model
from src.syntra_llm.tokenizer import train_tokenizer

def generate_synthetic_data(num_samples=100, seq_len=64):
    """Generate synthetic text data."""
    # Simple synthetic data: repeated patterns
    texts = []
    for i in range(num_samples):
        text = f"This is sample text number {i}. " * (seq_len // 10)
        texts.append(text[:seq_len])
    return texts

def main():
    # Generate data
    texts = generate_synthetic_data()
    dataset = Dataset.from_dict({"text": texts})

    # Train tokenizer if not exists
    tokenizer_path = "models/tokenizer"
    if not os.path.exists(tokenizer_path):
        print("Training tokenizer...")
        tokenizer = train_tokenizer("data/sample_text.txt")
        tokenizer.save_model(tokenizer_path)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=64)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Model
    model = create_tiny_model()
    model.resize_token_embeddings(len(tokenizer))

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args
    training_args = TrainingArguments(
        output_dir="models/checkpoints",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=50,
        logging_steps=10,
        save_total_limit=1,
        evaluation_strategy="no",
        learning_rate=5e-4,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train
    trainer.train()
    trainer.save_model("models/syntra_model")
    print("Model trained and saved to models/syntra_model")

if __name__ == "__main__":
    import os
    main()