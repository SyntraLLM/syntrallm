#!/bin/bash
# Upload to Hugging Face

# Set your HF token
# export HF_TOKEN=your_token_here

MODEL_NAME="SyntraLLM/syntra-llm"
HF_PATH="models/syntra-hf"

# Login
huggingface-cli login --token $HF_TOKEN

# Create repo if not exists
huggingface-cli repo create $MODEL_NAME --type model || echo "Repo may already exist"

# Upload
huggingface-cli upload $MODEL_NAME $HF_PATH/ --repo-type model

echo "Uploaded to https://huggingface.co/$MODEL_NAME"