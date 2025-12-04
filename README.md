# Syntra LLM
![Syntra banner](public/syntra-banner.jpg)
A small, production-ready LLM project. This monorepo provides a complete setup for training, inference, and publishing to Hugging Face Model Hub.

## Features

- Tiny GPT-2 style model training on synthetic data
- BPE tokenizer training
- Inference script
- Hugging Face integration
- Docker containerization
- CI/CD with GitHub Actions

## Quick Start

### Prerequisites

- Python 3.10+
- Git
- (Optional) Docker for containerized inference

### Installation

1. Clone the repo:
   ```bash
   git clone https://github.com/SyntraLLM/syntrallm.git
   cd syntrallm
   ```

2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

### Training a Model

1. Train tokenizer:
   ```bash
   python scripts/train_tokenizer.py
   ```

2. Train model:
   ```bash
   python scripts/train.py
   ```

3. Convert to Hugging Face format:
   ```bash
   python scripts/convert_to_hf.py
   ```

### Inference

Run inference:
```bash
python scripts/inference.py
```

Or start the server:
```bash
python scripts/inference_server.py
```

Or using Docker:
```bash
docker-compose up
```

API endpoint: POST /generate with JSON {"prompt": "your prompt", "max_length": 50, "temperature": 0.7}

### Upload to Hugging Face

1. Set up Hugging Face account and get token from https://huggingface.co/settings/tokens

2. Run upload script:
   ```bash
   export HF_TOKEN=your_token_here
   bash scripts/upload_to_hf.sh
   ```

### Download Pre-trained Model

The trained Syntra model is available on Hugging Face at [SyntraLLM/syntra-llm](https://huggingface.co/SyntraLLM/syntra-llm).

To download and use the model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "SyntraLLM/syntra-llm"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example inference
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

## Project Structure

```
syntrallm/
├── src/
│   └── syntra_llm/
│       ├── __init__.py
│       ├── model.py
│       └── tokenizer.py
├── scripts/
│   ├── train.py
│   ├── inference.py
│   ├── train_tokenizer.py
│   ├── convert_to_hf.py
│   └── upload_to_hf.sh
├── data/
│   └── sample_text.txt
├── models/
├── tests/
│   └── test_inference.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── Dockerfile
├── docker-compose.yml
├── model_card.md
├── pyproject.toml
├── README.md
└── LICENSE
```

## Version

v0.1.0

## License

See LICENSE file.
