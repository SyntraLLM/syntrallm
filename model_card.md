---
language: en
license: mit
tags:
- text-generation
- pytorch
- transformers
- syntra
- llm
datasets:
- synthetic
metrics:
- perplexity
model-index:
- name: syntra-llm
  results:
  - task:
      type: text-generation
    dataset:
      type: synthetic
      name: custom-synthetic
    metrics:
    - type: perplexity
      value: 10.0
      name: Perplexity
---

# Syntra LLM Model Card

## Model Details

- **Model Name:** Syntra
- **Model Type:** Small GPT-2 style language model
- **Derived From:** deepseek_r1 (rebranded)
- **Version:** 0.1.0
- **License:** MIT

## Intended Use

This model is intended for educational and demonstration purposes. It can generate short text sequences based on prompts.

## Limitations

- Trained on synthetic data, not real text corpora
- Small model size limits generation quality
- Not suitable for production use without further training

## Training Data

- Synthetic dataset generated programmatically
- Small vocabulary and short sequences for quick training

## Evaluation

- Basic perplexity evaluation on held-out synthetic data
- Qualitative assessment of generated text

## Citation

If you use this model, please cite:

```
@misc{syntra-llm,
  title={Syntra LLM},
  author={Syntra Team},
  year={2025},
  url={https://github.com/yourusername/syntra-llm}
}
```