from tokenizers import ByteLevelBPETokenizer

def train_tokenizer(text_file, vocab_size=1000):
    """Train a BPE tokenizer on text file."""
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(files=[text_file], vocab_size=vocab_size, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
    return tokenizer