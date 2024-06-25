INFINIGPT_CONFIG = {
    "vocab_size": 50257,                # Vocabulary size
    "context_length": 1024,             # Context length
    "emb_dim": 128,                     # Embedding dimension
    "num_attention_heads": 8,           # Number of attention heads
    "n_layers": 12,                     # Number of layers
    "drop_rate": 0.1,                   # Dropout rate
    "qkv_bias": False,                  # Query-Key-Value bias
    "hidden_size": 4096,                # Hidden size
}

## segment length = 2048
## sequence length = 32768
## num of segments = 32768/2048 = 16