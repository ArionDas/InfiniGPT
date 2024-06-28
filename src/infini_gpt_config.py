INFINIGPT_CONFIG = {
    "vocab_size": 50257,                # Vocabulary size
    "input_dim": 768,                   # Input dimension 
    "output_dim": 768,                  # Output dimension               
    "context_length": 8196,             # Context length
    "emb_dim": 128,                     # Embedding dimension
    "num_attention_heads": 8,           # Number of attention heads
    "n_layers": 12,                     # Number of layers
    "drop_rate": 0.1,                   # Dropout rate
    "qkv_bias": False,                  # Query-Key-Value bias
    "hidden_dim": 2048,                 # Hidden dimension
    "segment_len": 2048,                # Segment length
    "key_value_dim": 64,                # Key & Value dimensions
    "update": "delta",                  # Update rule
}

## segment length = 2048
## sequence length = 32768
## num of segments = 32768/2048 = 16