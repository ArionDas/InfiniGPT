INFINIGPT_CONFIG = {
    "vocab_size": 50257,                # Vocabulary size
    "input_dim": 768,                   # Input dimension 
    "output_dim": 768,                  # Output dimension               
    "context_length": 8192,             # Context length
    "stride": 2048,                     # Stride length
    "emb_dim": 128,                     # Embedding dimension
    "num_attention_heads": 8,           # Number of attention heads
    "n_layers": 12,                     # Number of layers
    "drop_rate": 0.1,                   # Dropout rate
    "qkv_bias": False,                  # Query-Key-Value bias
    "hidden_dim": 4096,                 # Hidden dimension
    "segment_len": 2048,                # Segment length
    "batch_size": 64,                   # Batch size
    "key_value_dim": 64,                # Key & Value dimensions
    "update": "delta",                  # Update rule
    "num_epochs": 10,                   # Epochs
    "drop_rate" : 0.1,                  # Dropout rate
    "learning_rate": 1e-4,              # Learning rate
    "weight_decay": 0.1,                # Weight decay
}
