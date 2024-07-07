import torch
import torch.nn as nn
import requests

import tiktoken
from torch.utils.data import Dataset, DataLoader
from data_preprocessing import InfiniGPTDataset, InfiniGPTDataLoader
from infini_transformer import InfiniTransformer
from infini_gpt_config import INFINIGPT_CONFIG


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x-mean) / torch.sqrt(var + self.eps)
        
        return self.scale * norm_x + self.shift


class InfiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.position_embeddings = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])
        
        self.infini_transformer_blocks = nn.Sequential(
            *[InfiniTransformer(config["emb_dim"], config["hidden_dim"], config["key_value_dim"], config["key_value_dim"], config["num_attention_heads"], "relu", config["segment_len"], config["update"], False, None, True, config["drop_rate"]) for _ in range(config["n_layers"])]
        )
        
        self.final_norm = LayerNorm(config["emb_dim"])
        self.out_head = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)
        
    def forward(self, context):
        batch_size, seq_len = context.shape
        token_embeddings = self.token_embeddings(context)
        position_embeddings = self.position_embeddings(torch.arange(seq_len, device=context.device))
        
        x = token_embeddings + position_embeddings
        x = self.drop_emb(x)
        x = self.infini_transformer_blocks(x)
        x = self.final_norm(x)
        
        logits = self.out_head(x)
        return logits
    
def generate_text_simple(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):

        # Crop current context if it exceeds the supported context length
        idx_cond = idx[:, -context_size:]

        # Get the predictions
        with torch.no_grad():
            logits = model(idx_cond)

        # Focus only on the last time step
        # (batch, n_token, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        # Get the idx of the vocab entry with the highest logits value
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx
        
def main():
    
    ### Hyperparameters
    config = INFINIGPT_CONFIG
    vocab_size = config["vocab_size"]
    input_dim = config["input_dim"]
    embedding_dim = config["emb_dim"]
    output_dim = config["output_dim"]
    context_length = config["context_length"]
    segment_length = config["segment_len"]
    num_heads = config["num_attention_heads"]
    num_layers = config["n_layers"]
    key_value_dim = config["key_value_dim"]
    dim_hidden = config["hidden_dim"]
    drop_rate = config["drop_rate"]
    
    torch.manual_seed(123)
    model = InfiniGPT(config)
    model.eval()
    
    start_context = "Hello, I like"    
    
    tokenizer = tiktoken.get_encoding("gpt2")
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=config["context_length"]
    )
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    
    print(decoded_text)
    
    """
    dataloader = InfiniGPTDataLoader(txt, batch_size=8, max_length=context_length, stride=context_length)
    input_embeddings = torch.Tensor
    for batch in dataloader:
        x, y = batch
        
        token_embeddings = nn.Embedding(vocab_size, embedding_dim)
        position_embeddings = nn.Embedding(context_length, embedding_dim)
        
        input_embeddings = token_embeddings + position_embeddings
    
    infini_transformer = InfiniGPT(embedding_dim, dim_hidden, key_value_dim, key_value_dim, num_heads, "relu", segment_length, "delta", False, None, True, 0.1)
    
    batch = input_embeddings
    print("Batch shape ->", batch.shape)
    context_vecs = infini_transformer(batch)
    print("Context vector shape ->", context_vecs.shape)
    print(context_vecs)
    """
    
if __name__ == '__main__':
    main()
    
## doc Q/A - prototype
## increase acc