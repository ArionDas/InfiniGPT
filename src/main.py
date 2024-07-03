import os
import requests

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_preprocessing import InfiniGPTDataset, InfiniGPTDataLoader
from infini_transformer import InfiniTransformer
from infini_gpt_config import INFINIGPT_CONFIG


def main():
    
    ### Hyperparameters
    config = INFINIGPT_CONFIG
    vocab_size = config["vocab_size"]
    input_dim = config["input_dim"]
    embedding_dim = config["emb_dim"]
    output_dim = config["output_dim"]
    context_length = config["context_length"]
    num_heads = config["num_attention_heads"]
    num_layers = config["n_layers"]
    key_value_dim = config["key_value_dim"]
    dim_hidden = config["hidden_dim"]
    drop_rate = config["drop_rate"]
    
    ### Embeddings
    token_embedding_layer = nn.Embedding(vocab_size, embedding_dim)
    position_embedding_layer = nn.Embedding(context_length, embedding_dim)

    url = "https://github.com/ArionDas/InfiniGPT/blob/eb3abdc6eaf1d8f17be7e92e81fd641a710aae26/data/book_clean.txt"
    response = requests.get(url)
    txt = response.text
    
    dataloader = InfiniGPTDataLoader(txt, batch_size=8, max_length=context_length, stride=context_length)
    input_embeddings = torch.Tensor
    for batch in dataloader:
        x, y = batch
        
        token_embeddings = token_embedding_layer(x)
        position_embeddings = position_embedding_layer(torch.arange(context_length))
        
        input_embeddings = token_embeddings + position_embeddings

    ##print(input_embeddings.shape)
    
    torch.manual_seed(123)
    segment_len = 2048
    
    infini_transformer = InfiniTransformer(embedding_dim, dim_hidden, key_value_dim, key_value_dim, num_heads, "relu", segment_len, "delta", False, None, True, 0.1)
    
    batch = input_embeddings
    print("Batch shape ->", batch.shape)
    context_vecs = infini_transformer(batch)
    print("Context vector shape ->", context_vecs.shape)
    print(context_vecs)
    
if __name__ == "__main__":
    main()