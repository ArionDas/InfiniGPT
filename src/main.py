import os
import requests

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_preprocessing import InfiniGPTDataset, InfiniGPTDataLoader
from attention import CausalSelfAttention, MultiHeadAttention


def main():
    
    ### Hyperparameters
    vocab_size = 50257
    output_dim = 256
    max_length = 4
    context_length = 1024
    
    ### Embeddings
    token_embedding_layer = nn.Embedding(vocab_size, output_dim)
    position_embedding_layer = nn.Embedding(context_length, output_dim)

    url = "https://github.com/ArionDas/InfiniGPT/blob/eb3abdc6eaf1d8f17be7e92e81fd641a710aae26/data/book_clean.txt"
    response = requests.get(url)
    txt = response.text
    
    dataloader = InfiniGPTDataLoader(txt, batch_size=8, max_length=max_length, stride=max_length)
    
    for batch in dataloader:
        x, y = batch
        
        token_embeddings = token_embedding_layer(x)
        position_embeddings = position_embedding_layer(torch.arange(max_length))
        
        input_embeddings = token_embeddings + position_embeddings

    ##print(input_embeddings.shape)
    
    torch.manual_seed(123)
    
    context_length = max_length
    d_in = output_dim
    num_heads = 2
    d_out = d_in // num_heads
    
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads)
    
    batch = input_embeddings
    context_vecs = mha(batch)
    
    print(context_vecs.shape)
    
if __name__ == "__main__":
    main()