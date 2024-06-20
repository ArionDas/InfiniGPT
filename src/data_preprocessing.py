import os
import pandas as pd
import pathlib
import textwrap
import requests

import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from IPython.display import display
from IPython.display import Markdown

def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

### InfiniGPT Dataset Class
class InfiniGPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        ## Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        
        ## Sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


### DataLoader function
def InfiniGPTDataLoader(txt, batch_size=4, max_length=256, stride=128,
                        shuffle=False, drop_last=True, num_workers=0):
    ## initializing the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    ## creating the dataset
    dataset = InfiniGPTDataset(txt, tokenizer, max_length, stride)
    
    ## dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    
    return dataloader
"""
batch_size -> The number of samples (sequences) that will be grouped together and processed by the model at each training step. 
              In this case, a batch_size of 4 means 4 sequences will be processed together.
max_length -> The maximum length allowed for each sequence. Sequences exceeding this length will be truncated. 
              Here, max_length is set to 256, so sequences will be at most 256 tokens long.
stride -> This parameter controls how much the sequence window is shifted when creating new sequences from the text for training. 
          A stride of 128 means the window will be moved 128 positions between consecutive sequences. 
          This can introduce overlap between sequences, which can be helpful for the model to capture longer-range dependencies in the text.
shuffle -> This boolean value determines whether the data is shuffled before creating batches. 
           Shuffling helps to reduce the influence of the order in which the data is presented to the model and can improve generalization. 
           Here, shuffle is set to True, which means the data will be shuffled before creating batches.
drop_last -> This parameter specifies whether to drop the last incomplete batch if the total number of samples is not perfectly divisible by the batch_size. 
             Dropping the last incomplete batch can simplify training and memory management. 
             Here, drop_last is set to True, so any leftover incomplete batch will be discarded.
num_workers -> This parameter defines the number of worker processes to use for data loading. 
               Using multiple workers can improve data loading performance, especially when dealing with large datasets. 
               Here, num_workers is set to 0, which means the data will be loaded using the main process only.
"""


def main():

    ### Hyperparameters
    vocab_size = 50257
    output_dim = 256
    context_length = 1024
    max_length = 4

    ### Embeddings
    token_embedding_layer = nn.Embedding(vocab_size, output_dim)
    position_embedding_layer = nn.Embedding(context_length, output_dim)

    url = "https://raw.githubusercontent.com/ArionDas/InfiniGPT/main/data/book_clean.txt"
    response = requests.get(url)
    txt = response.text

    dataloader = InfiniGPTDataLoader(txt, batch_size=8, max_length=max_length, stride=max_length)

    for batch in dataloader:
        X, y = batch
        
        token_embeddings = token_embedding_layer(X)
        position_embeddings = position_embedding_layer(torch.arange(max_length))
        
        input_embeddings = token_embeddings + position_embeddings
        
        break

    print(input_embeddings.shape)
    
if __name__=="__main__":
    main()