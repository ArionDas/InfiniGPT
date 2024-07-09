import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class InfiniGPTDataset(Dataset):
    def __init__(self, text_data, tokenizer, max_length, stride):
        
        self.input_text = []
        self.target_text = []
        
        tokens = tokenizer.encode(text_data, allowed_special={"<|endoftext|>"})
        
        ## chunking the book into overlapping sequences of max_length
        ## 1) this helps by allowing the model to train on smaller segments at a time, reducing the memory footprint during training
        ## 2) ensures better contextual learning, as the model is trained on both the beginning and ending of sentences
        
        for i in range(0, len(tokens) - max_length, stride):
            input_chunk = tokens[i: i + max_length]
            target_chunk = tokens[i + 1: i + max_length + 1]
            self.input_text.append(torch.tensor(input_chunk))
            self.target_text.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_text)
    
    def __getitem__(self, index):
        return self.input_text[index], self.target_text[index]
    
def InfiniGPT_dataloader(text_data, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2")
    
    dataset = InfiniGPTDataset(text_data, tokenizer, max_length, stride)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    
    return dataloader
        