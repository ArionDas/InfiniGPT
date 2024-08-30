import os
import matplotlib.pyplot as plt
import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import urllib.request
import tiktoken

# from infini_gpt_config import INFINIGPT_CONFIG
from activation import ACTIVATIONS
from positional_embeddings import PositionEmbeddings

## InfiniGPT Config ##
INFINIGPT_CONFIG = {
    "vocab_size": 50257,                # Vocabulary size
    "input_dim": 1024,                   # Input dimension 
    "output_dim": 1024,                  # Output dimension               
    "context_length": 1024,             # Context length
    "stride": 1024,                     # Stride length
    "emb_dim": 1024,                     # Embedding dimension
    "num_attention_heads": 16,           # Number of attention heads
    "n_layers": 12,                      # Number of layers
    "drop_rate": 0.1,                   # Dropout rate
    "qkv_bias": False,                  # Query-Key-Value bias
    "hidden_dim": 4096,                 # Hidden dimension
    "segment_len": 512,                # Segment length
    "batch_size": 16,                   # Batch size
    "key_value_dim": 64,                # Key & Value dimensions
    "update": "delta",                  # Update rule
    "num_epochs": 10,                   # Epochs
    "drop_rate" : 0.1,                  # Dropout rate
    "learning_rate": 1e-4,              # Learning rate
    "weight_decay": 0.01,                # Weight decay
}

## Parameters for testing model
"""INFINIGPT_CONFIG = {
    "vocab_size": 50257,                # Vocabulary size
    "input_dim": 256,                   # Input dimension 
    "output_dim": 256,                  # Output dimension               
    "context_length": 512,             # Context length
    "stride": 256,                     # Stride length
    "emb_dim": 256,                     # Embedding dimension
    "num_attention_heads": 2,           # Number of attention heads
    "n_layers": 2,                      # Number of layers
    "drop_rate": 0.1,                   # Dropout rate
    "qkv_bias": False,                  # Query-Key-Value bias
    "hidden_dim": 2048,                 # Hidden dimension
    "segment_len": 512,                # Segment length
    "batch_size": 4,                   # Batch size
    "key_value_dim": 64,                # Key & Value dimensions
    "update": "delta",                  # Update rule
    "num_epochs": 1,                   # Epochs
    "drop_rate" : 0.1,                  # Dropout rate
    "learning_rate": 1e-4,              # Learning rate
    "weight_decay": 0.01,                # Weight decay
}"""

## DataLoader ##
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

## Compressive Memory ##
class CompressiveMemory(nn.Module):

    def __init__(
        self,
        dim_input: int,
        dim_key: int,
        dim_value: int,
        num_attention_heads: int,
        segment_length: int,
        sampling_factor: Optional[int] = None,
        update: str = "linear",
        causal: bool = False,
        position_embedder: Optional[PositionEmbeddings] = None,
        init_state_learnable: bool = False
    ):
        """Compressive Memory Module.

        Args:
            dim_input (int): Input dimension.
            dim_key (int): Key dimension.
            dim_value (int): Value dimension.
            num_attention_heads (int): Number of attention heads.
            segment_length (int): Segment length (note that it should be a factor of the input sequence length).
            sampling_factor (Optional[int], optional): Default: None.
            update (str, optional): Type of memory update rule to use ("linear" or "delta"). Default: "linear".
            causal (bool, optional): Whether to use causal attention masking. Defaults to False.
            position_embedder (Optional[PositionEmbeddings], optional): Position embedding module. Default: None.
            init_state_learnable (bool, optional): Whether the initial memory and normalization are learnable. Default: False.
        """
        super(CompressiveMemory, self).__init__()

        # Input parameters
        self.num_attention_heads = num_attention_heads
        self.segment_length = segment_length
        self.sampling_factor = sampling_factor

        self.dim_input = dim_input
        self.dim_key = dim_key
        self.dim_value = dim_value

        self.update = update
        self.causal = causal

        self.position_embedder = position_embedder

        # Projections for stacked SDP attention
        self.proj_keys = nn.Linear(dim_input, num_attention_heads * dim_key, bias=False)
        self.proj_values = nn.Linear(dim_input, num_attention_heads * dim_value, bias=False)
        self.proj_query = nn.Linear(dim_input, num_attention_heads * dim_key, bias=False)

        # For weighted average of dot-product and memory-based attention
        self.betas = nn.Parameter(torch.randn(1, num_attention_heads, 1, dim_value))

        # Output projections
        self.proj_out = nn.Linear(num_attention_heads * dim_value, dim_input, bias=False)

        # If init_state_learnable is set, create parameters for the initial memory matrix and normalization vector
        # otherwise, set them to None
        if init_state_learnable:
            self.memory = nn.Parameter(torch.randn(1, self.num_attention_heads, self.dim_key, self.dim_value))
            self.norm = nn.Parameter(torch.ones(1, self.num_attention_heads, self.dim_key, 1))
        else:
            self.memory = None
            self.norm = None

    def forward(self, x: torch.Tensor, sample_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Applying Compressive Memory Attention to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_input).
            sample_mask (Optional[torch.Tensor], optional): Mask tensor of shape (batch_size, seq_len) used to sample the input sequence. Defaults to None.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_input).
        """
        batch_size, seq_len, _ = x.shape

        num_segments = (seq_len // self.segment_length) + (seq_len % self.segment_length > 0) ##effectively calculating number of segments
        #print("Number of segments ->", num_segments)
        out = [] ## output buffer

        # Memory intialization and normalization
        if self.memory is not None and self.norm is not None:
            mem = self.memory
            z = self.norm
        else:
            mem = torch.zeros(1, self.num_attention_heads, self.dim_key, self.dim_value)
            z = torch.ones(batch_size, self.num_attention_heads, self.dim_key, 1) / self.dim_key

        ## Projections to get the key, value, and query tensors
        """
        We introcude a new dimension to use it to divide the input sequence into segments as mentioned in the paper.
        """
        key_proj = self.proj_keys(x).unsqueeze(1).view((batch_size, self.num_attention_heads, x.size(1), self.dim_key))
        value_proj = self.proj_values(x).unsqueeze(1).view((batch_size, self.num_attention_heads, x.size(1), self.dim_value))
        query_proj = self.proj_query(x).unsqueeze(1).view((batch_size, self.num_attention_heads, x.size(1), self.dim_key))

        ## Iterating over segments
        for segment in range(num_segments):
            segment_starting_index = segment * self.segment_length
            segment_ending_index = min(segment_starting_index + self.segment_length, x.size(1))
            segment_length = segment_ending_index - segment_starting_index

            # Extracting a segment from key, value and query tensors
            k = key_proj[:, :, segment_starting_index:segment_ending_index, :]
            v = value_proj[:, :, segment_starting_index:segment_ending_index, :]
            q = query_proj[:, :, segment_starting_index:segment_ending_index, :]

            # Applying a sample mask
            if sample_mask is not None:
                if self.sampling_factor is None:
                    raise ValueError("Sampling Mask provided but sampling factor not specified.")
                segment_starting_index_seg = segment * self.segment_length * self.sampling_factor
                segment_ending_index_seg = min(segment_starting_index_seg + self.segment_length * self.sampling_factor, sample_mask.size(1))
                sample_mask_seg = sample_mask[:, segment_starting_index_seg:segment_ending_index_seg]
            else:
                sample_mask_seg = None

            # If position embedder is specified, add positional embeddings to q and k
            ## Note that I didn't use it, as its not the essence of the paper. Position Embeddings have been calculated during data preprocessing.
            ## Nonetheless, I have included dingo-actual's implementation of RoPE & YaRN embeddings.
            if self.position_embedder is not None:
                if sample_mask is None:
                    k_pos = self.position_embedder(k, total_seq_len=seq_len, offset=segment_starting_index)
                    q_pos = self.position_embedder(q, total_seq_len=seq_len, offset=segment_starting_index)
                else:
                    k_pos = self.position_embedder(k, total_seq_len=seq_len, offset=segment_starting_index_seg, select_mask=sample_mask_seg)
                    q_pos = self.position_embedder(q, total_seq_len=seq_len, offset=segment_starting_index_seg, select_mask=sample_mask_seg)

            # sigma(q) is pre-calculated for updating memory and calculating attention
            # shape: (batch_size, num_attention_heads, segment_length, dim_key)
            sigma_query = (nn.functional.elu(q) + 1.0)

            # Applying Scaled Dot Product attention
            if self.position_embedder is not None:
                scores = q_pos @ k_pos.transpose(-2, -1) / self.dim_key ** 0.5
            else:
                scores = q @ k.transpose(-2, -1) / self.dim_key ** 0.5

            # If causal mask specified, calculate and apply it
            if self.causal:
                causal_mask = torch.tril(torch.ones((segment_length, segment_length), dtype=torch.bool), diagonal=0)
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).repeat((batch_size, self.num_attention_heads, 1, 1))
                scores.causal_masked_fill_(torch.logical_not(causal_mask), float('-inf'))

            # Calculate SDP attention, completing equation (2) of the paper
            dot_product_attention = nn.functional.softmax(scores, dim=-1) @ v

            # Normalized linear attention
            # shape: (batch_size, num_attention_heads, segment_lengthgth, dim_value)
            att_mem = (sigma_query @ mem) / (sigma_query @ z)

            # Applying memory update
            sigma_k = nn.functional.elu(k) + 1.0
            """Linear & Delta Update

                In the 'Linear' case, we don't remove already existing parts from the memory.
                But, in 'Delta' case, we remove the already existing parts to avoid repetition.

                Refer to the paper for more details
            """
            if self.update == "linear":
                mem = mem + sigma_k.transpose(-2, -1) @ v
            elif self.update == "delta":
                mem = mem + sigma_k.transpose(-2, -1) @ (v - (sigma_k @ mem) / (sigma_k @ z))

            # Normalization term update
            z = z + (nn.functional.elu(k) + 1.0).sum(dim=-2, keepdim=True).transpose(-2, -1)

            # Weighted average of dot-product and memory-based attention
            att = nn.functional.sigmoid(self.betas) * att_mem + (1 - nn.functional.sigmoid(self.betas)) * dot_product_attention
            att = att.view((batch_size, segment_length, self.num_attention_heads * self.dim_value))

            # Append to output buffer
            out.append(self.proj_out(att))

        out = torch.concat(out, dim=1)
        return out

    
## Infini Transformer ##
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["emb_dim"], 4 * config["emb_dim"]),
            GELU(),
            nn.Linear(4 * config["emb_dim"], config["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)

class InfiniTransformer(nn.Module):
    """Transformer layer with compressive memory."""

    def __init__(
        self,
        dim_input: int,
        emb_dim: int,
        dim_hidden: int,
        dim_key: int,
        dim_value: int,
        num_attention_heads: int,
        activation: str,
        segment_length: int,
        update: str = "linear",
        causal: bool = False,
        position_embedder: Optional[PositionEmbeddings] = None,
        init_state_learnable: bool = False,
        dropout: float = 0.0,
        **kwargs
    ):
        """Initializes the module.

        Args:
            dim_input (int): Input dimension.
            dim_hidden (int): Hidden dimension for the MLP.
            dim_key (int): Key dimension for the CompressiveMemory.
            dim_value (int): Value dimension for the CompressiveMemory.
            num_heads (int): Number of attention heads for the CompressiveMemory.
            activation (str): Activation function to use for the MLP. Must be a key in the ACTIVATIONS dictionary.
            segment_length (int): Segment length for the CompressiveMemory.
            update (str, optional): Type of memory update rule to use for the CompressiveMemory ("linear" or "delta"). Defaults to "linear".
            causal (bool, optional): Whether to use causal attention masking for the CompressiveMemory. Defaults to False.
            position_embedder (Optional[PositionEmbeddings], optional): Position embedding module for the CompressiveMemory. Defaults to None.
            init_state_learnable (bool, optional): Whether the initial state of the CompressiveMemory should be learnable. Defaults to False.
            dropout (float, optional): Dropout rate for the MLP. Defaults to 0.0.
        """
        super().__init__()

        # If sampling_factor passed to kwargs, use it, otherwise set to None
        sampling_factor = kwargs.get("sampling_factor", None)

        config = INFINIGPT_CONFIG
        # Multi-head attention
        self.infini_attention = CompressiveMemory(
            dim_input=dim_input,
            dim_key=dim_key,
            dim_value=dim_value,
            num_attention_heads=num_attention_heads,
            segment_length=segment_length,
            sampling_factor=sampling_factor,
            update=update,
            causal=causal,
            position_embedder=position_embedder,
            init_state_learnable=init_state_learnable
        )
        # MLP
        if activation not in ACTIVATIONS:
            raise ValueError(f"Invalid activation function: {activation}")
        if activation in ["relu"]:
            act = ACTIVATIONS[activation](dim_hidden)
        else:
            act = ACTIVATIONS[activation]()

        self.mlp = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Dropout(dropout),
            act,
            nn.Linear(dim_hidden, dim_input),
            nn.Dropout(dropout)
        )

        self.ff = FeedForward(config) ## Here, GELU() has been used instead of ReLU(), use self.mlp for using ReLU()
        self.drop_emb = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_input)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim_input).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, dim_input).
        """
        # Apply infini attention, followed by fully connected layer and layer normalization
        original = x
        x = self.layer_norm(x)
        x = self.infini_attention(x)
        x = self.drop_emb(x)
        x = x + original

        original = x
        x = self.layer_norm(x)
        x = self.ff(x)
        x = self.drop_emb(x)
        x = x + original

        return x
    
## Infini GPT ##
class InfiniGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embeddings = nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.position_embeddings = nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_emb = nn.Dropout(config["drop_rate"])

        self.infini_transformer_blocks = nn.Sequential(
            *[InfiniTransformer(config["emb_dim"], config["emb_dim"], config["hidden_dim"], config["key_value_dim"], config["key_value_dim"], config["num_attention_heads"], "relu", config["segment_len"], config["update"], False, None, True, config["drop_rate"]) for _ in range(config["n_layers"])]
        )

        self.final_norm = nn.LayerNorm(config["emb_dim"])
        self.ff = nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, context):
        batch_size, seq_len = context.shape
        token_embeddings = self.token_embeddings(context)
        position_embeddings = self.position_embeddings(torch.arange(seq_len, device=context.device))

        x = token_embeddings + position_embeddings
        x = self.drop_emb(x)
        x = self.infini_transformer_blocks(x)
        x = self.final_norm(x)
        logits = self.ff(x)
        return logits
    
    
## Model ##
def count_parameters(model):
    total_params = 0
    for param in model.parameters():
        if param.requires_grad:  # Check if parameter is trainable
            total_params += param.numel()  # Add the number of elements in the parameter
    
    ## Adding model information for reference
    with open("model_info.txt", "w") as f:
        total_params = total_params
        f.write(f"Number of Trainable Parameters: {total_params}\n")
        f.write(f"Model Architecture:\n\n")
        f.write(str(model))
    print(f"\nTotal number of trainable parameters = {total_params}\n")
    return total_params
  
  
## Training Script ##
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten()) ## suitable for multi-class & multi-label classifications
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    step = -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:

            optimizer.zero_grad()

            loss = calc_loss_batch(input_batch, target_batch, model, device)

            loss.backward() ## calculating loss gradients

            optimizer.step() ## updating model weights using loss gradients

            tokens_seen += input_batch.numel() ## .numel() gets the number of elements in a PyTorch tensor. Returns an integer regardless of dimension

            step += 1


            if step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                
                log_directory = "./logs"
                if not os.path.exists(log_directory):
                    os.makedirs(log_directory)
                with open(os.path.join(log_directory, "logs.txt"), "a") as f:
                    f.write(f"Ep {epoch+1} (Step {step:06d}): Train loss {train_loss:.3f}, Val loss {val_loss:.3f}\n")

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    fig.savefig("./training_info/plot.png")
    print("\n Saved loss over time plot. \n")
    
"""## Inference ##
def generate_text(model, prompt, max_tokens, context_size):
    model.eval()
    for _ in range(max_tokens):
        prompt = prompt[:, :context_size]
        logits = model(prompt)
        logits = logits[:, -1, :]
        logit_probs = nn.functional.softmax(logits, dim=-1)
        next_prompt = torch.multinomial(logit_probs, num_samples=1)
        prompt = torch.cat((prompt, next_prompt), dim=1)
    return prompt"""


def main(config):

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """ Text Data """
    text_data = ""
    file_path = "./data/book_clean.txt"

    file = open(file_path, errors="ignore")
    text_data = file.read()
    print(f"Total number of characters in the text dataset = {len(text_data)}")


    """ Model """

    model = InfiniGPT(INFINIGPT_CONFIG)
    model.to(device)
    print(f"\nConfigured InfiniGPT model...\n")
    count_parameters(model)
    optimizer = torch.optim.AdamW(
                model.parameters(), config["learning_rate"], weight_decay=config["weight_decay"]
                ) ## weight decay somewhat helps in preventing overfitting


    """ Dataloaders """
    train_ratio = 0.90
    split = int(train_ratio * len(text_data))

    train_loader = InfiniGPT_dataloader(
        text_data[:split],
        batch_size = config["batch_size"],
        max_length = config["context_length"],
        stride = config["stride"],
        drop_last = True,
        shuffle = True,
    )

    val_loader = InfiniGPT_dataloader(
        text_data[split:],
        batch_size = config["batch_size"],
        max_length = config["context_length"],
        stride = config["stride"],
        drop_last = False,
        shuffle = True,
    )


    """ Model Training """
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"\nTraining InfiniGPT model...\n")
    train_losses, val_losses, tokens_seen = train_model(model, train_loader, val_loader, optimizer, device,
                                                       num_epochs=config["num_epochs"], eval_freq=1, eval_iter=1,
                                                       tokenizer=tokenizer)

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    config = INFINIGPT_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_losses, val_losses, tokens_seen, model = main(config)

    epochs_tensor = torch.linspace(0, config["num_epochs"], len(train_losses))
    
    
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    directory = "./training_info"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, "plot.png"))
    
    model_directory = "./model"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    torch.save(model.state_dict(), os.path.join(model_directory, "text_infinigpt_model.pth"))
    
    print("InfiniGPT Model trained!!")
    torch.cuda.empty_cache()
