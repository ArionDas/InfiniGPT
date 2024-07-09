import math
from typing import Optional, Tuple

import torch
from torch import nn

from activation import ACTIVATIONS
from compressive_memory import CompressiveMemory
from positional_embeddings import PositionEmbeddings
from infini_gpt_config import INFINIGPT_CONFIG

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