from typing import Literal, Optional, Union

import torch
from torch import nn

from positional_embeddings import PositionEmbeddings

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
        print("Number of segments ->", num_segments)
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