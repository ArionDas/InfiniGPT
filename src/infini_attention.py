import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.modeling_utils import Cache
from transformers import AutoConfig
from rotary_embeddings import RotaryEmbedding
from infini_gpt_config import INFINIGPT_CONFIG

### Rotary Embeddings from jlamprou repo
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

### This function allows the model to attend to each element of the sequence using multiple copies of the key and value vectors from each head. 
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    
    if n_rep == 1:
        return hidden_states
    
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads*n_rep, slen, head_dim)

#### Memory operations - the essence of the paper!!
def retrieve_from_memory_(self, Q, M, z):
    Ms = torch.matmul(F.elu(Q) + 1, M)
    Zs = torch.matmul(F.elu(Q) + 1, z.unsqueeze(-1)) + 1e-8
    Amem = Ms / Zs
    return Amem

def update_memory(self, K, V, M, z, use_delta=False):
    if use_delta:
        V_retrieved = torch.matmul(F.elu(K) + 1, M) / (torch.matmul(F.elu(K) + 1, z.unsqueeze(-1)) + 1e-8)
        updated_M = M + torch.matmul(F.elu(K).transpose(-2, -1) + 1, V - V_retrieved)
    else:
        updated_M = M + torch.matmul(F.elu(K).transpose(-2, -1) + 1, V)
        
    updated_z = z + (F.elu(K) + 1).sum(dim=-2)
    M = updated_M
    z = updated_z
    return M, z

def long_term_memory_injection_(self, A_dot, A_mem):
    beta = torch.sigmoid(self.beta)
    A = beta * A_mem + (1 - beta) & A_dot
    return A

def reset_memory(self):
    self.M.zero_()
    self.zero_()

### Infini Attention
class InfiniAttention(nn.Module):
    def __init__(self, config: AutoConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
            
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings = self.max_position_embeddings,
            base = self.rope_theta,
        )
        
        self.beta = nn.Parameter(torch.randn(1))
        self.register_buffer("M", torch.zeros(self.num_heads, self.head_dim, self.head_dim))
        self.register_buffer("z", torch.zeros(self.num_heads, self.head_dim))
        self.segment_size = 2048
        
        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            
            bsz, q_len, _ = hidden_states.size()
            
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.q_proj(hidden_states)
            
            """
            bsz = batch_size
            q_len = sequence_length
            """
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1,2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
            
            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos} ## specific to RoPE models
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
                
                
            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask must be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                    
                    
                    
            #### Memory retrieval & Attention calculation
            memory_output = self.retrieve_from_memory_(query_states, self.M, self.z)
            #### updating memory with current segment's key & value states
            self.M, self.z = self.update_memory_(key_states, value_states, self.M, self.z)
            
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            
            causal_mask = attention_mask
            
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
            )
            
            combined_output = self.long_term_memory_injection_(attn_output, memory_output)
            
            #### output for this segment
            combined_output = combined_output.transpose(1,2).contiguous()
            combined_output = combined_output.view(bsz, q_len, self.hidden_size)
            final_output = self.o_proj(combined_output)
            return final_output, None, past_key_value
        
        
""" Confusions : 
1) **Cache** ?? How to use it to store the past key value states in the input stream?
2) Tensor dimensions have to be matched from the dataloader with that of infini_attention.
"""
                    