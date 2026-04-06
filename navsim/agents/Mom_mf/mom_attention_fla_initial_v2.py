from __future__ import annotations

import math
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from typing import Optional, Tuple, TYPE_CHECKING

from fla.modules import FusedRMSNormGated, RMSNorm, ShortConvolution
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
from fla.layers.utils import get_unpad_data, index_first_axis, pad_input, unpad_input

if TYPE_CHECKING:
    from fla.models.utils import Cache

# ============================================================
# Helper functions (simplified versions from MoMAttention)
# ============================================================

import torch

def _ensure_bf16(x: torch.Tensor) -> torch.Tensor:
    if x is None:
        return None
    # 只有 CUDA 上才强制；CPU bf16 可能也行但看你需求
    if x.is_cuda and x.dtype == torch.float32:
        return x.to(torch.bfloat16)
    return x


def _upad_input(q, k, v, g, beta, mask):
    """Unpad input based on mask."""
    batch_size, seq_len = mask.shape[:2]
    
    # Get indices where mask is True
    indices = mask.flatten().nonzero(as_tuple=False).flatten()
    
    # Flatten and index
    q_unpad = rearrange(q, 'b s ... -> (b s) ...')[indices]
    k_unpad = rearrange(k, 'b s ... -> (b s) ...')[indices]
    v_unpad = rearrange(v, 'b s ... -> (b s) ...')[indices]
    g_unpad = rearrange(g, 'b s ... -> (b s) ...')[indices]
    beta_unpad = rearrange(beta, 'b s ... -> (b s) ...')[indices]
    
    # Compute cumulative sequence lengths
    seq_lens = mask.sum(dim=1)
    cu_seqlens = torch.cat([
        torch.tensor([0], device=mask.device, dtype=torch.int32),
        seq_lens.cumsum(dim=0).to(torch.int32)
    ])
    
    max_seq_len = seq_lens.max().item()
    
    return q_unpad, k_unpad, v_unpad, g_unpad, beta_unpad, indices, cu_seqlens, max_seq_len


def transform(hidden_states, routing_mask, num_memories, selected_memories, attention_mask):
    """Transform hidden states based on routing mask."""
    batch_size, seq_len, hidden_dim = hidden_states.shape
    
    # Create memory-specific hidden states
    hidden_list = []
    indices_list = []
    sorted_indices_list = []
    max_lens = []
    
    for mem_idx in range(num_memories):
        # Get mask for this memory
        mem_mask = routing_mask[:, :, mem_idx].bool()  # [B, L]
        
        if attention_mask is not None:
            mem_mask = mem_mask & attention_mask
        
        # Collect hidden states for this memory
        mem_hidden = []
        mem_indices = []
        mem_sorted_indices = []
        mem_max_len = 0
        
        for b in range(batch_size):
            mask_b = mem_mask[b]
            indices_b = mask_b.nonzero(as_tuple=False).flatten()
            
            if len(indices_b) > 0:
                mem_hidden.append(hidden_states[b, indices_b])
                mem_max_len = max(mem_max_len, len(indices_b))
            else:
                mem_hidden.append(torch.empty(0, hidden_dim, device=hidden_states.device, dtype=hidden_states.dtype))
            
            mem_indices.append(indices_b)
            mem_sorted_indices.append(torch.arange(len(indices_b), device=hidden_states.device))
        
        hidden_list.append(mem_hidden)
        indices_list.append(mem_indices)
        sorted_indices_list.append(mem_sorted_indices)
        max_lens.append(mem_max_len)
    
    # Find global max length across all memories
    global_max_len = max(max_lens) if max_lens else 1
    
    # Pad and stack with global max length
    transformed_hidden = []
    for mem_idx in range(num_memories):
        mem_hidden_padded = []
        for b in range(batch_size):
            h = hidden_list[mem_idx][b]
            current_len = len(h)
            if current_len < global_max_len:
                padding = torch.zeros(global_max_len - current_len, hidden_dim, 
                                    device=h.device, dtype=h.dtype)
                h = torch.cat([h, padding], dim=0) if current_len > 0 else padding
            mem_hidden_padded.append(h)
        transformed_hidden.append(torch.stack(mem_hidden_padded, dim=0))
    
    transformed_hidden = torch.stack(transformed_hidden, dim=0)  # [num_memories, B, global_max_len, D]
    
    # Create mask using global max length
    mask = torch.zeros(num_memories, batch_size, global_max_len, device=hidden_states.device, dtype=torch.bool)
    for mem_idx in range(num_memories):
        for b in range(batch_size):
            actual_len = len(indices_list[mem_idx][b])
            if actual_len > 0:
                mask[mem_idx, b, :actual_len] = True
    
    return transformed_hidden, indices_list, sorted_indices_list, global_max_len, mask, mask


def reconstruct(o, indices, sorted_indices, batch_size, seq_len, topk, routing_weights, mask):
    """Reconstruct output from memory-specific outputs."""
    num_memories = o.shape[0]
    hidden_dim = o.shape[-1]
    
    # Initialize output
    output = torch.zeros(batch_size, seq_len, hidden_dim, device=o.device, dtype=o.dtype)
    
    # Reconstruct from each memory
    for mem_idx in range(num_memories):
        for b in range(batch_size):
            idx = indices[mem_idx][b]
            if len(idx) > 0:
                # Get the valid outputs for this memory and batch
                valid_outputs = o[mem_idx, b, :len(idx)]
                
                # Get routing weights for this memory
                mem_weight = routing_weights[b, idx, :]  # [len(idx), topk]
                
                # Find which positions selected this memory
                mem_mask = (mem_weight > 0).any(dim=-1)  # [len(idx)]
                
                # Get the weight for this memory
                weights = mem_weight[mem_mask].sum(dim=-1, keepdim=True)  # [sum(mem_mask), 1]
                
                if weights.numel() > 0:
                    weighted_output = valid_outputs[mem_mask] * weights
                    output[b, idx[mem_mask]] += weighted_output
    
    return output


# ============================================================
# Self-Attention using MoM mechanism
# ============================================================

class MoMAttentionSelf(nn.Module):
    """
    Self-attention using MoM mechanism (based on flash-linear-attention implementation).
    """

    def __init__(
        self,
        mode: str = "chunk",
        hidden_size: int = 256,
        num_heads: int = 4,
        head_dim: int = 64,
        expand_v: float = 2.0,
        layer_idx: int = 0,
        use_output_gate: bool = True,
        use_short_conv: bool = False,  # Simplified: disable for now
        conv_size: int = 4,
        norm_eps: float = 1e-5,
        num_memories: int = 8,
        topk: int = 2,
        shared_mem: bool = False,
        single_kv_proj: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.expand_v = expand_v
        self.layer_idx = layer_idx
        self.use_output_gate = use_output_gate
        self.use_short_conv = use_short_conv
        self.num_memories = num_memories
        self.topk = topk
        self.shared_mem = shared_mem
        self.single_kv_proj = single_kv_proj
        
        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = int(self.key_dim * self.expand_v)
        self.head_v_dim = int(self.head_dim * self.expand_v)
        
        assert mode in ['chunk', 'fused_recurrent'], f"Not supported mode `{mode}`."
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
        self.gate = nn.Linear(hidden_size, self.num_memories, bias=False)

        if self.single_kv_proj:
            self.k_proj = nn.Linear(hidden_size, self.key_dim, bias=False)
            self.v_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.b_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
            self.a_proj = nn.Linear(hidden_size, self.num_heads, bias=False)
        else:
            self.k_proj = nn.ModuleList([
                nn.Linear(hidden_size, self.key_dim, bias=False)
                for _ in range(self.num_memories)
            ])
            self.v_proj = nn.ModuleList([
                nn.Linear(hidden_size, self.value_dim, bias=False)
                for _ in range(self.num_memories)
            ])
            self.b_proj = nn.ModuleList([
                nn.Linear(hidden_size, self.num_heads, bias=False)
                for _ in range(self.num_memories)
            ])
            self.a_proj = nn.ModuleList([
                nn.Linear(hidden_size, self.num_heads, bias=False)
                for _ in range(self.num_memories)
            ])
        
        # Initialize A_log as a parameter
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        # Delta rule parameters
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True
        
        # Output processing
        if use_output_gate:
            self.g_proj = nn.Linear(hidden_size, self.value_dim, bias=False)
            self.o_norm = FusedRMSNormGated(self.head_v_dim, eps=norm_eps)
        else:
            self.o_norm = RMSNorm(self.head_v_dim, eps=norm_eps)
        
        self.o_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.silu = nn.SiLU()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, None, Optional[Cache], torch.Tensor]:
        """Forward pass using MoM attention mechanism."""
        
        hidden_states = _ensure_bf16(hidden_states)

        # Process attention mask
        if attention_mask is not None:
            attention_mask = (attention_mask == 1)
            assert len(attention_mask.shape) == 2
        
        # Determine mode
        mode = 'fused_recurrent' if (hidden_states.shape[1] <= 64 and not self.training) else self.mode
        if self.training:
            assert mode == 'chunk', "Only chunk mode is supported in training."
        
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        
        # Router: compute memory selection
        router_logits = self.gate(hidden_states)  # [B, L, num_memories]
        scores = F.softmax(router_logits, dim=2, dtype=torch.float)
        routing_weights, selected_memories = torch.topk(scores, self.topk, dim=-1)  # [B, L, topk]
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)
        
        # Create full routing weight matrix
        routing_weights_full = torch.zeros(
            batch_size, seq_len, self.num_memories,
            dtype=routing_weights.dtype, device=routing_weights.device
        ).scatter(-1, selected_memories, routing_weights)
        routing_mask = routing_weights_full.bool().int()
        
        # Transform hidden states by memory
        hidden_transformed, indices, sorted_indices, max_len, mask, mask_2 = transform(
            hidden_states, routing_mask, self.num_memories, selected_memories, attention_mask
        )
        
        # Project Q, K, V for each memory
        q = self.q_proj(hidden_transformed)
        
        if self.single_kv_proj:
            k = self.k_proj(hidden_transformed)
            v = self.v_proj(hidden_transformed)
            beta = self.b_proj(hidden_transformed).sigmoid()
            g = -self.A_log.float().exp() * F.softplus(
                self.a_proj(hidden_transformed).float() + self.dt_bias
            )
        else:
            k = torch.stack([
                k_expert(hidden_transformed[i]) 
                for i, k_expert in enumerate(self.k_proj)
            ], dim=0)
            v = torch.stack([
                v_expert(hidden_transformed[i]) 
                for i, v_expert in enumerate(self.v_proj)
            ], dim=0)
            beta = torch.stack([
                b_expert(hidden_transformed[i]).sigmoid() 
                for i, b_expert in enumerate(self.b_proj)
            ], dim=0)
            g = torch.stack([
                -self.A_log.float().exp() * F.softplus(
                    a_expert(hidden_transformed[i]).float() + self.dt_bias
                )
                for i, a_expert in enumerate(self.a_proj)
            ], dim=0)
        
        # Reshape for processing: [num_memories, B, L, D] -> [(num_memories*B), L, D]
        q = rearrange(q, 'e b l d -> (e b) l d')
        k = rearrange(k, 'e b l d -> (e b) l d')
        v = rearrange(v, 'e b l d -> (e b) l d')
        g = rearrange(g, 'e b l h -> (e b) l h')
        beta = rearrange(beta, 'e b l h -> (e b) l h')
        mask_flat = rearrange(mask_2, 'e b l -> (e b) l')
        
        # Apply SiLU activation
        q = self.silu(q)
        k = self.silu(k)
        v = self.silu(v)
        
        # Reshape to multi-head format
        q = rearrange(q, 'b l (h d) -> b l h d', h=self.num_heads)
        k = rearrange(k, 'b l (h d) -> b l h d', h=self.num_heads)
        v = rearrange(v, 'b l (h d) -> b l h d', h=self.num_heads)
        
        # Compute cumulative sequence lengths for each memory-batch combination
        seq_lens = mask_flat.sum(dim=1)
        cu_seqlens = torch.cat([
            torch.tensor([0], device=hidden_states.device, dtype=torch.int32),
            seq_lens.cumsum(dim=0).to(torch.int32)
        ])
        
        # Flatten inputs for cu_seqlens usage (batch must be 1)
        # Extract only valid tokens based on mask
        total_tokens = seq_lens.sum().item()
        q_flat = torch.zeros(1, total_tokens, self.num_heads, self.head_dim, 
                           device=q.device, dtype=q.dtype)
        k_flat = torch.zeros(1, total_tokens, self.num_heads, self.head_dim, 
                           device=k.device, dtype=k.dtype)
        v_flat = torch.zeros(1, total_tokens, self.num_heads, int(self.head_dim * self.expand_v), 
                           device=v.device, dtype=v.dtype)
        g_flat = torch.zeros(1, total_tokens, self.num_heads, device=g.device, dtype=g.dtype)
        beta_flat = torch.zeros(1, total_tokens, self.num_heads, device=beta.device, dtype=beta.dtype)
        
        # Fill flattened tensors
        start_idx = 0
        for i in range(q.shape[0]):
            length = seq_lens[i].item()
            if length > 0:
                q_flat[0, start_idx:start_idx+length] = q[i, :length]
                k_flat[0, start_idx:start_idx+length] = k[i, :length]
                v_flat[0, start_idx:start_idx+length] = v[i, :length]
                g_flat[0, start_idx:start_idx+length] = g[i, :length]
                beta_flat[0, start_idx:start_idx+length] = beta[i, :length]
                start_idx += length
        
        # Apply gated delta rule
        if mode == 'chunk':
            o_flat, _ = chunk_gated_delta_rule(
                q=q_flat, k=k_flat, v=v_flat, g=g_flat, beta=beta_flat,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:  # fused_recurrent
            o_flat, _ = fused_recurrent_gated_delta_rule(
                q=q_flat, k=k_flat, v=v_flat, g=g_flat, beta=beta_flat,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        
        # Unflatten output back to [num_memories*B, L, H, D] format
        num_mem_batch = self.num_memories * batch_size
        o = torch.zeros(num_mem_batch, max_len, self.num_heads, int(self.head_dim * self.expand_v),
                       device=o_flat.device, dtype=o_flat.dtype)
        start_idx = 0
        for i in range(num_mem_batch):
            length = seq_lens[i].item()
            if length > 0:
                o[i, :length] = o_flat[0, start_idx:start_idx+length]
                start_idx += length
        
        # Reshape back: [B*num_memories, L, H, D] -> [num_memories, B, L, H*D]
        o = rearrange(o, '(e b) l h d -> e b l (h d)', b=batch_size)
        
        # Reconstruct output from memories
        o = reconstruct(
            o, indices=indices, sorted_indices=sorted_indices,
            batch_size=batch_size, seq_len=seq_len, topk=self.topk,
            routing_weights=routing_weights, mask=mask
        )
        
        # Apply output normalization and projection
        o = rearrange(o, 'b l (h d) -> b l h d', h=self.num_heads)
        
        if self.use_output_gate:
            g_out = rearrange(
                self.g_proj(hidden_states), 
                'b l (h d) -> b l h d', 
                h=self.num_heads, 
                d=self.head_v_dim
            )
            o = self.o_norm(o, g_out)
        else:
            o = self.o_norm(o)
        
        o = rearrange(o, 'b l h d -> b l (h d)')
        o = self.o_proj(o)
        
        return o, None, past_key_values, router_logits.view(-1, self.num_memories)


# ============================================================
# Cross-Attention (Simplified Implementation)
# ============================================================

class MoMAttentionCross(nn.Module):
    """
    Simplified Cross-attention using standard attention mechanism.
    Query comes from one source, Key/Value from another.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_heads: int = 4,
        head_dim: int = 64,
        **kwargs,
    ) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        query: torch.Tensor,  # [B, Q, D]
        keyval: torch.Tensor,  # [B, K, D]
        attention_mask_kv: Optional[torch.Tensor] = None,  # [B, K]
        **kwargs,
    ) -> Tuple[torch.Tensor, None, None, None]:
        
        B, Q, D = query.shape
        _, K, _ = keyval.shape
        
        # Project
        q = self.q_proj(query)  # [B, Q, num_heads*head_dim]
        k = self.k_proj(keyval)  # [B, K, num_heads*head_dim]
        v = self.v_proj(keyval)  # [B, K, num_heads*head_dim]
        
        # Reshape to multi-head
        q = rearrange(q, 'b q (h d) -> b h q d', h=self.num_heads)
        k = rearrange(k, 'b k (h d) -> b h k d', h=self.num_heads)
        v = rearrange(v, 'b k (h d) -> b h k d', h=self.num_heads)
        
        # Compute attention scores
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
        
        # Apply mask if provided
        if attention_mask_kv is not None:
            mask = attention_mask_kv.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, K]
            scores = scores.masked_fill(~mask.bool(), float('-inf'))
        
        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v)
        
        # Reshape and project output
        output = rearrange(output, 'b h q d -> b q (h d)')
        output = self.o_proj(output)
        
        return output, None, None, None


# ============================================================
# Quick tests
# ============================================================

def mom_self_attn_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    attn = MoMAttentionSelf(
        hidden_size=256,
        num_heads=4,
        head_dim=64,
        layer_idx=0,
        mode="chunk",
        num_memories=8,
        topk=2,
    ).to(device=device, dtype=dtype)
    
    attn.eval()

    B, T, D = 4, 31, 256
    x = torch.randn(B, T, D, device=device, dtype=dtype)

    with torch.no_grad():
        y, _, _, router_logits = attn(x, attention_mask=None, past_key_values=None, use_cache=False)
    
    print("[MoM Self-Attn] Success!")
    print("Output Shape:", y.shape)
    print("Router logits Shape:", router_logits.shape)


def mom_cross_attn_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    cross = MoMAttentionCross(
        hidden_size=256,
        num_heads=4,
        head_dim=64,
    ).to(device=device, dtype=dtype)
    
    cross.eval()

    B, Q, K, D = 4, 31, 641, 256
    query = torch.randn(B, Q, D, device=device, dtype=dtype)
    keyval = torch.randn(B, K, D, device=device, dtype=dtype)

    kv_mask = torch.ones(B, K, device=device, dtype=torch.bool)

    with torch.no_grad():
        y, _, _, _ = cross(query, keyval, attention_mask_kv=kv_mask)
    
    print("[MoM Cross-Attn] Success!")
    print("Output Shape:", y.shape)


if __name__ == "__main__":
    print("Testing MoM Attention modules...")
    print("=" * 50)
    mom_self_attn_test()
    print("=" * 50)
    mom_cross_attn_test()
    print("=" * 50)
    print("All tests completed!")
