from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from fla.layers import MomAttention
from fla.layers.attn import Attention
# from fla.models.mom.configuration_mom import MomConfig
from fla.models.utils import Cache, FLAGenerationMixin
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from fla.modules import GatedMLP as MomMLP

from navsim.agents.Mom_mf.mom_config import MomConfig 

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack


try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    from fla.models.modeling_layers import GradientCheckpointingLayer

logger = logging.get_logger(__name__)

class MomBlock(GradientCheckpointingLayer):

    def __init__(self, config: MomConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps, dtype=torch.float32)
        if config.attn is not None and layer_idx in config.attn['layers']:
            self.attn = Attention(
                hidden_size=config.hidden_size,
                num_heads=config.attn['num_heads'],
                num_kv_heads=config.attn['num_kv_heads'],
                window_size=config.attn['window_size'],
                max_position_embeddings=config.max_position_embeddings,
                layer_idx=layer_idx,
            )
        else:
            if config.mom_backend == 'gated_deltanet':
                self.attn = MomAttention(
                    mode=config.attn_mode,
                    hidden_size=config.hidden_size,
                    expand_v=config.expand_v,
                    head_dim=config.head_dim,
                    num_heads=config.num_heads,
                    use_output_gate=config.use_output_gate,
                    use_short_conv=config.use_short_conv,
                    conv_size=config.conv_size,
                    norm_eps=config.norm_eps,
                    layer_idx=layer_idx,
                    num_memories=config.num_memories,
                    topk=config.topk,
                    capacity=config.capacity,
                    shared_mem=config.shared_mem,
                    single_kv_proj=config.single_kv_proj,
                )
            else:
                raise NotImplementedError(f"The MoM backend {config.mom_backend} is not currently supported.")
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps, dtype=torch.float32)
        self.mlp = MomMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs: Unpack[dict],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        # NOTE: no temporal embedding is applied in this block.
        # Auto-initialize Cache for recurrent single-frame inference
        if use_cache and past_key_values is None:
            past_key_values = Cache()
        residual = hidden_states
        if hasattr(self, 'attn_norm'):
            hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values, router_logits = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        if hasattr(self, 'mlp_norm'):
            hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
        hidden_states = self.mlp(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, attentions, past_key_values, router_logits)

        return outputs


class Mom_DoubleSelfAttnBlock(nn.Module):
    """
    Double Self-Attention Block for MoM architecture.
    This block processes query and key-value sequences with two attention mechanisms:
    1. Query self-attention
    2. Cross-attention between query and key-value sequences
    """
    def __init__(
        self, 
        config: MomConfig,
        layer_idx: int = None
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        
        # Pre-normalization layer (only for first layer)
        if hasattr(config, 'norm_first') and config.norm_first and layer_idx == 0:
            self.pre_norm = RMSNorm(
                hidden_size=config.hidden_size,
                eps=config.norm_eps,
                dtype=torch.float32
            )

        # Normalization layers
        self.qkv_norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.norm_eps,
            dtype=torch.float32
        )

        self.ffn_norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.norm_eps,
            dtype=torch.float32
        )

        # Attention block
        self.query_attn = MomBlock(
            config=config,
            layer_idx=layer_idx * 2 if layer_idx is not None else 0
        )

        # QKV cross-attention
        if config.mom_backend == 'gated_deltanet':
            self.qkv_self_attn = MomAttention(
                mode=config.attn_mode,
                hidden_size=config.hidden_size,
                expand_v=config.expand_v,
                head_dim=config.head_dim,
                num_heads=config.num_heads,
                use_output_gate=config.use_output_gate,
                use_short_conv=config.use_short_conv,
                conv_size=config.conv_size,
                norm_eps=config.norm_eps,
                layer_idx=layer_idx * 2 + 1 if layer_idx is not None else 1,
                num_memories=config.num_memories,
                topk=config.topk,
                capacity=config.capacity,
                shared_mem=config.shared_mem,
                single_kv_proj=config.single_kv_proj,
            )
        else:
            raise NotImplementedError(f"The MoM backend {config.mom_backend} is not currently supported.")

        # Feed-forward network
        self.ffn = MomMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self, 
        query: torch.Tensor,
        keyval: torch.Tensor,
        frames: Optional[torch.Tensor] = None,
        past_query: Optional[Cache] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for double self-attention block.
        
        Args:
            query: Query tensor [B, query_len, hidden_size]
            keyval: Key-value tensor [B, keyval_len, hidden_size]
            frames: Optional frame indices for video processing
            past_query: Cache for query attention
            past_key_values: Cache for cross-attention
            use_cache: Whether to use and return cache
            output_attentions: Whether to output attention weights
            
        Returns:
            Tuple containing:
                - hidden_states: Output tensor
                - cross_attentions: Cross-attention weights
                - past_query: Updated query cache
                - past_key_values: Updated cross-attention cache
                - query_router_logits: Router logits from query attention
                - qkv_router_logits: Router logits from cross-attention
        """
        # Auto-initialize Caches for recurrent single-frame inference
        if use_cache and past_query is None:
            past_query = Cache()
        if use_cache and past_key_values is None:
            past_key_values = Cache()

        # Query self-attention
        query, self_attentions, past_query, query_router_logits = self.query_attn(
            hidden_states=query,
            past_key_values=past_query,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        
        # Apply pre-normalization if exists
        residual = self.pre_norm(query) if hasattr(self, 'pre_norm') else query

        B, query_len, _ = query.size()
        
        # Handle frame-based processing (for video data)
        if frames is not None:
            keyval_padding = torch.zeros_like(query, device=query.device)
            qkv = torch.cat([keyval, keyval_padding], dim=1)

            # Place query at the right position based on frame indices
            frame_indices = frames * 64 + 1
            batch_indices = torch.arange(B, device=query.device).unsqueeze(1)
            query_indices = torch.arange(query_len, device=query.device).unsqueeze(0)
            position_indices = frame_indices.unsqueeze(1) + query_indices
            
            qkv[batch_indices, position_indices, :] = residual

            # Cross-attention
            qkv_attn, cross_attentions, past_key_values, qkv_router_logits = self.qkv_self_attn(
                hidden_states=self.qkv_norm(qkv),
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs
            )
            
            # Extract query attention results
            query_attn = qkv_attn[batch_indices, position_indices, :]
        else:
            # Standard concatenation for non-video data
            qkv = torch.cat([keyval, residual], dim=1)
            
            # Cross-attention
            qkv_attn, cross_attentions, past_key_values, qkv_router_logits = self.qkv_self_attn(
                hidden_states=self.qkv_norm(qkv),
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs
            )
            
            # Extract query part from output
            query_attn = qkv_attn[:, -query_len:, :]

        # Feed-forward network with residual connection
        hidden_states, residual = self.ffn_norm(query_attn, residual, True)
        hidden_states = self.ffn(hidden_states, **kwargs)
        hidden_states = residual + hidden_states

        outputs = (
            hidden_states,
            cross_attentions,
            past_query,
            past_key_values,
            query_router_logits,
            qkv_router_logits
        )

        return outputs

def mom_block_test():
    import torch
    import torch.nn as nn
    
    # -----------------------------------------------------------------------
    # 1. 准备配置 (Mock Config)
    #    为了让 MomBlock 正常初始化，我们需要提供它所依赖的所有参数。
    #    这里参考了 mom_backend='gated_deltanet' 的分支逻辑。
    # -----------------------------------------------------------------------
    from fla.models import MomConfig    

    config = MomConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------------
    # 2. 初始化模型
    # -----------------------------------------------------------------------
    print(f"Initializing MomBlock model on {device}...")
    model = nn.ModuleList(
        [
            MomBlock(
                config=config,
                layer_idx=i
            )
            for i in range(2)
        ]
    )
    model.to(device)
    model.eval()
    print("Model initialized.")
    # -----------------------------------------------------------------------
    # 3. 准备输入数据
    # -----------------------------------------------------------------------
    B = 4      # Batch size
    seq_len = 32   # Sequence length
    
    # input_states: [Batch, Seq, Hidden]
    hidden_states = torch.randn(B, seq_len, config.hidden_size, device=device, requires_grad=True)
    for layer in model:
        # ------------------- 修改了这一行 -------------------
        # 显式指定 v_first=v_first，避免被当成 attention_mask
        keyval, *_ = layer(
            hidden_states,
            attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
        )
    # -----------------------------------------------------------------------
    # 4. 前向传播 (Forward Pass)
    # -----------------------------------------------------------------------

    print("Output Shape: ", keyval.shape)
    print(f"Running Forward pass (Batch={B}, Seq={seq_len})...")

    # -----------------------------------------------------------------------
    # 对比测试：chunk 模式（全序列）vs recurrent 模式（逐帧），用相同输入
    # -----------------------------------------------------------------------
    print("\n--- Chunk vs Recurrent error comparison ---")
    T = 8
    x_seq = torch.randn(B, T, config.hidden_size, device=device)

    # --- chunk 模式：整段序列一次性推理，取最后一个 token 输出 ---
    with torch.no_grad():
        h_chunk = x_seq
        for layer in model:
            h_chunk, *_ = layer(h_chunk, attention_mask=None, past_key_values=None, use_cache=False)
    chunk_last = h_chunk[:, -1:, :]   # [B, 1, D]

    # --- recurrent 模式：逐帧输入，链式传递 past_key_values ---
    layer_past_key_values = [None] * len(model)
    with torch.no_grad():
        for t in range(T):
            x_t = x_seq[:, t:t+1, :]  # [B, 1, D]，与 chunk 输入完全一致
            for i, layer in enumerate(model):
                x_t, _, layer_past_key_values[i], _ = layer(
                    x_t,
                    attention_mask=None,
                    past_key_values=layer_past_key_values[i],
                    use_cache=True,
                )
    recurrent_last = x_t  # 最后一步输出

    max_err  = (chunk_last - recurrent_last).abs().max().item()
    mean_err = (chunk_last - recurrent_last).abs().mean().item()
    print(f"  chunk_last   shape : {chunk_last.shape}")
    print(f"  recurrent_last shape: {recurrent_last.shape}")
    print(f"  Max  abs error: {max_err:.6e}")
    print(f"  Mean abs error: {mean_err:.6e}")
    print("MomBlock single-frame test passed!")


def cross_attn_test():
    from fla.models import MomConfig    

    config = MomConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_layers = 2
    model = nn.ModuleList(
        [
            Mom_DoubleSelfAttnBlock(
                config=config,
                layer_idx=i
            )
            for i in range(num_layers)
        ]
    )
    model.to(device)
    model.eval()

    B = 4              # Batch size
    query_len = 16     # Query sequence length
    keyval_len = 32    # Key-value sequence length
    
    # Prepare inputs
    query = torch.randn(B, query_len, config.hidden_size, device=device, requires_grad=True)
    keyval = torch.randn(B, keyval_len, config.hidden_size, device=device, requires_grad=True)
    
    print(f"Input shapes:")
    print(f"  - Query:  {query.shape}")
    print(f"  - Keyval: {keyval.shape}")
    print()
    
    # Forward pass through all layers
    past_query = None
    past_key_values = None
    
    for i, layer in enumerate(model):
        print(f"Processing layer {i}...")
        output = layer(
            query=query,
            keyval=keyval,
            frames=None,
            past_query=past_query,
            past_key_values=past_key_values,
            use_cache=True,
                output_attentions=True,
        )
        
        # Unpack outputs
        hidden_states, cross_attentions, past_query, past_key_values, query_router_logits, qkv_router_logits = output
        
        # Update query for next layer
        query = hidden_states
        
        print(f"  ✓ Output shape: {hidden_states.shape}")
        print(f"  ✓ Cross-attention shape: {cross_attentions.shape if cross_attentions is not None else 'None'}")
        print(f"  ✓ Query router logits shape: {query_router_logits.shape if query_router_logits is not None else 'None'}")
        print(f"  ✓ QKV router logits shape: {qkv_router_logits.shape if qkv_router_logits is not None else 'None'}")
        print()
    
    print(f"Final output shape: {hidden_states.shape}")
    print(f"Expected shape: [{B}, {query_len}, {config.hidden_size}]")
    assert hidden_states.shape == (B, query_len, config.hidden_size), "Output shape mismatch!"
    print("✓ Standard mode test passed!")
    print()

    # -----------------------------------------------------------------------
    # 对比测试：chunk 模式（全序列 query）vs recurrent 模式（逐帧 query），用相同输入
    # keyval 固定不变（代表当前帧传感器特征），query 沿时间步展开
    # -----------------------------------------------------------------------
    print("--- Chunk vs Recurrent error comparison ---")
    T = 8
    query_seq = torch.randn(B, T, config.hidden_size, device=device)
    keyval_fixed = torch.randn(B, keyval_len, config.hidden_size, device=device)

    # --- chunk 模式：整段 query 序列一次性推理，取最后一个 token 输出 ---
    with torch.no_grad():
        q_chunk = query_seq
        for layer in model:
            q_chunk, *_ = layer(
                query=q_chunk,
                keyval=keyval_fixed,
                frames=None,
                past_query=None,
                past_key_values=None,
                use_cache=False,
            )
    chunk_last = q_chunk[:, -1:, :]  # [B, 1, D]

    # --- recurrent 模式：逐帧输入 query，链式传递两套 past，keyval 每步一致 ---
    layer_past_query = [None] * num_layers
    layer_past_key_values = [None] * num_layers
    with torch.no_grad():
        for t in range(T):
            q_t = query_seq[:, t:t+1, :]  # [B, 1, D]
            for i, layer in enumerate(model):
                q_t, _, layer_past_query[i], layer_past_key_values[i], _, _ = layer(
                    query=q_t,
                    keyval=keyval_fixed,
                    frames=None,
                    past_query=layer_past_query[i],
                    past_key_values=layer_past_key_values[i],
                    use_cache=True,
                )
    recurrent_last = q_t  # 最后一步输出

    max_err  = (chunk_last - recurrent_last).abs().max().item()
    mean_err = (chunk_last - recurrent_last).abs().mean().item()
    print(f"  chunk_last    shape: {chunk_last.shape}")
    print(f"  recurrent_last shape: {recurrent_last.shape}")
    print(f"  Max  abs error: {max_err:.6e}")
    print(f"  Mean abs error: {mean_err:.6e}")
    assert q_t.shape == (B, 1, config.hidden_size), "Single-frame output shape mismatch!"
    print("✓ Single-frame recurrent test passed!")
    print()


def recurrent_inference_test():
    """Test single-frame recurrent inference: feed one frame at a time and chain recurrent states."""
    from fla.models import MomConfig

    config = MomConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_layers = 2
    model = nn.ModuleList(
        [Mom_DoubleSelfAttnBlock(config=config, layer_idx=i) for i in range(num_layers)]
    )
    model.to(device)
    model.eval()

    B = 4
    query_len = 1    # 单帧：每次只输入1个 query token
    keyval_len = 32  # 每帧的 keyval 特征长度
    T = 5            # 模拟5个时间步

    # 初始 recurrent states（None 会在第一次调用时自动初始化为 Cache()）
    layer_past_query = [None] * num_layers
    layer_past_key_values = [None] * num_layers

    for t in range(T):
        query = torch.randn(B, query_len, config.hidden_size, device=device)
        keyval = torch.randn(B, keyval_len, config.hidden_size, device=device)

        for i, layer in enumerate(model):
            (query, _, layer_past_query[i], layer_past_key_values[i], _, _) = layer(
                query=query,
                keyval=keyval,
                frames=None,
                past_query=layer_past_query[i],
                past_key_values=layer_past_key_values[i],
                use_cache=True,
            )

        print(f"  Step {t}: output shape = {query.shape}")

    print(f"Recurrent inference test passed! Final output: {query.shape}")


if __name__ == "__main__":
    mom_block_test()
    cross_attn_test()
    # recurrent_inference_test()
