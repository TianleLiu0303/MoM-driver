from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from typing import Optional, Tuple, TYPE_CHECKING

from fla.layers import MomAttention  # 引入flash-linear-attention中的MomAttention

if TYPE_CHECKING:
    from fla.models.utils import Cache  # 引入Cache用于缓存

# ============================================================
# Self-Attention using MomAttention (from flash-linear-attention)
# ============================================================

class MoMAttentionSelf(nn.Module):
    """
    Self-attention using flash-linear-attention's MoM mechanism.
    """

    def __init__(
        self,
        mode: str = "chunk",  # 默认使用 "chunk"
        hidden_size: int = 256,
        num_heads: Optional[int] = 4,
        layer_idx: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.layer_idx = layer_idx

        # 使用 flash-linear-attention 中的 MomAttention
        self.mom_attention = MomAttention(
            mode=mode,
            hidden_size=hidden_size,
            num_heads=num_heads,
            **kwargs
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass using MomAttention.
        """
        # 通过 flash-linear-attention 中的 MomAttention 计算输出
        output, _, past_key_values, router_logits = self.mom_attention(
            hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            # 移除 mode=mode 参数，让 MomAttention 内部处理
            **kwargs
        )

        return output, None, past_key_values, router_logits


# ============================================================
# Cross-Attention (Custom Implementation)
# ============================================================

class MoMAttentionCross(nn.Module):
    """
    Custom Cross-attention mechanism conditioned on query using `MomAttention` for self-attention.
    """

    def __init__(
        self,
        mode: str = "chunk",  # 默认使用 "chunk"
        hidden_size: int = 256,
        num_heads: Optional[int] = 4,
        cond_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__()

        self.mode = mode
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.cond_scale = cond_scale

        # 使用 flash-linear-attention 中的 MomAttention 来实现 self-attention
        self.mom_attention = MomAttention(
            mode=mode,
            hidden_size=hidden_size,
            num_heads=num_heads,
            **kwargs
        )

        # Query-to-conditioning projection (learnable)
        self.cond_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        query: torch.Tensor,                 # [B, Q, D]
        keyval: torch.Tensor,                # [B, K, D]
        attention_mask_kv: Optional[torch.Tensor] = None,  # [B, K], optional
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache], Optional[torch.Tensor]]:
        B, Q, D = query.shape
        B2, K, D2 = keyval.shape
        assert B == B2 and D == D2 == self.hidden_size

        # 扩展 keyval (类似 RWKV7 CrossAttention)
        expanded_keyval = keyval.unsqueeze(1).expand(B, Q, K, D).reshape(B * Q, K, D)

        # 对 query 进行条件投影 [B*Q, 1, D] -> [B*Q, K, D]
        q_tok = query.reshape(B * Q, 1, D)
        cond = self.cond_proj(q_tok) * self.cond_scale
        hidden_states = expanded_keyval + cond  # 将 query token 用于 condition keyval

        # 如果提供了 kv mask，扩展 kv mask
        if attention_mask_kv is not None:
            assert attention_mask_kv.shape == (B, K)
            expanded_mask = attention_mask_kv.unsqueeze(1).expand(B, Q, K).reshape(B * Q, K)
        else:
            expanded_mask = None

        # 使用 MomAttention 进行 self-attention
        # 移除 mode 参数，让 MomAttention 内部处理
        output, _, _, router_logits = self.mom_attention(
            hidden_states,
            attention_mask=expanded_mask,
            use_cache=False,
            # 移除 mode=self.mode 参数
            **kwargs
        )

        # 选择最后一个 token 作为 cross-attention 的输出
        output = output[:, -1, :]  # [B*Q, D] -> [B, Q, D]
        output = output.reshape(B, Q, D)  # 确保形状正确

        return output, None, None, router_logits


# ============================================================
# Quick tests to check functionality
# ============================================================

def mom_self_attn_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    attn = MoMAttentionSelf(
        hidden_size=256,
        num_heads=4,
        layer_idx=0,
        mode="chunk",
    ).to(device=device, dtype=dtype)
    
    # 设置为评估模式进行测试
    attn.eval()

    B, T, D = 4, 31, 256
    x = torch.randn(B, T, D, device=device, dtype=dtype)

    with torch.no_grad():  # 评估模式下不需要梯度
        y, _, _, router_logits = attn(x, attention_mask=None, past_key_values=None, use_cache=False)
    
    print("[MoM Self-Attn] Success!")
    print("Output Shape:", y.shape)
    if router_logits is not None:
        print("Router logits Shape:", router_logits.shape)
    else:
        print("Router logits: None")


def mom_cross_attn_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    cross = MoMAttentionCross(
        hidden_size=256,
        num_heads=4,
        mode="chunk",
        cond_scale=1.0,
    ).to(device=device, dtype=dtype)
    
    # 设置为评估模式进行测试
    cross.eval()

    B, Q, K, D = 4, 31, 641, 256
    query = torch.randn(B, Q, D, device=device, dtype=dtype)
    keyval = torch.randn(B, K, D, device=device, dtype=dtype)

    kv_mask = torch.ones(B, K, device=device, dtype=torch.int32)

    with torch.no_grad():  # 评估模式下不需要梯度
        y, _, _, router_logits = cross(query, keyval, attention_mask_kv=kv_mask)
    
    print("[MoM Cross-Attn] Success!")
    print("Output Shape:", y.shape)
    if router_logits is not None:
        print("Router logits Shape:", router_logits.shape)
    else:
        print("Router logits: None")


if __name__ == "__main__":
    print("Testing MoM Attention modules...")
    print("=" * 50)
    mom_self_attn_test()
    print("=" * 50)
    mom_cross_attn_test()
    print("=" * 50)
    print("All tests completed!")