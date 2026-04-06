"""
MoM-based decoder blocks for trajectory refinement and scoring.

Replaces standard Transformer decoder (self-attn + cross-attn Block) with
MoM-based DoubleSelfAttnBlock:
  - Query self-attention via MomBlock
  - Cross-attention by concatenating [keyval, query] → MomBlock → extract query portion
  - FFN with residual

Two cross-attention modes (controlled by use_state_frozen in MoMConfig):

  LICA (use_state_frozen=False, baseline):
    concat([keyval, query]) → MomBlock → take last query_len tokens.
    Introduces two biases:
      ε_decay:       scene state is attenuated by query-position decay gates
      ε_contam:      query tokens write K/V into the shared state, so query_i
                     reads a state contaminated by earlier query_j (j < i)

  State-Frozen (use_state_frozen=True, proposed):
    Same concat layout but MomAttention receives frozen_from=kv_len so that
    at all query positions beta=0 (no write) and g=0 (decay=exp(0)=1, no
    attenuation).  Each query token independently reads the frozen scene
    state — equivalent to standard Transformer cross-attention semantics.
    This is the necessary and sufficient condition to eliminate both biases
    (Theorem 2 in the paper).

    Note: query-to-query communication is handled by the separate
    query_attn step (Step 1) before cross-attention, so State-Frozen does
    NOT remove inter-query interaction.

Fixes bugs from RWKV7_DoubleSelfAttnBlock:
  1. No hardcoded `64` for frame token length — computed dynamically
  2. No `.cuda()` calls — uses tensor's own device
  3. Proper v_first variable management
  4. Proper frame masking of invalid (left-padded) keyval tokens
"""

from dataclasses import asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.models.mom.configuration_mom import MomConfig as FlaMomConfig
from fla.models.mom.modeling_mom import MomBlock
from fla.models.utils import Cache

from navsim.agents.MoM_driver.mom_backbone import build_fla_mom_config


class MoMDoubleSelfAttnBlock(nn.Module):
    """MoM-based double self-attention block.

    Replaces RWKV7_DoubleSelfAttnBlock with MomBlock.

    Architecture:
      1. Query self-attention: query → MomBlock → updated query
      2. Cross-attention: concat([keyval, query]) → MomBlock → extract last query_len tokens
      3. Fused residual + FFN (handled inside MomBlock's mlp)

    Frame masking: When `frames` is provided, invalid (left-padded) keyval
    frames are zeroed out before concatenation with query.
    """

    def __init__(
        self,
        hidden_size: int,
        config,
        layer_idx: int = 0,
        tokens_per_frame: int = 64,
        use_state_frozen: bool = False,
        freeze_beta_only: bool = False,
        freeze_g_only: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.tokens_per_frame = tokens_per_frame
        self.use_state_frozen = use_state_frozen
        self.freeze_beta_only = freeze_beta_only
        self.freeze_g_only = freeze_g_only

        # Build MomConfig for decoder blocks from config.mom_decoder
        # Use 2 layers since we have two MomBlock instances (query + cross)
        fla_config = build_fla_mom_config(config.mom_decoder, num_hidden_layers=2)

        # MomBlock for query self-attention (Step 1)
        self.query_attn = MomBlock(fla_config, layer_idx=layer_idx * 2)
        # MomBlock for cross-attention (Step 3, supports State-Frozen)
        self.cross_attn = MomBlock(fla_config, layer_idx=layer_idx * 2 + 1)

    # Minimum sequence length for MomBlock in training mode.
    # MomBlock's shared_o uses fused_recurrent for seq_len <= 64, which
    # asserts chunk mode in training. We pad to exceed this threshold.
    _MIN_SEQ_LEN = 65

    def _pad_and_run_mom(
        self,
        mom_block: nn.Module,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        frozen_from: int = -1,
        freeze_beta_from: int = -1,
        freeze_g_from: int = -1,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
    ):
        """Run MomBlock, padding input if seq_len <= _MIN_SEQ_LEN to avoid
        fused_recurrent assertion in training mode.

        :param mom_block: MomBlock instance
        :param x: [B, seq_len, d_model]
        :param attention_mask: optional [B, seq_len] int mask
        :param frozen_from: if >= 0, positions [frozen_from, seq_len) are
            State-Frozen (beta=0, g=0). Padding does NOT shift this index
            because padding is appended AFTER the original tokens.
        :param freeze_beta_from: if >= 0, positions [freeze_beta_from, seq_len)
            have beta=0.
        :param freeze_g_from: if >= 0, positions [freeze_g_from, seq_len)
            have g=0.
        :param past_key_values: Cache for recurrent single-frame inference.
        :param use_cache: whether to use and return recurrent state.
        :return: ([B, seq_len, d_model], updated Cache or None)
        """
        if use_cache and past_key_values is None:
            past_key_values = Cache()

        seq_len = x.size(1)
        if self.training and seq_len <= self._MIN_SEQ_LEN:
            pad_len = self._MIN_SEQ_LEN + 1 - seq_len
            x_padded = F.pad(x, (0, 0, 0, pad_len))  # pad seq dim with zeros
            if attention_mask is not None:
                mask_padded = F.pad(attention_mask, (0, pad_len), value=0)
            else:
                # Mark original tokens as valid=1, padding as valid=0
                mask_padded = torch.ones(x.size(0), self._MIN_SEQ_LEN + 1,
                                         dtype=torch.int, device=x.device)
                mask_padded[:, seq_len:] = 0
            # frozen_from is an absolute index: unaffected by end-padding
            out, _, _, _ = mom_block(
                x_padded,
                attention_mask=mask_padded,
                frozen_from=frozen_from,
                freeze_beta_from=freeze_beta_from,
                freeze_g_from=freeze_g_from,
            )
            return out[:, :seq_len, :], None
        else:
            out, _, past_key_values, _ = mom_block(
                x,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                frozen_from=frozen_from,
                freeze_beta_from=freeze_beta_from,
                freeze_g_from=freeze_g_from,
            )
            return out, past_key_values

    def forward(
        self,
        query: torch.Tensor,
        keyval: torch.Tensor,
        frames: Optional[torch.Tensor] = None,
        past_query_cache: Optional[Cache] = None,
        past_cross_cache: Optional[Cache] = None,
        use_cache: bool = False,
    ):
        """
        :param query: [B, query_len, d_model] — trajectory tokens
        :param keyval: [B, kv_len, d_model] — scene features
        :param frames: [B] — valid frame counts (for masking left-padded frames in keyval)
        :param past_query_cache: recurrent Cache for query_attn (single-frame inference only)
        :param past_cross_cache: recurrent Cache for cross_attn (single-frame inference only)
        :param use_cache: whether to update and return caches
        :return: (query [B, query_len, d_model], past_query_cache, past_cross_cache)
                 last two are None when use_cache=False
        """
        # Step 1: Query self-attention (no State-Frozen — queries interact freely)
        query, past_query_cache = self._pad_and_run_mom(
            self.query_attn, query,
            past_key_values=past_query_cache, use_cache=use_cache,
        )

        # Step 2: If frames provided, mask invalid keyval frames
        if frames is not None:
            keyval = self._mask_invalid_frames(keyval, frames)

        # Step 3: Cross-attention via concat → MomBlock → extract query
        # MomBlock includes internal residual (input + attn + mlp), so
        # the query portion of the output already contains the original
        # query values. No external residual is needed.
        #
        # State-Frozen (use_state_frozen=True):
        #   frozen_from = kv_len tells MomAttention that all positions
        #   from kv_len onward are query positions where beta=0 and g=0.
        #   This eliminates both ε_decay and ε_contam (Theorem 1 & 2).
        # Fine-grained ablations:
        #   freeze_beta_only=True  -> beta=0 only
        #   freeze_g_only=True     -> g=0 only
        #
        # LICA baseline (use_state_frozen=False):
        #   frozen_from = -1, no masking applied (original behaviour).
        query_len = query.size(1)
        kv_len = keyval.size(1)
        if self.use_state_frozen:
            frozen_from = kv_len
            freeze_beta_from = -1
            freeze_g_from = -1
        else:
            frozen_from = -1
            freeze_beta_from = kv_len if self.freeze_beta_only else -1
            freeze_g_from = kv_len if self.freeze_g_only else -1

        qkv = torch.cat([keyval, query], dim=1)  # [B, kv_len + query_len, d_model]
        qkv, past_cross_cache = self._pad_and_run_mom(
            self.cross_attn,
            qkv,
            frozen_from=frozen_from,
            freeze_beta_from=freeze_beta_from,
            freeze_g_from=freeze_g_from,
            past_key_values=past_cross_cache,
            use_cache=use_cache,
        )
        query = qkv[:, -query_len:, :]  # extract query portion

        return query, past_query_cache, past_cross_cache

    def _mask_invalid_frames(
        self,
        keyval: torch.Tensor,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        """Mask invalid (left-padded) frame tokens in keyval.

        :param keyval: [B, kv_len, d_model]
        :param frames: [B] — valid frame count per sample
        :return: masked keyval with invalid frames zeroed out
        """
        if keyval.numel() == 0:
            return keyval

        if not torch.is_tensor(frames):
            frames = torch.tensor(frames, device=keyval.device)
        if frames.dim() == 0:
            frames = frames.view(1)
        frames = frames.to(device=keyval.device, dtype=torch.long)

        B, seq_len, _ = keyval.size()
        tpf = self.tokens_per_frame

        if seq_len < tpf:
            return keyval

        total_frames = seq_len // tpf
        if total_frames <= 0:
            return keyval

        # Clamp valid frames
        valid_frames = torch.clamp(frames, min=0, max=total_frames)
        invalid_frames = total_frames - valid_frames

        if torch.all(invalid_frames == 0):
            return keyval

        # Build per-frame mask: [B, total_frames]
        frame_ids = torch.arange(total_frames, device=keyval.device).unsqueeze(0)  # [1, total_frames]
        invalid_mask = frame_ids < invalid_frames.unsqueeze(1)  # [B, total_frames]

        # Expand to per-token mask: [B, total_frames * tpf]
        token_mask = invalid_mask.unsqueeze(-1).expand(-1, -1, tpf).reshape(B, -1)
        # Handle case where seq_len is not exactly total_frames * tpf
        token_mask = token_mask[:, :seq_len]

        return keyval.masked_fill(token_mask.unsqueeze(-1), 0.0)


class MoMTransformerDecoder(nn.Module):
    """Stack of MoMDoubleSelfAttnBlock layers for trajectory decoding.

    Returns intermediate outputs from each layer (for iterative refinement).
    """

    def __init__(self, config) -> None:
        super().__init__()
        num_layers = config.ref_num
        d_model = config.tf_d_model
        tokens_per_frame = config.num_cams * config.num_scene_tokens
        use_state_frozen = getattr(config, "use_state_frozen", False)
        freeze_beta_only = getattr(config, "freeze_beta_only", False)
        freeze_g_only = getattr(config, "freeze_g_only", False)
        self.use_past_key_cache = getattr(config, "use_past_key_cache", False)

        self.layers = nn.ModuleList([
            MoMDoubleSelfAttnBlock(
                hidden_size=d_model,
                config=config,
                layer_idx=i,
                tokens_per_frame=tokens_per_frame,
                use_state_frozen=use_state_frozen,
                freeze_beta_only=freeze_beta_only,
                freeze_g_only=freeze_g_only,
            )
            for i in range(num_layers)
        ])

        # Recurrent caches: one Cache per layer, for query_attn and cross_attn
        # Length == ref_num (== len(self.layers)), matching layer_idx layout.
        # Populated lazily on first use; reset between scenes via reset_cache().
        self.past_query_caches: list = [None] * num_layers
        self.past_cross_caches: list = [None] * num_layers

    def reset_cache(self) -> None:
        """Reset recurrent states. Call at the start of each new scene."""
        self.past_query_caches = [None] * len(self.layers)
        self.past_cross_caches = [None] * len(self.layers)

    def forward(
        self,
        x: torch.Tensor,
        x_cross: torch.Tensor,
        frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param x: query tokens [B, query_len, d_model]
        :param x_cross: scene features [B, kv_len, d_model]
        :param frames: [B] valid frame counts for masking
        :return: stacked intermediate outputs [num_layers, B, query_len, d_model]
        """
        intermediate = []
        for i, layer in enumerate(self.layers):
            if self.use_past_key_cache:
                x, self.past_query_caches[i], self.past_cross_caches[i] = layer(
                    x, x_cross, frames=frames,
                    past_query_cache=self.past_query_caches[i],
                    past_cross_cache=self.past_cross_caches[i],
                    use_cache=True,
                )
            else:
                x, _, _ = layer(x, x_cross, frames=frames)
            intermediate.append(x)
        return torch.stack(intermediate)


class MoMTransformerDecoderScorer(nn.Module):
    """Stack of MoMDoubleSelfAttnBlock layers for scoring.

    Returns only the final output (no intermediate outputs needed).
    """

    def __init__(self, num_layers: int, d_model: int, config) -> None:
        super().__init__()
        tokens_per_frame = config.num_cams * config.num_scene_tokens
        use_state_frozen = getattr(config, "use_state_frozen", False)
        freeze_beta_only = getattr(config, "freeze_beta_only", False)
        freeze_g_only = getattr(config, "freeze_g_only", False)
        self.use_past_key_cache = getattr(config, "use_past_key_cache", False)

        self.layers = nn.ModuleList([
            MoMDoubleSelfAttnBlock(
                hidden_size=d_model,
                config=config,
                layer_idx=i,
                tokens_per_frame=tokens_per_frame,
                use_state_frozen=use_state_frozen,
                freeze_beta_only=freeze_beta_only,
                freeze_g_only=freeze_g_only,
            )
            for i in range(num_layers)
        ])

        # Recurrent caches aligned with scorer_ref_num (== len(self.layers))
        self.past_query_caches: list = [None] * num_layers
        self.past_cross_caches: list = [None] * num_layers

    def reset_cache(self) -> None:
        """Reset recurrent states. Call at the start of each new scene."""
        self.past_query_caches = [None] * len(self.layers)
        self.past_cross_caches = [None] * len(self.layers)

    def forward(
        self,
        x: torch.Tensor,
        x_cross: torch.Tensor,
        frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param x: query tokens [B, query_len, d_model]
        :param x_cross: scene features [B, kv_len, d_model]
        :param frames: [B] valid frame counts for masking
        :return: final output [B, query_len, d_model]
        """
        for i, layer in enumerate(self.layers):
            if self.use_past_key_cache:
                x, self.past_query_caches[i], self.past_cross_caches[i] = layer(
                    x, x_cross, frames=frames,
                    past_query_cache=self.past_query_caches[i],
                    past_cross_cache=self.past_cross_caches[i],
                    use_cache=True,
                )
            else:
                x, _, _ = layer(x, x_cross, frames=frames)
        return x
