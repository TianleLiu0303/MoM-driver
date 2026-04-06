"""
MoM backbone: DINOv2+LoRA image encoder + MoM temporal fusion.

Replaces VoVNet/Hydra backbone with DINOv2+LoRA per-camera encoding,
then fuses multi-frame scene tokens via MoM blocks (from fla library).

4 cameras (f0, b0, l0, r0), each producing num_scene_tokens (16) tokens
per frame. 10 frames total. Receives unified image tensor [B, seq_len, N, C, H, W].
"""

import logging
import math
from dataclasses import asdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.vision_transformer import VisionTransformer
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import save_file
from torch.nn.parameter import Parameter
from PIL import Image

from fla.models.mom.configuration_mom import MomConfig as FlaMomConfig
from fla.models.mom.modeling_mom import MomBlock
from fla.models.utils import Cache

logger = logging.getLogger(__name__)


def build_fla_mom_config(block_config, num_hidden_layers: int) -> FlaMomConfig:
    """Build fla MomConfig from a MoMBlockConfig dataclass.

    :param block_config: MoMBlockConfig instance whose fields match FlaMomConfig
    :param num_hidden_layers: number of MoM layers to instantiate
    :return: FlaMomConfig for constructing MomBlock instances
    """
    params = asdict(block_config)
    params["num_hidden_layers"] = num_hidden_layers
    return FlaMomConfig(**params)


# ============================================================================
# GridMask augmentation (adapted from DrivoR)
# ============================================================================

class GridMask(nn.Module):
    """Grid masking augmentation for training regularization."""

    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.0):
        super().__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x
        n, c, h, w = x.size()
        x = x.view(-1, h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0

        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

        mask = torch.tensor(mask, dtype=x.dtype, device=x.device)
        if self.mode == 1:
            mask = 1 - mask
        mask = mask.expand_as(x)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).to(x.dtype).to(x.device)
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask

        return x.view(n, c, h, w)


# ============================================================================
# DINOv2 + LoRA components (adapted from DrivoR dinov2_lora.py)
# ============================================================================

class timm_ViT(VisionTransformer):
    """Custom VisionTransformer that supports scene token prepending."""

    def _pos_embed(self, x: torch.Tensor, scene_tokens: torch.Tensor = None) -> torch.Tensor:
        if self.pos_embed is None:
            return x.view(x.shape[0], -1, x.shape[-1])

        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                new_size=(H, W),
                old_size=self.patch_embed.grid_size,
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        if scene_tokens is not None:
            x = torch.cat([scene_tokens, x], dim=1)

        return self.pos_drop(x)

    def forward_features(self, x: torch.Tensor, scene_tokens: torch.Tensor = None,
                         attn_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x, scene_tokens)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        if attn_mask is not None:
            for blk in self.blocks:
                x = blk(x, attn_mask=attn_mask)
        elif self.grad_checkpointing and not torch.jit.is_scripting():
            from torch.utils.checkpoint import checkpoint_sequential
            x = checkpoint_sequential(self.blocks, len(self.blocks), x)
        else:
            x = self.blocks(x)

        x = self.norm(x)
        return x


class _LoRA_qkv_timm(nn.Module):
    """LoRA adapter for timm ViT's qkv linear layer."""

    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v,
                 linear_a_k, linear_b_k, layer_norm_q=None, layer_norm_v=None,
                 layer_norm_k=None):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k
        self.dim = qkv.in_features

        self.layernorm_q = layer_norm_q
        self.layernorm_v = layer_norm_v
        self.layernorm_k = layer_norm_k

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(self.layernorm_q(x)))
        new_v = self.linear_b_v(self.linear_a_v(self.layernorm_v(x)))
        qkv[:, :, :self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv


class LoRA_ViT_timm(nn.Module):
    """LoRA-wrapped ViT from timm."""

    def __init__(self, vit_model: timm_ViT, r: int, lora_layer=None):
        super().__init__()

        if r == 0:
            for param in vit_model.parameters():
                param.requires_grad = False
            self.lora_vit = vit_model
        else:
            if lora_layer:
                self.lora_layer = lora_layer
            else:
                self.lora_layer = list(range(len(vit_model.blocks)))

            self.w_As = []
            self.w_Bs = []

            for param in vit_model.parameters():
                param.requires_grad = False

            for t_layer_i, blk in enumerate(vit_model.blocks):
                if t_layer_i not in self.lora_layer:
                    continue
                w_qkv_linear = blk.attn.qkv
                self.dim = w_qkv_linear.in_features
                w_a_linear_q = nn.Linear(self.dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, self.dim, bias=False)
                w_a_linear_v = nn.Linear(self.dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, self.dim, bias=False)
                w_a_linear_k = nn.Identity()
                w_b_linear_k = nn.Identity()
                layer_norm_q = nn.Identity()
                layer_norm_v = nn.Identity()
                layer_norm_k = nn.Identity()

                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)

                blk.attn.qkv = _LoRA_qkv_timm(
                    w_qkv_linear,
                    w_a_linear_q, w_b_linear_q,
                    w_a_linear_v, w_b_linear_v,
                    w_a_linear_k, w_b_linear_k,
                    layer_norm_q, layer_norm_v, layer_norm_k,
                )

            self.reset_parameters()
            self.lora_vit = vit_model

    def reset_parameters(self):
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, x, scene_tokens=None):
        return self.lora_vit.forward_features(x, scene_tokens)


class ImgEncoder(nn.Module):
    """DINOv2+LoRA image encoder.

    Adapted from DrivoR's ImgEncoder but made self-contained (no DrivoR imports).
    Extracts `num_scene_tokens` scene tokens per camera via learnable prefix tokens.
    """

    model_names = (
        "timm/vit_small_patch14_dinov2.lvd142m",
        "timm/vit_base_patch14_dinov2.lvd142m",
        "timm/vit_large_patch14_dinov2.lvd142m",
        "timm/vit_giant_patch14_dinov2.lvd142m",
        "timm/vit_small_patch14_reg4_dinov2.lvd142m",
        "timm/vit_base_patch14_reg4_dinov2.lvd142m",
        "timm/vit_large_patch14_reg4_dinov2.lvd142m",
        "timm/vit_giant_patch14_reg4_dinov2.lvd142m",
        "timm/vit_small_patch16_dinov3.lvd1689m",
        "timm/vit_large_patch16_dinov3.lvd1689m",
    )

    def __init__(self, config):
        super().__init__()

        model_name = config.image_backbone_model_name
        self.num_prefix_tokens = config.num_scene_tokens
        if model_name not in self.model_names:
            raise ValueError(f"Unknown model name: {repr(model_name)}")
        else:
            print("loading ", model_name)

        pretrained_cfg_overlay = {"file": config.image_backbone_model_weights}
        in_chans = 3

        # numpy pickle compatibility hack
        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        try:
            try:
                self.model = timm.create_model(
                    model_name,
                    pretrained=True,
                    pretrained_cfg_overlay=pretrained_cfg_overlay,
                    img_size=(config.image_size[1], config.image_size[0]),
                    num_classes=0,
                    in_chans=in_chans,
                )
            except Exception:
                self.model = timm.create_model(
                    model_name,
                    pretrained=True,
                    img_size=(config.image_size[1], config.image_size[0]),
                    num_classes=0,
                    in_chans=in_chans,
                )
        finally:
            np.load = np_load_old

        self.model.__class__ = timm_ViT
        self.patch_size = self.model.patch_embed.patch_size[0]
        self.use_lora = config.image_backbone_use_lora
        self.finetune = config.image_backbone_finetune

        self.neck = nn.Linear(self.model.num_features, config.tf_d_model)
        self.num_features = self.model.num_features

        if self.use_lora:
            self.model = LoRA_ViT_timm(self.model, r=config.image_backbone_lora_rank)
        elif self.finetune:
            for param in self.model.parameters():
                param.requires_grad = True
            self.model.train()
        else:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = True

        self.use_feature_pooling = config.image_backbone_use_feature_pooling
        if self.use_feature_pooling:
            self.pool_proj = nn.AdaptiveAvgPool1d(self.num_prefix_tokens)

        self.focus_front_cam = config.image_backbone_focus_front_cam
        self.compress_fc = config.image_backbone_compress_fc
        if self.focus_front_cam and not self.compress_fc:
            raise ValueError(
                "MoM ImgEncoder only supports focus_front_cam when image_backbone_compress_fc=True."
            )
        if self.compress_fc:
            self.compress_fc_layer = nn.Linear(3957, self.num_prefix_tokens)

    def forward(self, img: torch.Tensor, scene_tokens: torch.Tensor) -> torch.Tensor:
        """
        :param img: [B, N, C, H, W] — N cameras
        :param scene_tokens: [B, N, num_scene_tokens, num_features] — learnable
        :return: [B, N*num_scene_tokens, tf_d_model]
        """
        B, N, C, H, W = img.size()
        img = rearrange(img, 'b n c h w -> (b n) c h w')
        if self.use_grid_mask:
            img = self.grid_mask(img)

        scene_tokens = rearrange(scene_tokens, 'b n t c -> (b n) t c')

        if self.use_lora:
            tokens = self.model(img, scene_tokens)
        elif self.finetune:
            tokens = self.model.forward_features(img, scene_tokens)
        else:
            tokens = self.model.forward_features(img, scene_tokens)

        if self.use_feature_pooling:
            tokens = self.pool_proj(tokens.transpose(1, 2)).transpose(1, 2)
        elif self.focus_front_cam:
            tokens = rearrange(tokens, '(b n) t c -> b n t c', b=B, n=N)
            front = tokens[:, 0, :, :]
            front = self.compress_fc_layer(front.transpose(1, 2)).transpose(1, 2)
            others = tokens[:, 1:, :self.num_prefix_tokens, :].reshape(B, -1, self.num_features)
            tokens = torch.cat([front, others], dim=1)
        elif self.num_prefix_tokens > 0:
            tokens = tokens[:, :self.num_prefix_tokens]

        tokens = self.neck(tokens)
        if not self.focus_front_cam:
            tokens = rearrange(tokens, '(b n) t c -> b (n t) c', b=B, n=N)

        return tokens


# ============================================================================
# MoM Temporal Fusion backbone
# ============================================================================

class MoMBackbone(nn.Module):
    """DINOv2+LoRA image encoder + MoM temporal fusion backbone.

    Pipeline:
      1. Receive unified image tensor [B, seq_len, N, C, H, W]
      2. Reshape to [B*seq_len, N, C, H, W] for DINOv2
      3. DINOv2+LoRA with scene_embeds → [B*seq_len, N*16, 256]
      4. Reshape to [B, seq_len, tokens_per_frame, 256]
      5. Add per-frame positional embedding [1, 1, tokens_per_frame, 256]
      6. Reshape to [B, seq_len*tokens_per_frame, 256]
      7. Build attention_mask from valid_frame_len
      8. Apply MomBlock layers with attention_mask
      9. LayerNorm → scene_features [B, seq_len*tokens_per_frame, 256]
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_len = config.cam_seq_len
        self.num_cams = config.num_cams
        self.num_scene_tokens = config.num_scene_tokens
        if config.image_backbone_focus_front_cam and not config.image_backbone_compress_fc:
            raise ValueError(
                "MoMBackbone requires image_backbone_compress_fc=True when image_backbone_focus_front_cam=True."
            )
        self.tokens_per_frame = self.num_cams * self.num_scene_tokens  # 4 * 16 = 64
        self.d_model = config.tf_d_model

        # --- DINOv2+LoRA image encoder (reads config directly) ---
        self.image_encoder = ImgEncoder(config)

        # Learnable scene embed tokens: [1, num_cams, num_scene_tokens, vit_features]
        self.scene_embeds = nn.Parameter(
            torch.randn(1, self.num_cams, self.num_scene_tokens, self.image_encoder.num_features) * 1e-6,
            requires_grad=True,
        )

        # --- MoM temporal fusion ---
        # Per-frame positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, self.tokens_per_frame, self.d_model))
        self.drop = nn.Dropout(config.embd_pdrop)

        # Build fla MomConfig directly from config.mom_temporal
        num_layers = config.mom_temporal_num_layers
        fla_config = build_fla_mom_config(config.mom_temporal, num_hidden_layers=num_layers)

        self.mom_blocks = nn.ModuleList([
            MomBlock(fla_config, layer_idx=i)
            for i in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(self.d_model)

        # Recurrent caches: one Cache per MomBlock layer, length == mom_temporal_num_layers.
        # Only used when use_past_key_cache=True; reset between scenes via reset_cache().
        self.use_past_key_cache = getattr(config, "use_past_key_cache", False)
        self.past_key_caches: list = [None] * num_layers

        # Init weights for custom layers (not DINOv2)
        self._init_custom_weights()

    def reset_cache(self) -> None:
        """Reset temporal recurrent states. Call at the start of each new scene."""
        self.past_key_caches = [None] * len(self.mom_blocks)

    def _init_custom_weights(self):
        """Initialize pos_emb, drop, mom_blocks, ln_f with standard init."""
        nn.init.normal_(self.pos_emb, mean=0.0, std=0.02)
        self.ln_f.bias.data.zero_()
        self.ln_f.weight.data.fill_(1.0)

    def forward(
        self,
        image: torch.Tensor,
        valid_frame_len: torch.Tensor,
        return_router_logits: bool = False,
    ) -> torch.Tensor:
        """
        :param image: [B, seq_len, N, C, H, W] — unified multi-cam multi-frame tensor
        :param valid_frame_len: [B] — number of valid (non-padded) frames per sample
        :param return_router_logits: if True, also return list of router_logits per MomBlock layer.
            Each element has shape [B, seq_len*tokens_per_frame, num_memories].
            Used for memory slot routing visualization (analysis only). Default False.
        :return: scene_features [B, seq_len*tokens_per_frame, d_model]
            or (scene_features, router_logits_list) when return_router_logits=True
        """
        B, seq_len, num_cams, C, H, W = image.shape

        # Step 1: Reshape to [B*seq_len, N, C, H, W] for DINOv2
        img = image.reshape(B * seq_len, num_cams, C, H, W)

        # Step 2: Prepare scene tokens and run DINOv2
        scene_tokens = self.scene_embeds.expand(B * seq_len, -1, -1, -1)  # [B*seq_len, N, 16, vit_feat]
        x = self.image_encoder(img, scene_tokens)  # [B*seq_len, N*16, d_model]

        # Step 3: Reshape to [B, seq_len, tokens_per_frame, d_model]
        x = x.reshape(B, seq_len, self.tokens_per_frame, self.d_model)

        # Step 4: Add per-frame positional embedding (broadcast across frames)
        x = x + self.pos_emb.unsqueeze(1)  # pos_emb: [1, 1, tpf, d_model]

        # Step 5: Reshape to [B, seq_len*tokens_per_frame, d_model]
        x = x.reshape(B, seq_len * self.tokens_per_frame, self.d_model)
        x = self.drop(x)

        # Step 6: Apply MomBlock layers
        all_router_logits = []
        if self.use_past_key_cache:
            # Streaming mode: pass recurrent state per layer, no attention mask needed
            for i, block in enumerate(self.mom_blocks):
                if self.past_key_caches[i] is None:
                    self.past_key_caches[i] = Cache()
                x, _, self.past_key_caches[i], router_logits = block(
                    x, past_key_values=self.past_key_caches[i], use_cache=True
                )
                if return_router_logits and router_logits is not None:
                    all_router_logits.append(router_logits)
        else:
            # Standard mode: build attention mask and process all frames at once
            attention_mask = self._build_attention_mask(valid_frame_len, B, x.device)
            for block in self.mom_blocks:
                x, _, _, router_logits = block(x, attention_mask=attention_mask)
                if return_router_logits and router_logits is not None:
                    all_router_logits.append(router_logits)

        # Step 7: LayerNorm
        x = self.ln_f(x)

        if return_router_logits:
            # router_logits per layer: raw logits before softmax, shape [B, seq_len*tpf, num_memories]
            # softmax across dim=-1 gives per-token routing probability to each memory slot
            return x, all_router_logits
        return x

    def _build_attention_mask(
        self,
        valid_frame_len: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Build attention mask for MoM temporal fusion.

        Left-padded frames → 0, valid frames → 1.
        Mask shape: [B, total_tokens] where total_tokens = seq_len * tokens_per_frame.

        :param valid_frame_len: [B] tensor of valid frame counts
        :param batch_size: batch size
        :param device: tensor device
        :return: [B, total_tokens] int mask
        """
        total_tokens = self.seq_len * self.tokens_per_frame

        # start_indices: first valid token index per sample
        start_indices = (self.seq_len - valid_frame_len) * self.tokens_per_frame
        start_indices = start_indices.unsqueeze(1).to(device)  # [B, 1]

        range_tensor = torch.arange(total_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
        attention_mask = (range_tensor >= start_indices).int()

        return attention_mask
