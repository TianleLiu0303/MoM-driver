"""
Implements the MoM vision backbone (replacing RWKV7).
"""

import copy
import timm
import torch
import torch.nn.functional as F
from torch import nn

from navsim.agents.Mom_mf.mom_block_fla import MomBlock  # 使用 MomBlock 而不是 DoubleSelfAttnBlock
from navsim.agents.Mom_mf.mom_config import MomConfig


class MoMBackbone(nn.Module):
    """Multi-scale Fusion MoM Model for image + LiDAR feature fusion."""

    def __init__(self, config: MomConfig):
        super().__init__()
        self.config = config

        self.image_encoder = timm.create_model(config.image_architecture, pretrained=True, features_only=True)
        if config.use_ground_plane:
            in_channels = 2 * config.lidar_seq_len
        else:
            in_channels = 1

        if config.latent:
            self.lidar_latent = nn.Parameter(
                torch.randn(
                    (1, in_channels, config.lidar_resolution_width, config.lidar_resolution_height),
                    requires_grad=True,
                )
            )

        self.avgpool_img = nn.AdaptiveAvgPool2d((self.config.img_vert_anchors, self.config.img_horz_anchors))

        self.lidar_encoder = timm.create_model(
            config.lidar_architecture,
            pretrained=False,
            in_chans=in_channels,
            features_only=True,
        )
        self.global_pool_lidar = nn.AdaptiveAvgPool2d(output_size=1)
        self.avgpool_lidar = nn.AdaptiveAvgPool2d((self.config.lidar_vert_anchors, self.config.lidar_horz_anchors))
        lidar_time_frames = [config.lidar_seq_len for i in range(4)]

        self.global_pool_img = nn.AdaptiveAvgPool2d(output_size=1)
        start_index = 0
        # Some networks have a stem layer
        if len(self.image_encoder.return_layers) > 4:
            start_index += 1

        # 使用 MomBlock 替换 RWKV7Block
        self.moms = nn.ModuleList(
            [
                GPT_MoM(
                    n_embd=self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                    config=config,
                    lidar_time_frames=lidar_time_frames[i],
                )
                for i in range(4)
            ]
        )
        
        self.lidar_channel_to_img = nn.ModuleList(
            [
                nn.Conv2d(
                    self.lidar_encoder.feature_info.info[start_index + i]["num_chs"],
                    self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                    kernel_size=1,
                )
                for i in range(4)
            ]
        )
        self.img_channel_to_lidar = nn.ModuleList(
            [
                nn.Conv2d(
                    self.image_encoder.feature_info.info[start_index + i]["num_chs"],
                    self.lidar_encoder.feature_info.info[start_index + i]["num_chs"],
                    kernel_size=1,
                )
                for i in range(4)
            ]
        )

        self.num_image_features = self.image_encoder.feature_info.info[start_index + 3]["num_chs"]
        self.perspective_upsample_factor = (
            self.image_encoder.feature_info.info[start_index + 3]["reduction"]
            // self.config.perspective_downsample_factor
        )

        if self.config.transformer_decoder_join:
            self.num_features = self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"]
        else:
            if self.config.add_features:
                self.lidar_to_img_features_end = nn.Linear(
                    self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"],
                    self.image_encoder.feature_info.info[start_index + 3]["num_chs"],
                )
                self.num_features = self.image_encoder.feature_info.info[start_index + 3]["num_chs"]
            else:
                self.num_features = (
                    self.image_encoder.feature_info.info[start_index + 3]["num_chs"]
                    + self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"]
                )

        # FPN fusion
        channel = self.config.bev_features_channels
        self.relu = nn.ReLU(inplace=True)
        if self.config.detect_boxes or self.config.use_bev_semantic:
            self.upsample = nn.Upsample(
                scale_factor=self.config.bev_upsample_factor, mode="bilinear", align_corners=False
            )
            self.upsample2 = nn.Upsample(
                size=(
                    self.config.lidar_resolution_height // self.config.bev_down_sample_factor,
                    self.config.lidar_resolution_width // self.config.bev_down_sample_factor,
                ),
                mode="bilinear",
                align_corners=False,
            )

            self.up_conv5 = nn.Conv2d(channel, channel, (3, 3), padding=1)
            self.up_conv4 = nn.Conv2d(channel, channel, (3, 3), padding=1)
            self.c5_conv = nn.Conv2d(self.lidar_encoder.feature_info.info[start_index + 3]["num_chs"], channel, (1, 1))

    def top_down(self, x):
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample2(p4)))
        return p3

    def forward(self, image, lidar):
        """
        Image + LiDAR feature fusion using MoM blocks
        Args:
            image_list (list): list of input images
            lidar_list (list): list of input LiDAR BEV
        """
        image_features, lidar_features = image, lidar

        if self.config.latent:
            batch_size = lidar.shape[0]
            lidar_features = self.lidar_latent.repeat(batch_size, 1, 1, 1)

        image_layers = iter(self.image_encoder.items())
        lidar_layers = iter(self.lidar_encoder.items())

        # Stem layer
        if len(self.image_encoder.return_layers) > 4:
            image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)
        if len(self.lidar_encoder.return_layers) > 4:
            lidar_features = self.forward_layer_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)

        # Loop through the 4 blocks
        for i in range(4):
            image_features = self.forward_layer_block(image_layers, self.image_encoder.return_layers, image_features)
            lidar_features = self.forward_layer_block(lidar_layers, self.lidar_encoder.return_layers, lidar_features)
            image_features, lidar_features = self.fuse_features(image_features, lidar_features, i)

        if self.config.detect_boxes or self.config.use_bev_semantic:
            x4 = lidar_features

        image_feature_grid = None
        if self.config.use_semantic or self.config.use_depth:
            image_feature_grid = image_features

        if self.config.transformer_decoder_join:
            fused_features = lidar_features
        else:
            image_features = self.global_pool_img(image_features)
            image_features = torch.flatten(image_features, 1)
            lidar_features = self.global_pool_lidar(lidar_features)
            lidar_features = torch.flatten(lidar_features, 1)

            if self.config.add_features:
                lidar_features = self.lidar_to_img_features_end(lidar_features)
                fused_features = image_features + lidar_features
            else:
                fused_features = torch.cat((image_features, lidar_features), dim=1)

        if self.config.detect_boxes or self.config.use_bev_semantic:
            features = self.top_down(x4)
        else:
            features = None

        return features, fused_features, image_feature_grid

    def forward_layer_block(self, layers, return_layers, features):
        for name, module in layers:
            features = module(features)
            if name in return_layers:
                break
        return features

    def fuse_features(self, image_features, lidar_features, layer_idx):
        """
        Perform feature fusion using MomBlock (类似 RWKV7Block 的用法)
        """
        image_embd_layer = self.avgpool_img(image_features)
        lidar_embd_layer = self.avgpool_lidar(lidar_features)

        lidar_embd_layer = self.lidar_channel_to_img[layer_idx](lidar_embd_layer)

        # 使用 MoM 进行融合
        image_features_layer, lidar_features_layer = self.moms[layer_idx](image_embd_layer, lidar_embd_layer)
        
        lidar_features_layer = self.img_channel_to_lidar[layer_idx](lidar_features_layer)

        image_features_layer = F.interpolate(
            image_features_layer,
            size=(image_features.shape[2], image_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )
        lidar_features_layer = F.interpolate(
            lidar_features_layer,
            size=(lidar_features.shape[2], lidar_features.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        image_features = image_features + image_features_layer
        lidar_features = lidar_features + lidar_features_layer

        return image_features, lidar_features


class GPT_MoM(nn.Module):
    """The full MoM backbone, replacing RWKV."""

    def __init__(self, n_embd, config: MomConfig, lidar_time_frames):
        super().__init__()

        self.n_embd = n_embd
        self.config = config

        mom_config = copy.copy(config)
        mom_config.hidden_size = n_embd
        if mom_config.num_heads > 0:
            mom_config.head_dim = n_embd // mom_config.num_heads

        self.seq_len = config.cam_seq_len
        self.lidar_seq_len = config.lidar_seq_len
        self.lidar_time_frames = lidar_time_frames

        self.img_feature_size = config.img_vert_anchors * config.img_horz_anchors
        self.lidar_feature_size = config.lidar_vert_anchors * config.lidar_horz_anchors

        # Positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, self.img_feature_size + self.lidar_feature_size, self.n_embd))
        # Temporal embedding
        # self.time_emb = nn.Parameter(torch.zeros(1, self.seq_len, self.n_embd))

        self.drop = nn.Dropout(config.embd_pdrop)

        # MoM Blocks (替换 RWKV7Block)
        self.blocks = nn.ModuleList(
            [ 
                MomBlock(
                    config=mom_config,
                    layer_idx=layer_idx
                ) 
                for layer_idx in range(config.num_layers)
            ]
        )
        
        # Decoder head
        self.ln_f = nn.LayerNorm(n_embd)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=self.config.gpt_linear_layer_init_mean,
                std=self.config.gpt_linear_layer_init_std,
            )
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(self.config.gpt_layer_norm_init_weight)

    def forward(self, image_tensor, lidar_tensor):
        """
        Args:
            image_tensor (tensor): B * seq_len, C, H, W
            lidar_tensor (tensor): B * seq_len, C, H, W
        """
        expanded_batch_size = image_tensor.shape[0]
        batch_size = expanded_batch_size // self.seq_len
        lidar_h, lidar_w = lidar_tensor.shape[2:4]
        img_h, img_w = image_tensor.shape[2:4]

        # Reshape to sequence format
        image_tensor = image_tensor.permute(0, 2, 3, 1).contiguous().view(expanded_batch_size, -1, self.n_embd)
        lidar_tensor = lidar_tensor.permute(0, 2, 3, 1).contiguous().view(expanded_batch_size, -1, self.n_embd)
        token_embeddings = torch.cat((image_tensor, lidar_tensor), dim=1).contiguous()
        # token_embeddings = token_embeddings.view(batch_size, -1, self.n_embd)

        # Add embeddings
        # embedding = self.pos_emb.unsqueeze(1) + self.time_emb.unsqueeze(2)
        # embedding = embedding.view(1, -1, self.n_embd)

        # x = self.drop(embedding + token_embeddings)  # (B, seq_len * T, C)
        x = self.drop(self.pos_emb + token_embeddings)  # (B * seq_len, T, C)
        x = x.view(batch_size, -1, self.n_embd)

        # Process through MoM blocks
        # 使用 gradient checkpointing 来节省内存（如果在训练模式）
        for block in self.blocks:
            # MomBlock 的 forward 返回 (hidden_states, attentions, past_key_values, router_logits)
            if self.training and hasattr(self.config, 'use_gradient_checkpointing') and self.config.use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    None,  # attention_mask
                    None,  # past_key_values
                    False,  # use_cache
                    False,  # output_attentions
                    use_reentrant=False
                )
                x = x[0]  # 只取 hidden_states
            else:
                x, _, _, _ = block(
                    hidden_states=x,
                    attention_mask=None,
                    past_key_values=None,
                    use_cache=False,
                    output_attentions=False,
                )
        
        x = self.ln_f(x)

        # Restore original shape
        x = x.view(expanded_batch_size, -1, self.n_embd)
        image_tensor_out = (
            x[:, :self.img_feature_size, :]
            .view(expanded_batch_size, img_h, img_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        lidar_tensor_out = (
            x[:, self.img_feature_size :, :]
            .view(expanded_batch_size, lidar_h, lidar_w, -1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

        return image_tensor_out, lidar_tensor_out


def self_attn_test():
    """测试 MomBlock (对应原始的 RWKV7Block 测试)"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from navsim.agents.Mom_mf.mom_config import MomConfig
    
    config = MomConfig()
    config.hidden_size = 256
    config.num_heads = 4
    config.num_layers = 2
    
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
    
    B = 16
    seq_len = 31

    hidden_states = torch.randn(B, seq_len, 256).to(device)

    print("Testing MomBlock (替换 RWKV7Block)...")
    print(f"Input shape: {hidden_states.shape}")
    
    with torch.no_grad():
        for i, layer in enumerate(model):
            # MomBlock 返回 (hidden_states, attentions, past_key_values, router_logits)
            hidden_states, _, _, _ = layer(
                hidden_states=hidden_states,
                attention_mask=None,
                past_key_values=None,
                use_cache=False,
                output_attentions=False,
            )
            print(f"Layer {i} output shape: {hidden_states.shape}")

    print("\nSuccess!")
    print(f"Final output shape: {hidden_states.shape}")
    

def gpt_mom_fusion_test():
    """测试 GPT_MoM 的图像+LiDAR融合功能"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from navsim.agents.Mom_mf.mom_config import MomConfig
    
    config = MomConfig()
    config.cam_seq_len = 1  # 减小序列长度
    config.lidar_seq_len = 1
    config.img_vert_anchors = 4  # 减小锚点数量
    config.img_horz_anchors = 8
    config.lidar_vert_anchors = 4
    config.lidar_horz_anchors = 8
    config.num_layers = 1  # 减少层数
    config.hidden_size = 128  # 减小隐藏层维度
    config.num_heads = 4  # 确保 head 数量合理
    config.head_dim = 32  # 减小 head 维度
    config.embd_pdrop = 0.1
    config.gpt_linear_layer_init_mean = 0.0
    config.gpt_linear_layer_init_std = 0.02
    config.gpt_layer_norm_init_weight = 1.0
    
    n_embd = 128  # 匹配 hidden_size
    B = 2  # 减小 batch size
    seq_len = config.cam_seq_len
    
    model = GPT_MoM(
        n_embd=n_embd,
        config=config,
        lidar_time_frames=4
    )
    model.to(device)
    model.eval()
    
    # Image: (B*seq_len, C, H, W)
    img_h, img_w = config.img_vert_anchors, config.img_horz_anchors
    image_tensor = torch.randn(B * seq_len, n_embd, img_h, img_w).to(device)
    
    # LiDAR: (B*seq_len, C, H, W)
    lidar_h, lidar_w = config.lidar_vert_anchors, config.lidar_horz_anchors
    lidar_tensor = torch.randn(B * seq_len, n_embd, lidar_h, lidar_w).to(device)
    
    print("Testing GPT_MoM fusion...")
    print(f"Image input shape: {image_tensor.shape}")
    print(f"LiDAR input shape: {lidar_tensor.shape}")
    
    with torch.no_grad():
        image_out, lidar_out = model(image_tensor, lidar_tensor)
    
    print("\nSuccess!")
    print(f"Image output shape: {image_out.shape}")
    print(f"LiDAR output shape: {lidar_out.shape}")
    

if __name__ == "__main__":
    print("="*80)
    print("Test 1: MomBlock (替换 RWKV7Block)")
    print("="*80)
    self_attn_test()
    
    print("\n" + "="*80)
    print("Test 2: GPT_MoM Fusion")
    print("="*80)
    gpt_mom_fusion_test()
