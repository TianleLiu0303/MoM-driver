from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn

from fla.models.utils import Cache

from navsim.agents.rwkv7_mixed.rwkv_config import RWKVConfig
from navsim.agents.rwkv7_mixed.rwkv_fla_backbone import RWKVBackbone
from navsim.agents.rwkv7_mixed.rwkv_features import BoundingBox2DIndex
from navsim.agents.rwkv_block.rwkv7_block_fla import RWKV7_CrossAttBlock, RWKV7_DoubleSelfAttnBlock
from navsim.common.enums import StateSE2Index


class RWKVModel(nn.Module):
    """Torch module for RWKV."""

    def __init__(self, config: RWKVConfig):
        """
        Initializes RWKV torch module.
        :param config: global config dataclass of RWKV.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = RWKVBackbone(config)

        self._query_embedding = nn.Embedding(sum(self._query_splits), config.rwkv_d_model)
        self._keyval_len = config.lidar_seq_len * 8**2 + 1  # seq_len * 8x8 feature grid + trajectory
        self._feature_len = 8**2
        self._bev_time_embedding = nn.Parameter(torch.randn(1, config.lidar_seq_len, config.rwkv_d_model))
        self._bev_position_embedding = nn.Parameter(torch.randn(1, 8**2, config.rwkv_d_model))

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.rwkv_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.rwkv_d_model)
        self._status_embedding = nn.Parameter(torch.randn(1, config.rwkv_d_model))

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        self._rwkv_decoder = nn.ModuleList(
            [
                RWKV7_DoubleSelfAttnBlock(
                    hidden_size=config.rwkv_d_model,
                    norm_first=config.rwkv_d_norm_first,
                    num_heads=config.rwkv_d_num_head,
                    decay_low_rank_dim=config.decay_low_rank_dim,
                    gate_low_rank_dim=config.gate_low_rank_dim,
                    a_low_rank_dim=config.a_low_rank_dim,
                    v_low_rank_dim=config.v_low_rank_dim,
                    fuse_norm=config.fuse_norm,
                    n_layer=config.rwkv_d_num_layers,
                    layer_idx=i
                )
                for i in range(config.rwkv_d_num_layers)
            ]
        )
    
        self.fusion_drop = nn.Dropout(config.rwkv_fusion_drop)
        self.state_drop = nn.Dropout(config.rwkv_state_drop)
        self.ln_out = nn.LayerNorm(config.rwkv_d_model)

        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.rwkv_d_ffn,
            d_model=config.rwkv_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.rwkv_d_ffn,
            d_model=config.rwkv_d_model,
        )

    def forward(
            self, features: Dict[str, torch.Tensor],
            past_query: Optional[Cache] = None,
            past_key_values: Optional[Cache] = None,
            use_cache: Optional[bool] = False,
        ) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"] # (B, _)
        frames_feature: torch.Tensor = features["frames"].view(-1, )

        batch_size, seq_len, C, H, W = camera_feature.shape
        camera_feature = camera_feature.view(batch_size * seq_len, C, H, W) # (seq_len * B, _, _, _)

        batch_size, seq_len, C, H, W = lidar_feature.shape
        lidar_feature = lidar_feature.view(batch_size * seq_len, C, H, W) # (seq_len * B, _, _, _)

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)

        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)

        # multi frame
        expanded_batch_size, T, C = bev_feature.shape
        bev_feature = bev_feature.contiguous().view(batch_size, -1, C)
        bev_embedding = self._bev_time_embedding.unsqueeze(2) + self._bev_position_embedding.unsqueeze(1)
        bev_embedding = bev_embedding.view(1, -1, C)
        bev_feature = bev_feature + bev_embedding
        bev_padding = torch.zeros(batch_size, 1, C, device=bev_feature.device)
        bev_feature = torch.cat([bev_feature, bev_padding], dim=1)
        bev_feature = self.fusion_drop(bev_feature)

        status_encoding = self._status_encoding(status_feature)
        status_encoding = self.state_drop(status_encoding + self._status_embedding)
        bev_feature[torch.arange(batch_size), frames_feature * self._feature_len, :] = status_encoding
        keyval = bev_feature

        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = query
        query_v_first = None
        cross_query_v_first = None
        for block in self._rwkv_decoder:
            query_out, _, past_query, past_key_values, query_v_first, cross_query_v_first = block(
                query_out, keyval, 
                query_v_first, cross_query_v_first,
                frames_feature,
                past_query=past_query, 
                past_key_values=past_key_values, 
                use_cache=use_cache
            )
        query_out = self.ln_out(query_out)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)
        expanded_batch_size, C, H, W = bev_semantic_map.shape
        bev_semantic_map = bev_semantic_map.view(batch_size, -1, C, H, W)
        frames_indices = frames_feature - 1
        bev_semantic_map = bev_semantic_map[torch.arange(batch_size), frames_indices]

        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}
        trajectory = self._trajectory_head(trajectory_query)
        output.update(trajectory)

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output


class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}


class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, num_poses * StateSE2Index.size()),
        )

    def forward(self, object_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        poses = self._mlp(object_queries).reshape(-1, self._num_poses, StateSE2Index.size())
        poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
        return {"trajectory": poses}
