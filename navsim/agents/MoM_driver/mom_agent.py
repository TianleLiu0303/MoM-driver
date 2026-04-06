"""
MoM Agent: Training and inference interface.

Sensor config: 4 cameras (f0, b0, l0, r0) across 10 frames.
"""

from typing import Dict, Optional, List

import os
from pathlib import Path
import numpy as np

import torch

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.common.dataloader import MetricCacheLoader
from navsim.agents.score_module.compute_navsim_score import get_scores

try:
    from .mom_model import MoMModel
    from .mom_features import MoMFeatureBuilder, MoMTargetBuilder
    from .layers.losses.mom_loss import MoMDrivoRLoss
except ImportError:
    from navsim.agents.MoM_driver.mom_model import MoMModel
    from navsim.agents.MoM_driver.mom_features import MoMFeatureBuilder, MoMTargetBuilder
    from navsim.agents.MoM_driver.layers.losses.mom_loss import MoMDrivoRLoss


class MoMAgent(AbstractAgent):
    """Agent interface for MoM: DINOv2+LoRA + MoM fusion + DrivoR heads."""

    def __init__(self, config, checkpoint_path: Optional[str] = None, lr: Optional[float] = None):
        super().__init__()
        self._config = config
        self._checkpoint_path = checkpoint_path
        self._lr = lr
        self._mom_model = MoMModel(config)
        self._loss = MoMDrivoRLoss(
            trajectory_weight=config.trajectory_weight,
            inter_weight=config.inter_weight,
            sub_score_weight=config.sub_score_weight,
            final_score_weight=config.final_score_weight,
            pred_ce_weight=config.pred_ce_weight,
            pred_l1_weight=config.pred_l1_weight,
            pred_area_weight=config.pred_area_weight,
            prev_weight=config.prev_weight,
            agent_class_weight=config.agent_class_weight,
            agent_box_weight=config.agent_box_weight,
            bev_semantic_weight=config.bev_semantic_weight,
        )

        self.train_metric_cache_paths = None
        self.test_metric_cache_paths = None
        training_metric_cache_root = Path("/mnt/workspace/nanyi/navsim_workspace/exp/metric_cache_training_new")
        metadata_dir = training_metric_cache_root / "metadata"
        if metadata_dir.exists():
            metric_cache = MetricCacheLoader(training_metric_cache_root)
            self.train_metric_cache_paths = metric_cache.metric_cache_paths
            self.test_metric_cache_paths = metric_cache.metric_cache_paths

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if self._checkpoint_path:
            if torch.cuda.is_available():
                state_dict = torch.load(self._checkpoint_path)["state_dict"]
            else:
                state_dict = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
            self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mom_model.to(device)

    def get_sensor_config(self) -> SensorConfig:
        """Return sensor config for 4 cameras (f0, b0, l0, r0) across all frames."""
        return SensorConfig(
            cam_f0=self._config.cam_f0,
            cam_b0=self._config.cam_b0,
            cam_l0=self._config.cam_l0,
            cam_l1=self._config.cam_l1,
            cam_l2=self._config.cam_l2,
            cam_r0=self._config.cam_r0,
            cam_r1=self._config.cam_r1,
            cam_r2=self._config.cam_r2,
            lidar_pc=self._config.lidar_pc,
        )

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [MoMFeatureBuilder(config=self._config)]

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [MoMTargetBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        model_device = next(self._mom_model.parameters()).device
        for key, value in features.items():
            if isinstance(value, torch.Tensor):
                features[key] = value.to(model_device)
        return self._mom_model(features)

    def compute_score(self, targets: Dict[str, torch.Tensor], proposals: torch.Tensor, test: bool = False):
        metric_cache_paths = self.train_metric_cache_paths if self.training else self.test_metric_cache_paths
        if metric_cache_paths is None:
            raise RuntimeError("Metric cache paths not initialized. Set NAVSIM_EXP_ROOT.")

        proposals = proposals.detach()
        tokens = targets["token"]
        if isinstance(tokens, str):
            tokens = [tokens]

        data_points = []
        for token, poses in zip(tokens, proposals.cpu().numpy()):
            token_str = token
            if not isinstance(token_str, str):
                if isinstance(token_str, torch.Tensor):
                    token_str = token_str.item() if token_str.numel() == 1 else token_str
                token_str = str(token_str)
            data_points.append({
                "token": metric_cache_paths[token_str],
                "poses": poses,
                "test": test,
            })

        all_res = get_scores(data_points)

        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)
        final_scores = target_scores[:, :, -1]
        best_scores = torch.amax(final_scores, dim=-1)

        key_agent_corners = torch.FloatTensor(np.stack([res[1] for res in all_res])).to(proposals.device)
        key_agent_labels = torch.BoolTensor(np.stack([res[2] for res in all_res])).to(proposals.device)
        all_ego_areas = torch.BoolTensor(np.stack([res[3] for res in all_res])).to(proposals.device)

        return final_scores, best_scores, target_scores, key_agent_corners, key_agent_labels, all_ego_areas

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> tuple:
        loss_dict = self._loss(targets, predictions, self._config, self.compute_score)
        device = loss_dict["loss"].device

        def _to_tensor(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            return torch.tensor(float(value), device=device)

        zero = torch.tensor(0.0, device=device)
        return (
            _to_tensor(loss_dict["loss"]),
            _to_tensor(loss_dict.get("trajectory_loss", zero)),
            _to_tensor(loss_dict.get("final_score_loss", zero)),
            zero,  # agent_class_loss
            zero,  # agent_box_loss
            zero,  # bev_semantic_loss
            _to_tensor(loss_dict.get("pred_ce_loss", zero)),
            _to_tensor(loss_dict.get("pred_l1_loss", zero)),
            _to_tensor(loss_dict.get("pred_area_loss", zero)),
            _to_tensor(loss_dict.get("best_score", zero)),
            _to_tensor(loss_dict.get("score", zero)),
            zero,  # best_gt_pdm
            zero,  # average_gt_pdm
            zero,  # pred_gt_pdm
        )

    def get_optimizers(self):
        """Returns optimizer for training."""
        lr = self._lr if self._lr is not None else 1e-4
        optimizer = torch.optim.AdamW(self._mom_model.parameters(), lr=lr)
        return [optimizer]
