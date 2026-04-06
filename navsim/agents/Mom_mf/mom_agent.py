import os
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.Mom_mf.mom_config import MomConfig
from navsim.agents.Mom_mf.mom_fla_diffusion_model import MomModel
from navsim.agents.rwkv7_mf.rwkv_callback import RWKVCallback
from navsim.agents.rwkv7_mf.rwkv_loss import RWKV_loss
from navsim.agents.rwkv7_mf.rwkv_features import RWKVFeatureBuilder, RWKVTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.agents.score_module.compute_navsim_score import get_scores
from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
from nuplan.planning.utils.multithreading.worker_utils import worker_map


class MomAgent(AbstractAgent):
    """Agent interface for MoM baseline."""

    def __init__(
        self,
        config: MomConfig,
        lr: float,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initializes MoM agent.
        :param config: global config of MoM agent
        :param lr: learning rate during training
        :param checkpoint_path: optional path string to checkpoint, defaults to None
        """
        super().__init__()

        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self._mom_model = MomModel(config)

        metric_cache = MetricCacheLoader(Path(os.getenv("NAVSIM_EXP_ROOT") + "/metric_cache_training_new"))
        self.train_metric_cache_paths = metric_cache.metric_cache_paths
        self.test_metric_cache_paths = metric_cache.metric_cache_paths
        self.get_scores = get_scores

        if config.use_ray:
            self.worker = RayDistributedNoTorch(threads_per_node=16)
            self.worker_map = worker_map

        if config.use_gt_eval:
            self.requires_scene = True

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        if torch.cuda.is_available():
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path)["state_dict"]
        else:
            state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))[
                "state_dict"
            ]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()})
        self._mom_model.to(device="cuda")

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[i for i in range(self._config.cam_seq_len)])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [RWKVTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [RWKVFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        for key, value in features.items():
            features[key] = value.cuda()

        if targets is not None:
            for key, value in targets.items():
                if isinstance(value, torch.Tensor):
                    targets[key] = value.cuda()

        return self._mom_model(features, targets)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        if isinstance(predictions["denoised_trajectories"], list):
            target_scores_list = []
            for pred in predictions["denoised_trajectories"]:
                final_scores, best_scores, target_scores, gt_states, gt_valid, gt_ego_areas = self.compute_score(
                    targets, pred
                )
                target_scores_list.append(target_scores)
            return RWKV_loss(
                targets,
                predictions,
                target_scores_list,
                final_scores,
                best_scores,
                gt_states,
                gt_valid,
                gt_ego_areas,
                self._config,
            )

        final_scores, best_scores, target_scores, gt_states, gt_valid, gt_ego_areas = self.compute_score(
            targets, predictions["denoised_trajectories"]
        )
        best_traj = predictions["trajectory"].unsqueeze(1)
        pred_final_pdms_scores = self.compute_pdms_best_traj(targets, best_traj)

        return RWKV_loss(
            targets,
            predictions,
            pred_final_pdms_scores,
            target_scores,
            final_scores,
            best_scores,
            gt_states,
            gt_valid,
            gt_ego_areas,
            self._config,
        )

    def compute_score(self, targets, proposals):
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
        else:
            metric_cache_paths = self.test_metric_cache_paths

        proposals = proposals.detach()
        data_points = [
            {
                "token": metric_cache_paths[token],
                "poses": poses,
                "test": False,
            }
            for token, poses in zip(targets["token"], proposals.cpu().numpy())
        ]

        if self._config.use_ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)
        final_scores = target_scores[:, :, -1]
        best_scores = torch.amax(final_scores, dim=-1)
        key_agent_corners = torch.FloatTensor(np.stack([res[1] for res in all_res])).to(proposals.device)
        key_agent_labels = torch.BoolTensor(np.stack([res[2] for res in all_res])).to(proposals.device)
        all_ego_areas = torch.BoolTensor(np.stack([res[3] for res in all_res])).to(proposals.device)

        return final_scores, best_scores, target_scores, key_agent_corners, key_agent_labels, all_ego_areas

    def compute_pdms_best_traj(self, targets, proposals):
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
        else:
            metric_cache_paths = self.test_metric_cache_paths

        proposals = proposals.detach()
        data_points = [
            {
                "token": metric_cache_paths[token],
                "poses": poses,
                "test": False,
            }
            for token, poses in zip(targets["token"], proposals.cpu().numpy())
        ]

        if self._config.use_ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        else:
            all_res = self.get_scores(data_points)

        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)
        final_scores = target_scores[:, :, -1]

        return final_scores

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        optimizer = torch.optim.Adam(self._mom_model.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/score_epoch",
            },
        }

    def get_training_callbacks(self) -> List[pl.Callback]:
        """Inherited, see superclass."""
        best_model_checkpoint = ModelCheckpoint(
            monitor="val/score_epoch",
            mode="max",
            save_top_k=3,
            filename="best_{epoch}_{val/score_epoch:.4f}",
            save_weights_only=False,
        )
        every_5_epoch_checkpoint = ModelCheckpoint(
            save_top_k=-1,
            every_n_epochs=1,
            filename="{epoch}_{step}",
            save_weights_only=False,
        )

        return [RWKVCallback(self._config), best_model_checkpoint, every_5_epoch_checkpoint]


if __name__ == "__main__":
    config = MomConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MomModel(config).to(device)
    model.eval()

    batch_size = 1
    seq_len = config.cam_seq_len
    camera_feature = torch.rand(
        batch_size,
        seq_len,
        3,
        config.camera_height,
        config.camera_width,
        device=device,
    )
    lidar_feature = torch.rand(
        batch_size,
        seq_len,
        1,
        config.lidar_resolution_height,
        config.lidar_resolution_width,
        device=device,
    )
    status_feature = torch.rand(batch_size, 8, device=device)

    features = {
        "camera_feature": camera_feature,
        "lidar_feature": lidar_feature,
        "status_feature": status_feature,
    }

    with torch.no_grad():
        output = model(features, targets=None)

    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {tuple(value.shape)}")
        elif isinstance(value, list):
            print(f"{key}: list({len(value)})")
        else:
            print(f"{key}: {type(value)}")
