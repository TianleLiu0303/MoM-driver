"""
Feature builder for MoM camera-only model.

DrivoR-aligned: 4 cameras (f0, b0, l0, r0) stacked into a single
"image" tensor per frame. Multi-frame with left-padding via pad_sequence.
No LiDAR.

Ego status: [pose(3) + velocity(3) + acceleration(3) + driving_command(2)] = 11 dims.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from scipy.interpolate import CubicSpline

from navsim.common.dataclasses import AgentInput, Scene
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from navsim.planning.training.dataset import pad_status


# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Camera order (DrivoR-aligned): front, back, left, right
CAMERA_ORDER = ["cam_f0", "cam_b0", "cam_l0", "cam_r0"]


class MoMFeatureBuilder(AbstractFeatureBuilder):
    """Input feature builder for MoM camera-only model."""

    def __init__(self, config):
        """
        :param config: MoMConfig dataclass
        """
        self._config = config

    def get_unique_name(self) -> str:
        return "mom_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """
        Build features: 4 cameras stacked into a single "image" tensor per frame.

        Output keys:
          - image: [seq_len, N, 3, H, W]  (left-padded to cam_seq_len; N=4 cameras)
          - ego_status: [seq_len, 11]  (left-padded)
          - valid_frame_len: scalar int
        """
        features = {}
        actual_seq_len = len(agent_input.cameras)
        target_size = tuple(self._config.image_size)  # (W, H) = (1148, 672)

        # Build per-frame multi-camera image tensor
        frame_images = []
        for i in range(actual_seq_len):
            cameras = agent_input.cameras[i]
            cam_images = []
            for cam_name in CAMERA_ORDER:
                cam = getattr(cameras, cam_name)
                cam_images.append(self._process_camera_image(cam.image, target_size))
            # Stack cameras for this frame: [N, 3, H, W]
            frame_images.append(torch.stack(cam_images, dim=0))

        # Stack all frames: [T, N, 3, H, W]
        image_tensor = torch.stack(frame_images, dim=0)

        # Left-pad to cam_seq_len
        image_tensor = self._pad_image_sequence(image_tensor, self._config.cam_seq_len)
        features["image"] = image_tensor

        # Valid frame count (before padding)
        features["valid_frame_len"] = torch.tensor(actual_seq_len, dtype=torch.int32)

        # Ego status: [pose(3) + velocity(3) + acceleration(3) + driving_command(2)] = 11
        status_feature = []
        for ego_status in agent_input.ego_statuses[:actual_seq_len]:
            if ego_status is None:
                continue
            pose = torch.tensor(ego_status.ego_pose, dtype=torch.float32)
            velocity = torch.tensor(ego_status.ego_velocity, dtype=torch.float32)
            acceleration = torch.tensor(ego_status.ego_acceleration, dtype=torch.float32)
            driving_command = torch.tensor(ego_status.driving_command, dtype=torch.float32)
            ego_feature = torch.cat([pose, velocity, acceleration, driving_command], dim=-1)
            status_feature.append(ego_feature)

        if len(status_feature) == 0:
            status_feature = torch.zeros((1, 11), dtype=torch.float32)
        else:
            status_feature = torch.stack(status_feature)

        features["ego_status"], _ = pad_status(status_feature, self._config.cam_seq_len)

        return features

    @staticmethod
    def _pad_image_sequence(tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        """Left-pad image sequence tensor to target_len.

        :param tensor: [T, N, C, H, W]
        :param target_len: desired sequence length
        :return: [target_len, N, C, H, W] (left-padded with zeros)
        """
        T = tensor.shape[0]
        if T >= target_len:
            return tensor[-target_len:]
        pad_len = target_len - T
        pad_shape = (pad_len,) + tensor.shape[1:]
        padding = torch.zeros(pad_shape, dtype=tensor.dtype)
        return torch.cat([padding, tensor], dim=0)

    def _process_camera_image(self, image: Optional[np.ndarray], target_size) -> torch.Tensor:
        """
        Process a single camera image using DrivoR-style PIL resize + ImageNet normalization.

        :param image: raw camera image as numpy array (H, W, 3) or None
        :param target_size: (width, height) tuple
        :return: normalized tensor [3, H, W]
        """
        if image is None:
            h, w = target_size[1], target_size[0]
            return torch.zeros((3, h, w), dtype=torch.float32)

        # PIL resize (DrivoR style)
        im = Image.fromarray(image)
        im = im.resize(target_size)  # target_size = (W, H)

        # Normalize with ImageNet stats
        im = np.asarray(im, dtype=np.float32) / 255.0
        im = (im - IMAGENET_MEAN) / IMAGENET_STD

        # HWC -> CHW
        im = torch.from_numpy(im).permute(2, 0, 1)
        return im


class MoMTargetBuilder(AbstractTargetBuilder):
    """Output target builder for MoM."""

    def __init__(self, config):
        self._config = config

    def get_unique_name(self) -> str:
        return "mom_target"

    def compute_targets(self, scene: Scene) -> Dict[str, Any]:
        trajectory = torch.tensor(
            scene.get_future_trajectory(num_trajectory_frames=self._config.trajectory_sampling.num_poses).poses
        )

        if self._config.long_trajectory_additional_poses > 0:
            try:
                trajectory_long = scene.get_future_trajectory(
                    num_trajectory_frames=self._config.trajectory_sampling.num_poses
                    + self._config.long_trajectory_additional_poses
                ).poses
                x = np.arange(trajectory_long.shape[0], dtype=np.float32)
                alpha = (
                    2
                    * self._config.long_trajectory_additional_poses
                    / (self._config.trajectory_sampling.num_poses * (self._config.trajectory_sampling.num_poses + 1))
                )
                x_new = np.arange(trajectory.shape[0], dtype=np.float32)
                off_sets = np.cumsum((x_new + 1) * alpha)
                x_new += off_sets
                traj_ = []
                for i in range(3):
                    y = trajectory_long[:, i]
                    cs = CubicSpline(x, y)
                    traj_.append(cs(x_new))
                trajectory_long = np.stack(traj_, axis=1)

                trajectory_long = torch.tensor(trajectory_long)
                return {
                    "trajectory": trajectory,
                    "trajectory_long": trajectory_long,
                    "token": scene.scene_metadata.initial_token,
                }
            except Exception:
                return {
                    "trajectory": trajectory,
                    "trajectory_long": trajectory,
                    "token": scene.scene_metadata.initial_token,
                }

        return {
            "trajectory": trajectory,
            "token": scene.scene_metadata.initial_token,
        }
