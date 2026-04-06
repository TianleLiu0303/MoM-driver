"""
MoM Model: DINOv2+LoRA + MoM temporal fusion + MoM-based decoders + DrivoR heads.

Pipeline:
  1. DINOv2+LoRA per-camera per-frame → MoM temporal fusion → scene_features [B, 640, 256]
  2. ego_token + init_feature → traj_tokens [B, 64, 256]
  3. MoMTransformerDecoder (iterative refinement) → proposals [B, 64, 8, 3]
  4. MoMTransformerDecoderScorer → scorer → PDM scores → argmax → trajectory
"""

from typing import Any, Dict

import torch
import torch.nn as nn

try:
    from .mom_backbone import MoMBackbone
    from .transformer_decoder import MoMTransformerDecoder, MoMTransformerDecoderScorer
    from .score_module.scorer import Scorer
    from .layers.utils.mlp import MLP
except ImportError:
    from navsim.agents.MoM_driver.mom_backbone import MoMBackbone
    from navsim.agents.MoM_driver.transformer_decoder import MoMTransformerDecoder, MoMTransformerDecoderScorer
    from navsim.agents.MoM_driver.score_module.scorer import Scorer
    from navsim.agents.MoM_driver.layers.utils.mlp import MLP


class MoMModel(nn.Module):
    """Full MoM model with DINOv2+LoRA backbone, MoM temporal fusion,
    MoM-based trajectory decoder and scorer, and DrivoR-style heads."""

    def __init__(self, config):
        super().__init__()
        self._config = config
        self.poses_num = config.num_poses
        self.state_size = 3
        self.embed_dims = config.tf_d_model

        # --- Backbone: DINOv2+LoRA + MoM temporal fusion ---
        self.backbone = MoMBackbone(config)

        # --- Ego status encoding ---
        if config.full_history_status:
            self.hist_encoding = nn.Linear(11 * config.cam_seq_len, config.tf_d_model)
        else:
            self.hist_encoding = nn.Linear(11, config.tf_d_model)
        # --- Trajectory tokens (learnable proposals) ---
        if config.one_token_per_traj:
            self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.poses_num * self.state_size
        else:
            self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.state_size

        # --- MoM-based trajectory decoder ---
        self.trajectory_decoder = MoMTransformerDecoder(config=config)

        # --- MoM-based scorer decoder ---
        self.scorer_attention = MoMTransformerDecoderScorer(
            num_layers=config.scorer_ref_num,
            d_model=config.tf_d_model,
            config=config,
        )

        # --- Position embedding for scorer input (proposal positions → d_model) ---
        self.pos_embed = nn.Sequential(
            nn.Linear(self.poses_num * 3, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )

        # --- Trajectory heads (one per refinement step + initial) ---
        ref_num = config.ref_num
        self.traj_head = nn.ModuleList([
            MLP(config.tf_d_model, config.tf_d_ffn, traj_head_output_size)
            for _ in range(ref_num + 1)
        ])

        # --- Scorer ---
        self.scorer = Scorer(config)

    def forward(
        self,
        features: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        :param features: dict with keys:
            - image: [B, seq_len, N, C, H, W]  (unified multi-cam tensor)
            - ego_status: [B, seq_len, 11]
            - valid_frame_len: [B]
        :return: dict with trajectory, proposals, scores, etc.
        """
        image = features["image"]
        ego_status = features["ego_status"]
        valid_frame_len = features["valid_frame_len"]

        bs = image.shape[0]

        # --- Step 1: Backbone — DINOv2 + MoM temporal fusion ---
        # scene_features: [B, seq_len*tokens_per_frame, d_model]
        scene_features = self.backbone(image, valid_frame_len)

        # --- Step 2: Ego token + trajectory initialization ---
        if self._config.full_history_status:
            ego_input = ego_status.flatten(-2)  # [B, seq_len * 11]
        else:
            ego_input = ego_status[:, -1]  # [B, 11] — last frame only

        ego_token = self.hist_encoding(ego_input)[:, None]  # [B, 1, d_model]
        traj_tokens = ego_token + self.init_feature.weight[None]  # [B, proposal_num, d_model]

        # --- Step 3: Initial proposals ---
        proposals = self.traj_head[0](traj_tokens).reshape(bs, -1, self.poses_num, self.state_size)
        proposal_list = [proposals]

        # --- Step 4: Iterative refinement via MoM trajectory decoder ---
        token_list = self.trajectory_decoder(traj_tokens, scene_features, frames=valid_frame_len)
        for i in range(self._config.ref_num):
            tokens = token_list[i]
            proposals = self.traj_head[i + 1](tokens).reshape(bs, -1, self.poses_num, self.state_size)
            proposal_list.append(proposals)

        traj_tokens = token_list[-1]
        proposals = proposal_list[-1]

        output: Dict[str, Any] = {
            "proposals": proposals,
            "proposal_list": proposal_list,
        }

        # --- Step 5: Scoring ---
        embedded_traj = self.pos_embed(proposals.reshape(bs, proposals.shape[1], -1).detach())
        tr_out = self.scorer_attention(embedded_traj, scene_features, frames=valid_frame_len)
        tr_out = tr_out + ego_token

        (
            pred_logit,
            pred_logit2,
            pred_agents_states,
            pred_area_logit,
            bev_semantic_map,
            agent_states,
            agent_labels,
        ) = self.scorer(proposals, tr_out)

        output["pred_logit"] = pred_logit
        output["pred_logit2"] = pred_logit2
        output["pred_agents_states"] = pred_agents_states
        output["pred_area_logit"] = pred_area_logit
        output["bev_semantic_map"] = bev_semantic_map
        output["agent_states"] = agent_states
        output["agent_labels"] = agent_labels

        # --- Step 6: PDM score computation + trajectory selection ---
        pdm_score = (
            self._config.noc * pred_logit["no_at_fault_collisions"].sigmoid().log()
            + self._config.dac * pred_logit["drivable_area_compliance"].sigmoid().log()
            + self._config.ddc * pred_logit["driving_direction_compliance"].sigmoid().log()
            + (
                self._config.ttc * pred_logit["time_to_collision_within_bound"].sigmoid()
                + self._config.ep * pred_logit["ego_progress"].sigmoid()
                + self._config.comfort * pred_logit["comfort"].sigmoid()
            ).log()
        )

        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[torch.arange(bs, device=proposals.device), token]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score

        return output


if __name__ == "__main__":
    import os

    os.environ["NAVSIM_DEVKIT_ROOT"] = "/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim"

    try:
        from .mom_config import MoMConfig
    except ImportError:
        from navsim.agents.MoM_driver.mom_config import MoMConfig

    cfg = MoMConfig()
    cfg.double_score = True
    cfg.agent_pred = True
    cfg.area_pred = True
    cfg.bev_map = False
    cfg.bev_agent = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seq_len = cfg.cam_seq_len
    num_cams = cfg.num_cams
    H, W = cfg.image_size[1], cfg.image_size[0]  # 672, 1148

    total_features = {
        "image": torch.rand((1, seq_len, num_cams, 3, H, W), device=device),
        "ego_status": torch.rand((1, seq_len, 11), device=device),
        "valid_frame_len": torch.tensor([seq_len], device=device, dtype=torch.int32),
    }

    model = MoMModel(cfg).to(device)
    model.eval()

    with torch.no_grad():
        output = model.forward(total_features)
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        elif isinstance(value, list):
            print(f"{key}: list of {len(value)} items")
            if len(value) > 0 and isinstance(value[0], torch.Tensor):
                print(f"  - {value[0].shape}")
        elif isinstance(value, dict):
            print(f"{key}: dict with {len(value)} keys")
        else:
            print(f"{key}: {type(value)}")
    print("Forward pass successful!")

