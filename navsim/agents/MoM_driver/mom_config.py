from dataclasses import dataclass, field
from typing import List, Optional

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling


@dataclass
class MoMBlockConfig:
    """Config for a single MoM block group (temporal fusion or decoder).

    Field names match fla's MomConfig exactly, so we can construct
    FlaMomConfig(**vars(block_config), num_hidden_layers=...) directly.
    """

    hidden_size: int = 256
    num_heads: int = 4
    head_dim: int = 64
    expand_v: float = 1.0
    use_output_gate: bool = True
    use_short_conv: bool = True
    conv_size: int = 4
    hidden_ratio: Optional[int] = 4
    hidden_act: str = "swish"
    norm_eps: float = 1e-6
    num_memories: int = 4
    topk: int = 2
    capacity: float = 1.0
    shared_mem: bool = True
    single_kv_proj: bool = False
    mom_backend: str = "gated_deltanet"
    attn_mode: str = "chunk"
    fuse_norm: bool = True
    fuse_swiglu: bool = True


@dataclass
class MoMConfig:
    """Config for MoM camera-only multi-frame model with DINOv2+LoRA + MoM fusion."""

    # ----------------------
    # General + sequence
    # ----------------------
    trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5)
    cam_seq_len: int = 10

    # ----------------------
    # Sensor config: 4 cameras (f0, b0, l0, r0) across 10 frames
    # ----------------------
    cam_f0: List[int] = field(default_factory=lambda: list(range(10)))
    cam_b0: List[int] = field(default_factory=lambda: list(range(10)))
    cam_l0: List[int] = field(default_factory=lambda: list(range(10)))
    cam_r0: List[int] = field(default_factory=lambda: list(range(10)))
    cam_l1: List[int] = field(default_factory=list)
    cam_l2: List[int] = field(default_factory=list)
    cam_r1: List[int] = field(default_factory=list)
    cam_r2: List[int] = field(default_factory=list)
    lidar_pc: List[int] = field(default_factory=list)

    # Number of cameras
    num_cams: int = 4

    # ----------------------
    # DINOv2+LoRA image backbone params
    # ----------------------
    image_size: List[int] = field(default_factory=lambda: [574, 336])
    image_backbone_model_name: str = "timm/vit_small_patch14_reg4_dinov2.lvd142m"
    image_backbone_model_weights: str = "weights/vit_small_patch14_reg4_dinov2.lvd142m/model.safetensors"
    image_backbone_use_lora: bool = True
    image_backbone_finetune: bool = False
    image_backbone_lora_rank: int = 32
    image_backbone_focus_front_cam: bool = False
    image_backbone_use_feature_pooling: bool = False
    image_backbone_compress_fc: bool = False

    # ----------------------
    # Scene tokens + feature dims
    # ----------------------
    num_scene_tokens: int = 16
    tf_d_model: int = 256
    tf_d_ffn: int = 1024

    # ----------------------
    # MoM temporal fusion config
    # ----------------------
    mom_temporal_num_layers: int = 2
    mom_temporal: MoMBlockConfig = field(default_factory=MoMBlockConfig)

    # ----------------------
    # MoM decoder config (for trajectory and scorer decoders)
    # ----------------------
    mom_decoder: MoMBlockConfig = field(default_factory=MoMBlockConfig)

    # ----------------------
    # State-Frozen Cross-Attention (Innovation Point 2)
    # ----------------------
    # False  →  LICA baseline: concat([keyval, query]) → MomBlock, no masking.
    #           Introduces ε_decay and ε_contam biases (Theorem 1).
    # True   →  State-Frozen: beta=0, g=0 at all query positions in cross-attn.
    #           Eliminates both biases; each query independently reads the
    #           frozen scene state (Theorem 2). No extra parameters required.
    use_state_frozen: bool = False
    # Fine-grained ablations for Theorem 2.
    # freeze_beta_only=True: zero beta only at query positions, preserving g.
    # freeze_g_only=True: zero g only at query positions, preserving beta.
    # These switches are intended for ablations and should remain False during
    # standard training. If use_state_frozen=True, it should override both and
    # freeze beta and g together.
    freeze_beta_only: bool = False
    freeze_g_only: bool = False

    # ----------------------
    # Recurrent single-frame inference cache
    # ----------------------
    # False (default / training): standard chunk mode, no state carried between calls.
    # True  (online / infinite-frame eval): each forward pass inputs one frame's
    #   query+keyval; recurrent states are stored inside MoMTransformerDecoder and
    #   MoMTransformerDecoderScorer and chained across calls.  Call reset_cache()
    #   at the start of each new scene.  Has no effect during training.
    use_past_key_cache: bool = False

    # ----------------------
    # Init weights
    # ----------------------
    gpt_linear_layer_init_mean: float = 0.0
    gpt_linear_layer_init_std: float = 0.02
    gpt_layer_norm_init_weight: float = 1.0
    embd_pdrop: float = 0.1

    # ----------------------
    # Trajectory + decoder params (DrivoR-style)
    # ----------------------
    b2d: bool = False
    shared_refiner: bool = False
    ref_num: int = 4
    scorer_ref_num: int = 4
    proposal_num: int = 64
    num_poses: int = 8
    one_token_per_traj: bool = True
    full_history_status: bool = False
    long_trajectory_additional_poses: int = -1

    refiner_num_heads: int = 1
    refiner_ls_values: float = 0.0

    # ----------------------
    # Scorer params (DrivoR-style)
    # ----------------------
    double_score: bool = False
    agent_pred: bool = False
    area_pred: bool = False
    bev_map: bool = False
    bev_agent: bool = False
    num_bounding_boxes: int = 30

    # ----------------------
    # PDM score weights (DrivoR-style)
    # ----------------------
    noc: float = 1.0
    dac: float = 1.0
    ddc: float = 0.0
    ttc: float = 5.0
    ep: float = 5.0
    comfort: float = 2.0

    # ----------------------
    # Loss weights (DrivoR-style)
    # ----------------------
    trajectory_weight: float = 1.0
    inter_weight: float = 0.0
    sub_score_weight: float = 0.0
    final_score_weight: float = 1.0
    pred_ce_weight: float = 1.0
    pred_l1_weight: float = 0.1
    pred_area_weight: float = 2.0
    prev_weight: float = 0.0
    agent_class_weight: float = 1.0
    agent_box_weight: float = 0.1
    bev_semantic_weight: float = 1.0
    div_weight: float = 0.0
