# MoM Driver — 模型架构与工作流程说明

## 概述

MoM Driver（Mixture-of-Memories Driver）是一个纯视觉（camera-only）自动驾驶规划模型，核心设计是将 **DINOv2+LoRA** 图像编码器与 **MoM（Mixture-of-Memories）线性注意力块** 相结合，实现多帧时序融合和轨迹解码，最终输出最优驾驶轨迹。

代码位置：`navsim/agents/MoM_driver/`

---

## 整体流水线

```
相机图像（4路 × 10帧）
        │
        ▼
[1] DINOv2 + LoRA 图像编码（per-frame, per-camera）
        │  [B×seq_len, N×16, 256]
        ▼
[2] MoM 时序融合（MomBlock × 2）
        │  scene_features [B, 640, 256]   (10帧 × 64 tokens/帧)
        ▼
[3] Ego token + 可学习轨迹 proposal 初始化
        │  traj_tokens [B, 64, 256]
        ▼
[4] 迭代精化：MoMTransformerDecoder（× 4 层）
        │  proposal_list[4]，每个 [B, 64, 8, 3]
        ▼
[5] MoMTransformerDecoderScorer（× 4 层）
        │  tr_out [B, 64, 256]
        ▼
[6] Scorer 头部 → 每个 proposal 的 PDM 子分数
        │
        ▼
argmax(PDM score) → 最优轨迹 [B, 8, 3]
```

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `mom_config.py` | 所有超参数的数据类配置 |
| `mom_features.py` | 输入特征构建（图像预处理、ego 状态） |
| `mom_backbone.py` | DINOv2+LoRA 编码器 + MoM 时序融合主干网络 |
| `transformer_decoder.py` | MoM 双自注意力解码块（轨迹解码 + 评分） |
| `mom_model.py` | 完整模型，组合所有模块 |
| `score_module/scorer.py` | PDM 子分数预测头部 |
| `mom_agent.py` | 训练/推理接口，继承 `AbstractAgent` |
| `layers/losses/mom_loss.py` | 训练损失函数 |

---

## 各模块详解

### 1. 输入特征构建（`mom_features.py`）

**`MoMFeatureBuilder`** 负责将原始传感器数据转换为模型输入：

- **图像**：4 路相机（`cam_f0, cam_b0, cam_l0, cam_r0`），最多 10 帧，PIL resize 到 `(574×336)`，ImageNet 均值归一化，左填充到 `cam_seq_len=10`
- **输出张量**：`image [seq_len, 4, 3, H, W]`
- **Ego 状态**：11 维向量 = pose(3) + velocity(3) + acceleration(3) + driving_command(2)，同样左填充
- **valid_frame_len**：标量，记录未填充的有效帧数

**`MoMTargetBuilder`** 负责构建训练目标：
- 从 `Scene` 提取未来轨迹（8 个 pose，4 秒，间隔 0.5 秒）
- 支持 `long_trajectory_additional_poses` 的 CubicSpline 插值扩展

---

### 2. 图像编码器 — DINOv2 + LoRA（`mom_backbone.py: ImgEncoder`）

**基础模型**：`timm/vit_small_patch14_reg4_dinov2.lvd142m`（ViT-S/14，预训练权重冻结）

**LoRA 适配**（`LoRA_ViT_timm` / `_LoRA_qkv_timm`）：
- 对每个注意力层的 Q、V 投影添加 rank-32 低秩适配器（K 保持 Identity）
- 原始 ViT 权重全部冻结，只训练 LoRA 参数
- 初始化：A 矩阵 Kaiming uniform，B 矩阵全零（训练初始时 LoRA 输出为零）

**场景 Token 机制**：
- 每路相机配备 `num_scene_tokens=16` 个可学习前缀 token，shape `[1, num_cams, 16, vit_features]`
- 这些 token 被前置到 ViT 输入序列中，ViT 运行后取前 16 个输出 token 作为该相机的场景表示
- 通过 neck `Linear(vit_features → 256)` 投影到模型维度

**GridMask 增强**（训练期间 prob=0.7）：随机在图像上施加格栅遮挡，提升模型鲁棒性

**输出**：`[B, N×16, 256]` = `[B, 64, 256]`（每帧 4 路相机 × 16 tokens）

---

### 3. MoM 时序融合主干网络（`mom_backbone.py: MoMBackbone`）

将逐帧编码结果在时间轴上融合：

```
[B×seq_len, 64, 256]
    → reshape 到 [B, seq_len, 64, 256]
    → 加 per-frame 位置编码 pos_emb [1, 64, 256]
    → reshape 到 [B, 640, 256]  （展平时序）
    → 构建 attention_mask（左填充帧置 0）
    → 2 × MomBlock（fla 库，gated_deltanet 后端）
    → LayerNorm
    → scene_features [B, 640, 256]
```

**MomBlock 配置**（`MoMBlockConfig`）：

| 参数 | 值 |
|------|----|
| hidden_size | 256 |
| num_heads | 4 |
| head_dim | 64 |
| num_memories | 4 |
| topk | 2 |
| mom_backend | gated_deltanet |
| attn_mode | chunk |
| conv_size | 4 |

**Attention Mask 构建**：根据 `valid_frame_len` 计算每个样本的有效 token 起始位置，左填充帧对应 token 置为 0，防止无效帧影响时序融合。

---

### 4. 轨迹初始化（`mom_model.py`）

```python
# Ego 编码
ego_token = Linear(11, 256)(ego_status[:, -1])[:, None]   # [B, 1, 256]

# 可学习 proposal 嵌入
init_feature = Embedding(64, 256)

# 轨迹 tokens
traj_tokens = ego_token + init_feature.weight[None]        # [B, 64, 256]

# 初始 proposals（第 0 个轨迹头）
proposals_0 = traj_head[0](traj_tokens).reshape(B, 64, 8, 3)
```

64 个 proposal，每个预测 8 个未来 pose（x, y, θ），覆盖 4 秒规划时域。

---

### 5. MoM 解码块（`transformer_decoder.py`）

#### `MoMDoubleSelfAttnBlock`

每个解码块内部分三步：

**Step 1：Query 自注意力**
```
query [B, 64, 256] → MomBlock → 更新后的 query
```
query tokens 之间可以自由交互。

**Step 2：无效帧遮盖**
根据 `frames`（valid_frame_len），将 keyval 中左填充帧的 token 置零。

**Step 3：Cross-Attention（两种模式）**

```
concat([keyval, query]) → MomBlock → 取最后 query_len 个 token
```

| 模式 | `use_state_frozen` | 行为 |
|------|--------------------|------|
| **LICA baseline** | `False` | 直接 concat+MomBlock，query token 会写入状态，导致 ε_decay 和 ε_contam 偏差 |
| **State-Frozen（论文创新点）** | `True` | 在所有 query 位置设 beta=0（不写入）、g=0（无衰减），每个 query 独立读取冻结的场景状态，等价于标准 Transformer cross-attention，无额外参数 |

> **核心理论**（论文 Theorem 1 & 2）：LICA 模式存在两种偏差：
> - **ε_decay**：场景状态被 query 位置的衰减门控衰减
> - **ε_contam**：前序 query token 写入状态，导致后续 query 读到被污染的状态
>
> State-Frozen 通过 `frozen_from=kv_len` 参数，在不增加任何参数的前提下同时消除这两种偏差。

**训练期 padding**：当序列长度 ≤ 65 时，在末尾补零到 66，避免 `MomBlock` 内部 `fused_recurrent` 的 chunk mode 断言错误。

---

### 6. 迭代轨迹精化

```python
# 4 轮迭代精化
for i in range(4):
    traj_tokens = decoder_layer[i](traj_tokens, scene_features)
    proposals = traj_head[i+1](traj_tokens).reshape(B, 64, 8, 3)
    proposal_list.append(proposals)
```

- 共 5 个轨迹头（初始 + 4 轮精化），每个为 `MLP(256, 1024, 24)`（24 = 8×3）
- 所有中间 proposals 均保存，用于训练时的辅助 loss

---

### 7. 评分模块（`scorer.py`，`mom_model.py`）

**Proposal 位置编码**：
```python
pos_embed = MLP(24, 1024, 256)(proposals.reshape(B, 64, 24))   # [B, 64, 256]
```

**Scorer Cross-Attention**：
```python
tr_out = MoMTransformerDecoderScorer(pos_embed, scene_features)  # [B, 64, 256]
tr_out = tr_out + ego_token
```

**6 个独立评分头**（每个为 `Linear → ReLU → Linear → scalar`）：

| 分数 | 权重（默认） |
|------|-------------|
| no_at_fault_collisions (NOC) | 1.0 |
| drivable_area_compliance (DAC) | 1.0 |
| driving_direction_compliance (DDC) | 0.0 |
| time_to_collision_within_bound (TTC) | 5.0 |
| ego_progress (EP) | 5.0 |
| comfort | 2.0 |

**可选辅助头**（训练时）：
- `double_score`：额外一个 MLP 同时预测所有 6 个分数
- `agent_pred`：预测周围智能体包围盒（用于碰撞感知训练）
- `area_pred`：预测自车行驶区域合规性

---

### 8. PDM 分数计算与轨迹选择

```python
pdm_score = (
    1.0 * log(sigmoid(noc))
  + 1.0 * log(sigmoid(dac))
  + 0.0 * log(sigmoid(ddc))
  + log(5.0 * sigmoid(ttc) + 5.0 * sigmoid(ep) + 2.0 * sigmoid(comfort))
)

best_idx = argmax(pdm_score, dim=1)          # [B]
trajectory = proposals[batch_idx, best_idx]  # [B, 8, 3]
```

从 64 个候选 proposal 中选出 PDM 综合分最高的一条作为最终输出轨迹。

---

## 训练流程（`mom_agent.py`）

### 接口

`MoMAgent` 继承自 `AbstractAgent`，实现以下接口：

| 方法 | 说明 |
|------|------|
| `get_sensor_config()` | 返回 4 路相机配置 |
| `get_feature_builders()` | 返回 `MoMFeatureBuilder` |
| `get_target_builders()` | 返回 `MoMTargetBuilder` |
| `forward()` | 推理，返回轨迹及评分 |
| `compute_loss()` | 计算训练损失 |
| `get_optimizers()` | 返回 AdamW 优化器 |

### 损失函数（`MoMDrivoRLoss`）

| 损失项 | 默认权重 | 说明 |
|--------|----------|------|
| `trajectory_loss` | 1.0 | L1 轨迹回归损失 |
| `final_score_loss` | 1.0 | 预测 PDM 分数与 GT 的 BCE 损失 |
| `pred_ce_loss` | 1.0 | 智能体预测交叉熵（若启用） |
| `pred_l1_loss` | 0.1 | 智能体包围盒 L1（若启用） |
| `pred_area_loss` | 2.0 | 行驶区域预测损失（若启用） |

GT PDM 分数通过 `MetricCacheLoader` 离线缓存的评估结果实时读取，无需在线仿真。

### 优化器

```python
AdamW(mom_model.parameters(), lr=1e-4)
```

---

## 关键配置参数（`mom_config.py`）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cam_seq_len` | 10 | 输入帧数 |
| `num_cams` | 4 | 相机数量 |
| `image_size` | [574, 336] | 图像分辨率 (W, H) |
| `image_backbone_lora_rank` | 32 | LoRA 秩 |
| `num_scene_tokens` | 16 | 每路相机的场景 token 数 |
| `tf_d_model` | 256 | 模型特征维度 |
| `tf_d_ffn` | 1024 | FFN 中间维度 |
| `mom_temporal_num_layers` | 2 | 时序融合 MomBlock 数量 |
| `proposal_num` | 64 | 候选轨迹数量 |
| `num_poses` | 8 | 每条轨迹的 pose 数 |
| `ref_num` | 4 | 轨迹解码精化轮数 |
| `scorer_ref_num` | 4 | 评分解码精化轮数 |
| `use_state_frozen` | False | 是否启用 State-Frozen cross-attention |

---

## 架构创新点总结

1. **MoM 替代 Transformer**：用 `gated_deltanet` 线性注意力（MomBlock）替代标准 Transformer 的 self-attention 和 cross-attention，降低长序列计算复杂度（O(L) vs O(L²)）。

2. **State-Frozen Cross-Attention**：在 MoM 线性注意力的 cross-attention 中，通过冻结 query 位置的写入门（beta=0）和衰减门（g=0），消除 LICA 模式的 ε_decay 和 ε_contam 偏差，在不增加参数的前提下等价于标准 Transformer cross-attention 的语义。

3. **DINOv2+LoRA**：冻结强大的视觉基础模型，仅通过少量 LoRA 参数适配到驾驶场景，兼顾预训练表示能力和参数效率。

4. **端到端 PDM 评分**：轨迹生成与安全/舒适性评分在同一网络中联合学习，由预缓存的 PDM 指标提供训练信号，避免在线仿真开销。
