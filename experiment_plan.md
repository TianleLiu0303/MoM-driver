# MoM Driver — 消融实验计划

> **文档版本**：v1.1 | **更新日期**：2026-03-30  
> **基线模型**：MoM Driver（DINOv2-S + LoRA-32 + MoM 时序融合 + LICA Decoder）  
> **评估集**：navtest | **主要指标**：EPDMS（Extended PDM Score）

---

## 目录

1. [实验总览](#1-实验总览)
2. [基线配置](#2-基线配置)
3. [消融一：时序长度泛化性（Train × Eval History Length）](#3-消融一时序长度泛化性)
4. [消融二：Memory Slot 数量](#4-消融二memory-slot-数量)
   - 4.1 [Encoder（时序融合）](#41-encoder时序融合)
   - 4.2 [Decoder（轨迹/评分解码）](#42-decoder轨迹评分解码)
5. [消融三：图像主干网络](#5-消融三图像主干网络)
   - 5.1 [微调策略（Finetuning Strategy）](#51-微调策略)
   - 5.2 [主干模型规模（Backbone Scale）](#52-主干模型规模)
6. [消融四：State-Frozen — β 与 g 的解耦](#6-消融四state-frozenβ-与-g-的解耦)
7. [实验执行说明](#7-实验执行说明)
8. [结果汇总模板](#8-结果汇总模板)

---

## 1. 实验总览

| 消融组 | 子实验数 | 核心变量 | 修改位置 |
|--------|----------|----------|----------|
| 时序长度泛化 | 15（3 Train × 5 Eval） | `cam_seq_len` + `num_history_frames` | config + 训练脚本 |
| Memory Slots — Encoder | 4 | `mom_temporal.num_memories` / `topk` | `MoMConfig.mom_temporal` |
| Memory Slots — Decoder | 4 | `mom_decoder.num_memories` / `topk` | `MoMConfig.mom_decoder` |
| 图像主干微调策略 | 6（当前代码可直接支持） | `image_backbone_*` + token compression 方式 | `MoMConfig` |
| 图像主干规模 | 3 | `image_backbone_model_name` | `MoMConfig` |
| State-Frozen 解耦 | 4 | `use_state_frozen` + `freeze_beta_only` + `freeze_g_only` | `MoMConfig` + `transformer_decoder.py` + `flash-linear-attention/fla/layers/mom.py` |
| **合计** | **~36** | — | — |

---

## 2. 基线配置

所有消融实验均以以下配置作为**基线（Baseline）**，记为 **Base**。

### 2.1 关键超参数

```yaml
# navsim/planning/script/config/common/agent/mom_driver_agent.yaml

config:
  cam_seq_len: 10                              # 训练/评估历史帧数
  num_cams: 4                                  # f0, b0, l0, r0
  image_size: [574, 336]

  image_backbone_model_name: timm/vit_small_patch14_reg4_dinov2.lvd142m
  image_backbone_use_lora: true
  image_backbone_finetune: false
  image_backbone_lora_rank: 32
  image_backbone_use_feature_pooling: false
  image_backbone_compress_fc: false
  num_scene_tokens: 16                         # Registers 压缩方式

  tf_d_model: 256
  tf_d_ffn: 1024

  mom_temporal_num_layers: 2
  mom_temporal:
    num_memories: 4
    topk: 2

  mom_decoder:
    num_memories: 4
    topk: 2

  use_state_frozen: false                      # 默认 LICA baseline
  freeze_beta_only: false                      # 仅用于 SF-1
  freeze_g_only: false                         # 仅用于 SF-2
  ref_num: 4
  scorer_ref_num: 4
  proposal_num: 64
  num_poses: 8
```

### 2.2 训练脚本（基线）

```bash
python navsim/planning/script/run_training.py \
  agent=mom_driver_agent \
  experiment_name=base \
  "dataloader.params.batch_size"=1 \
  "dataloader.params.num_workers"=16 \
  "train_test_split.scene_filter.num_history_frames"=10 \
  "trainer.params.strategy"=ddp \
  train_test_split=navtrain \
  cache_path=$NAVSIM_EXP_ROOT/momdriver_cache \
  use_cache_without_dataset=true
```

### 2.3 评估脚本（基线）

```bash
python navsim/planning/script/run_pdm_score_rwkv.py \
  train_test_split=navtest \
  worker=ray_distributed \
  metric_cache_path=$CACHE_PATH \
  agent=mom_driver_agent \
  agent.checkpoint_path=$CHECKPOINT \
  "train_test_split.scene_filter.num_history_frames"=10 \
  enable_padding=true \
  experiment_name=eval_base
```

---

## 3. 消融一：时序长度泛化性

### 3.1 实验目的

验证 MoM 线性注意力的时序外推能力：在较短历史帧（4帧）上训练，测试在更长历史帧（8、12、20帧乃至流式输入）下的性能退化程度；以及训练帧数增大对跨长度泛化的影响。

### 3.2 实验设计

| 实验编号 | Train帧数 | Eval帧数 | 实验名称 |
|----------|-----------|----------|----------|
| T1-1 | 4 | 4 | `ablation_seq_t4_e4` |
| T1-2 | 4 | 8 | `ablation_seq_t4_e8` |
| T1-3 | 4 | 12 | `ablation_seq_t4_e12` |
| T1-4 | 4 | 20 | `ablation_seq_t4_e20` |
| T1-5 | 4 | ∞ (流式) | `ablation_seq_t4_einf` |
| T1-6 | 12 | 4 | `ablation_seq_t12_e4` |
| T1-7 | 12 | 8 | `ablation_seq_t12_e8` |
| T1-8 | 12 | 12 | `ablation_seq_t12_e12` |
| T1-9 | 12 | 20 | `ablation_seq_t12_e20` |
| T1-10 | 12 | ∞ (流式) | `ablation_seq_t12_einf` |
| T1-11 | 20 | 4 | `ablation_seq_t20_e4` |
| T1-12 | 20 | 8 | `ablation_seq_t20_e8` |
| T1-13 | 20 | 12 | `ablation_seq_t20_e12` |
| T1-14 | 20 | 20 | `ablation_seq_t20_e20` |
| T1-15 | 20 | ∞ (流式) | `ablation_seq_t20_einf` |

> **∞（流式）**：评估时不截断历史，使用场景可用的全部历史帧（通常 > 20帧），测试流式推理性能。

### 3.3 参数修改方法

#### 修改 1：`cam_seq_len`（控制训练帧数）

在 `mom_driver_agent.yaml` 中添加覆盖，或直接通过 Hydra CLI：

```bash
# 训练（以 Train=4 为例）
python navsim/planning/script/run_training.py \
  agent=mom_driver_agent \
  experiment_name=ablation_seq_t4_e4 \
  "agent.config.cam_seq_len"=4 \
  "train_test_split.scene_filter.num_history_frames"=4 \
  ...
```

> `cam_seq_len` 同时控制：  
> - `MoMFeatureBuilder` 中图像和 ego_status 的左填充目标长度  
> - `MoMBackbone.seq_len`（影响 `pos_emb` 大小和 `mom_blocks` 输入长度）  
> - `_build_attention_mask` 的 total_tokens 维度

**注意**：`cam_seq_len=4` 对应 `pos_emb` 形状为 `[1, 64, 256]`（帧位置编码），该 embedding 是逐帧广播的，**不受帧数变化影响**，因此 Eval 帧数与 Train 帧数不同时无需重新定义 pos_emb。但需确认 `_build_attention_mask` 在 Eval 帧数超过 Train 帧数时的正确性（详见下方代码确认）。

#### 修改 2：评估帧数覆盖（仅 Eval）

```bash
# 评估（以 Train=4, Eval=12 为例）
python navsim/planning/script/run_pdm_score_rwkv.py \
  agent=mom_driver_agent \
  agent.checkpoint_path=$CKPT_T4 \
  "agent.config.cam_seq_len"=12 \
  "train_test_split.scene_filter.num_history_frames"=12 \
  enable_padding=true \
  experiment_name=ablation_seq_t4_e12
```

#### 修改 3：流式评估（Eval=∞）

流式评估需要在 `MoMBackbone` 中支持动态 `seq_len`。需检查以下代码路径是否支持（`mom_backbone.py:404-483`）：

```python
# mom_backbone.py: forward() 中 seq_len 来自输入 shape，不依赖 config
B, seq_len, num_cams, C, H, W = image.shape   # 动态 seq_len ✓

# 但 pos_emb 形状固定为 [1, tokens_per_frame, d_model]，是按 token slot 广播的
x = x + self.pos_emb.unsqueeze(1)             # broadcast over frames ✓
```

> **结论**：`pos_emb` 是 per-token-slot（不是 per-frame），广播机制天然支持任意帧数。  
> 流式评估时只需在 eval 脚本中**不限制** `num_history_frames`，并设置 `enable_padding=false`（或使用所有可用历史）。

### 3.4 需要新增/确认的代码修改

- [ ] 确认 `_build_attention_mask`（`mom_backbone.py:485-510`）在 `valid_frame_len > cam_seq_len` 时不会越界（流式场景）
- [ ] 为流式评估模式在评估脚本中添加 `enable_streaming=true` 或等效参数
- [ ] 可选：为 `cam_seq_len` 差异（Train ≠ Eval）添加日志警告

---

## 4. 消融二：Memory Slot 数量

#### 实验目的

研究 MoM memory 数量与 `topk` 激活数对规划性能的影响，并在**一张对称表**中同时回答两个问题：

- 时序编码器（Encoder）是否需要更多 memories 来提升容量
- 轨迹/评分解码器（Decoder）是否也需要多槽，还是单槽更优

#### 实验设计

| Encoder Ablation ID | Enc Mem/TopK | Decoder 固定 | 实验名称 | Decoder Ablation ID | Dec Mem/TopK | Encoder 固定 | 实验名称 |
|---------------------|--------------|---------------|----------|---------------------|--------------|---------------|----------|
| MS-1 | 1/1 | 4/2 | `ablation_mem_enc_1k1` | MS-6 | 1/1 | 4/2 | `ablation_mem_dec_1k1` |
| MS-2 | 2/1 | 4/2 | `ablation_mem_enc_2k1` | MS-7 | 2/1 | 4/2 | `ablation_mem_dec_2k1` |
| MS-3 | 2/2 | 4/2 | `ablation_mem_enc_2k2` | MS-8 | 2/2 | 4/2 | `ablation_mem_dec_2k2` |
| MS-0$^\dagger$ | 4/2 | 4/2 | `base` | MS-0$^\dagger$ | 4/2 | 4/2 | `base` |
| MS-4 | 8/2 | 4/2 | `ablation_mem_enc_8k2` | MS-9 | 8/2 | 4/2 | `ablation_mem_dec_8k2` |
| MS-5 | 8/4 | 4/2 | `ablation_mem_enc_8k4` | MS-10 | 8/4 | 4/2 | `ablation_mem_dec_8k4` |

> **约束**：`topk <= num_memories`，否则 FLA 内部会报错。
> **正文表推荐格式**：左半边展示 encoder ablation（decoder 固定 4/2），右半边展示 decoder ablation（encoder 固定 4/2），每个单元仅报告 `EPDMS`。
> **符号说明**：`Mem/TopK` 表示 `num_memories/topk`；`$^\dagger$` 表示默认配置。

#### 推荐实验参数

```bash
# MS-1: 单槽 encoder
"agent.config.mom_temporal.num_memories"=1 \
"agent.config.mom_temporal.topk"=1

# MS-2: 2 槽 encoder, topk=1
"agent.config.mom_temporal.num_memories"=2 \
"agent.config.mom_temporal.topk"=1

# MS-3: 2 槽 encoder, topk=2
"agent.config.mom_temporal.num_memories"=2 \
"agent.config.mom_temporal.topk"=2

# MS-4: 8 槽 encoder, topk=2
"agent.config.mom_temporal.num_memories"=8 \
"agent.config.mom_temporal.topk"=2

# MS-5: 8 槽 encoder, topk=4
"agent.config.mom_temporal.num_memories"=8 \
"agent.config.mom_temporal.topk"=4

# MS-6: 单槽 decoder
"agent.config.mom_decoder.num_memories"=1 \
"agent.config.mom_decoder.topk"=1

# MS-7: 2 槽 decoder, topk=1
"agent.config.mom_decoder.num_memories"=2 \
"agent.config.mom_decoder.topk"=1

# MS-8: 2 槽 decoder, topk=2
"agent.config.mom_decoder.num_memories"=2 \
"agent.config.mom_decoder.topk"=2

# MS-9: 8 槽 decoder, topk=2
"agent.config.mom_decoder.num_memories"=8 \
"agent.config.mom_decoder.topk"=2

# MS-10: 8 槽 decoder, topk=4
"agent.config.mom_decoder.num_memories"=8 \
"agent.config.mom_decoder.topk"=4
```

#### 训练命令示例

```bash
# 以 MS-1 为例：仅修改 encoder，decoder 保持基线 4/2
python navsim/planning/script/run_training.py \
  agent=mom_driver_agent \
  experiment_name=ablation_mem_enc_1k1 \
  "agent.config.mom_temporal.num_memories"=1 \
  "agent.config.mom_temporal.topk"=1 \
  ...

# 以 MS-6 为例：仅修改 decoder，encoder 保持基线 4/2
python navsim/planning/script/run_training.py \
  agent=mom_driver_agent \
  experiment_name=ablation_mem_dec_1k1 \
  "agent.config.mom_decoder.num_memories"=1 \
  "agent.config.mom_decoder.topk"=1 \
  ...
```

#### 论文表格模板（LaTeX）

```latex
\begin{table}
\centering
\footnotesize
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1}
\caption{Ablation on memory slot configuration in MoM-Driver.
         Decoder is fixed at 4/2 when varying encoder (left),
         and encoder is fixed at 4/2 when varying decoder (right).
         $^\dagger$ denotes the default setting.}
\label{tab:memory_ablation}
\begin{tabularx}{\linewidth}{CCC!{\vrule width 0.8pt}CCC}
\toprule
\multicolumn{3}{c!{\vrule width 0.8pt}}{\textbf{Encoder Ablation} (Decoder: 4/2)}
& \multicolumn{3}{c}{\textbf{Decoder Ablation} (Encoder: 4/2)} \\
\cmidrule(lr){1-3} \cmidrule(lr){4-6}
\textbf{ID} & \textbf{Enc Mem/TopK} & \textbf{EPDMS}$\uparrow$
& \textbf{ID} & \textbf{Dec Mem/TopK} & \textbf{EPDMS}$\uparrow$ \\
\midrule
MS-1                   & 1/1          & & MS-6  & 1/1          & \\
MS-2                   & 2/1          & & MS-7  & 2/1          & \\
MS-3                   & 2/2          & & MS-8  & 2/2          & \\
\rowcolor{blue!5}
MS-0$^\dagger$         & \textbf{4/2} & & MS-0$^\dagger$ & \textbf{4/2} & \\
MS-4                   & 8/2          & & MS-9  & 8/2          & \\
MS-5                   & 8/4          & & MS-10 & 8/4          & \\
\bottomrule
\end{tabularx}
\end{table}
```

---

## 5. 消融三：图像主干网络

### 5.1 微调策略（基于当前 `mom_backbone.py` 的可运行版本）

当前 `navsim/agents/MoM_driver/mom_backbone.py` 已经原生支持 3 类 token 压缩路径：

- **Registers**：`num_scene_tokens > 0`，直接截取 learnable prefix scene tokens
- **Pooling**：`image_backbone_use_feature_pooling=True`，对完整 token 序列做 `AdaptiveAvgPool1d`
- **Decoder Compress**：`image_backbone_compress_fc=True`，仅支持 `image_backbone_focus_front_cam=True` 的前视压缩分支

需要特别注意：**No compression (`num_scene_tokens=0`) 目前并不能直接跑通完整 backbone**。原因是
`MoMBackbone.tokens_per_frame = num_cams * num_scene_tokens`，当 `num_scene_tokens=0` 时会导致
`tokens_per_frame=0`，从而在 `scene_embeds`、`pos_emb` 和 reshape 阶段失效。因此本节只保留
当前代码可直接支持的实验；No compression 若后续实现动态 token 数，再单独补充。

#### 实验设计

| 编号 | 压缩方式 | 配置约束 | Cam tokens | Scene tokens | 图像编码训练方式 | 实验名称 |
|------|----------|----------|-----------|--------------|-----------------|----------|
| (a) | Pooling | `use_feature_pooling=True` | 16 | 64 | Frozen | `ablation_bb_pool_frozen` |
| (b) | Pooling | `use_feature_pooling=True` | 16 | 64 | LoRA | `ablation_bb_pool_lora` |
| (c) | Decoder Compress | `compress_fc=True, focus_front_cam=True` | 16(front) + 48(other) | 64 | Frozen | `ablation_bb_dec_frozen` |
| (d) | Decoder Compress | `compress_fc=True, focus_front_cam=True` | 16(front) + 48(other) | 64 | LoRA | `ablation_bb_dec_lora` |
| (e) | Registers | `num_scene_tokens=16` | 16 | 64 | Full finetune | `ablation_bb_reg_fullft` |
| (f) | Registers | `num_scene_tokens=16` | 16 | 64 | Frozen | `ablation_bb_reg_frozen` |
| **(g)** | **Registers** | **num_scene_tokens=16** | **16** | **64** | **LoRA（基线）** | `base` |

> - **Pooling**：先拿完整 ViT 输出，再沿 token 维池化到 `num_scene_tokens`
> - **Decoder Compress**：当前实现只压缩前视相机，其余相机仍取前 `num_scene_tokens` 个 token
> - **Registers**：当前默认做法，使用可学习前缀 token 与图像 token 共同进入 ViT

#### 各实验参数修改

**实验 (a) — Pooling + Frozen**

```bash
python navsim/planning/script/run_training.py \
  agent=mom_driver_agent \
  experiment_name=ablation_bb_pool_frozen \
  "agent.config.num_scene_tokens"=16 \
  "agent.config.image_backbone_use_feature_pooling"=true \
  "agent.config.image_backbone_use_lora"=false \
  "agent.config.image_backbone_finetune"=false \
  ...
```

**实验 (b) — Pooling + LoRA**

```bash
python navsim/planning/script/run_training.py \
  agent=mom_driver_agent \
  experiment_name=ablation_bb_pool_lora \
  "agent.config.num_scene_tokens"=16 \
  "agent.config.image_backbone_use_feature_pooling"=true \
  "agent.config.image_backbone_use_lora"=true \
  "agent.config.image_backbone_lora_rank"=32 \
  ...
```

**实验 (c) — Decoder Compress + Frozen**

```bash
python navsim/planning/script/run_training.py \
  agent=mom_driver_agent \
  experiment_name=ablation_bb_dec_frozen \
  "agent.config.num_scene_tokens"=16 \
  "agent.config.image_backbone_focus_front_cam"=true \
  "agent.config.image_backbone_compress_fc"=true \
  "agent.config.image_backbone_use_lora"=false \
  "agent.config.image_backbone_finetune"=false \
  ...
```

**实验 (d) — Decoder Compress + LoRA**

```bash
python navsim/planning/script/run_training.py \
  agent=mom_driver_agent \
  experiment_name=ablation_bb_dec_lora \
  "agent.config.num_scene_tokens"=16 \
  "agent.config.image_backbone_focus_front_cam"=true \
  "agent.config.image_backbone_compress_fc"=true \
  "agent.config.image_backbone_use_lora"=true \
  ...
```

**实验 (e) — Registers + Full Finetune**

```bash
python navsim/planning/script/run_training.py \
  agent=mom_driver_agent \
  experiment_name=ablation_bb_reg_fullft \
  "agent.config.num_scene_tokens"=16 \
  "agent.config.image_backbone_use_feature_pooling"=false \
  "agent.config.image_backbone_compress_fc"=false \
  "agent.config.image_backbone_focus_front_cam"=false \
  "agent.config.image_backbone_use_lora"=false \
  "agent.config.image_backbone_finetune"=true \
  ...
```

**实验 (f) — Registers + Frozen**

```bash
python navsim/planning/script/run_training.py \
  agent=mom_driver_agent \
  experiment_name=ablation_bb_reg_frozen \
  "agent.config.num_scene_tokens"=16 \
  "agent.config.image_backbone_use_feature_pooling"=false \
  "agent.config.image_backbone_compress_fc"=false \
  "agent.config.image_backbone_focus_front_cam"=false \
  "agent.config.image_backbone_use_lora"=false \
  "agent.config.image_backbone_finetune"=false \
  ...
```

#### 代码修改清单（微调策略）

- [x] **Pooling 分支已支持**：`ImgEncoder.forward` 中 `self.use_feature_pooling=True` 时直接池化到 `num_scene_tokens`
- [x] **Decoder Compress 分支已支持**：`compress_fc=True` 且 `focus_front_cam=True` 时走前视压缩分支
- [ ] **No compression 尚未支持**：若要恢复论文 Table 2 的 full-token 实验，必须重写 `MoMBackbone.tokens_per_frame`、`scene_embeds`、`pos_emb` 和下游 reshape 逻辑
- [ ] **建议新增配置互斥检查**：`use_feature_pooling`、`compress_fc`、Registers 三类策略最好在 config 或 agent 初始化时显式互斥

### 5.2 主干模型规模（Backbone Scale）

#### 实验目的

对比不同 DINOv2 规模（Small / Base / Large）对性能和参数量的影响，所有配置统一使用 LoRA + Registers 策略。

#### 实验设计

| 编号 | 模型 | ViT 特征维度 | Params（ViT） | 实验名称 |
|------|------|-------------|--------------|----------|
| BB-S | ViT-S/14 reg4 | 384 | 22M | `base`（基线） |
| BB-B | ViT-B/14 reg4 | 768 | 86M | `ablation_bb_vitb` |
| BB-L | ViT-L/14 reg4 | 1024 | 307M | `ablation_bb_vitl` |

#### 参数修改方法

```bash
# ViT-B
python navsim/planning/script/run_training.py \
  agent=mom_driver_agent \
  experiment_name=ablation_bb_vitb \
  "agent.config.image_backbone_model_name"=timm/vit_base_patch14_reg4_dinov2.lvd142m \
  "agent.config.image_backbone_model_weights"=weights/vit_base_patch14_reg4_dinov2.lvd142m/model.safetensors \
  ...

# ViT-L
python navsim/planning/script/run_training.py \
  agent=mom_driver_agent \
  experiment_name=ablation_bb_vitl \
  "agent.config.image_backbone_model_name"=timm/vit_large_patch14_reg4_dinov2.lvd142m \
  "agent.config.image_backbone_model_weights"=weights/vit_large_patch14_reg4_dinov2.lvd142m/model.safetensors \
  ...
```

> `ImgEncoder.neck` 的 `in_features` 由 `self.model.num_features` 动态读取，**无需修改代码**，Hydra 参数覆盖即可。

---

## 6. 消融四：State-Frozen — β 与 g 的解耦

### 6.1 实验目的

这一组实验用于**支持 decoder cross-attention 设计动机**，而不是作为论文主贡献单独宣称。
按照论文附录 `Theoretical Analysis of Cross-Attention Design` 的表述，我们将
concatenation-based recurrent cross-attention 中的两类偏差拆解为：

- **额外衰减偏差**：query 继续推进 recurrent state 时，会对已累计的 key-value memory 引入额外衰减
- **query 污染偏差**：若 query 也允许写入 state，则后续 query 读取到的状态不再是同一个 key-value memory

在简化分析中，这两类效应分别对应 query 段上的：

- `g = 0`：不再对历史状态引入额外衰减（等价于附录中的 `\lambda_t = 1`）
- `beta = 0`：query 不写入 memory

因此，本节通过分别冻结 `beta` 与 `g`，验证完整 State-Frozen 是否比部分冻结更合理，并判断该设计在实际自动驾驶任务中的收益是否显著。

### 6.2 实验设计

| 编号 | `use_state_frozen` | `freeze_beta_only` | `freeze_g_only` | Query 写入 | Query 衰减 | 设计含义 | 实验名称 |
|------|--------------------|--------------------|-----------------|------------|------------|----------|----------|
| SF-0 | false | false | false | 允许 | 允许 | **LICA baseline** | `ablation_sf_lica` |
| SF-1 | false | true | false | 禁止 | 允许 | 只去除 query 污染 | `ablation_sf_beta0` |
| SF-2 | false | false | true | 允许 | 禁止 | 只去除额外衰减 | `ablation_sf_g0` |
| SF-3 | true | false | false | 禁止 | 禁止 | **完整 State-Frozen** | `ablation_sf_frozen` |

> - SF-0 对应附录中的 concatenation-based recurrent cross-attention
> - SF-1 只保留“共享 memory、无 query 写入”这一条件
> - SF-2 只保留“无额外衰减”这一条件
> - SF-3 对应附录中的 State-Frozen cross-attention：所有 query 从同一个 frozen key-value memory 读取
> - 当前实现已贯通 `MoMConfig` → `transformer_decoder.py` → 本地 `flash-linear-attention/fla/layers/mom.py`
> - 运行实验时需确保 Python 优先加载仓库内的 `flash-linear-attention/`，不要误用环境里旧版 `fla`

#### 论文中的预期解读

- 若 **SF-3 明显优于 SF-0**，说明完整 State-Frozen 在真实驾驶任务中有实际收益
- 若 **SF-1 与 SF-2 各有部分提升，但都不如 SF-3**，则支持“query 写入”和“额外衰减”都在起作用
- 若 **SF-3 仅小幅优于 SF-0**，则应在论文中将其定位为“设计分析与合理化”，而非主要性能来源
- 若 **SF-1 或 SF-2 已接近 SF-3**，则说明主要问题集中在单一偏差来源，应在论文讨论中如实说明

### 6.3 参数修改方法

四个实验现在都可以直接通过 Hydra 参数控制，无需再改代码。

```bash
# SF-0: LICA baseline
python navsim/planning/script/run_training.py \
  agent=mom_driver_agent \
  experiment_name=ablation_sf_lica \
  "agent.config.use_state_frozen"=false \
  "agent.config.freeze_beta_only"=false \
  "agent.config.freeze_g_only"=false \
  ...

# SF-1: 仅冻结 beta
python navsim/planning/script/run_training.py \
  agent=mom_driver_agent \
  experiment_name=ablation_sf_beta0 \
  "agent.config.use_state_frozen"=false \
  "agent.config.freeze_beta_only"=true \
  "agent.config.freeze_g_only"=false \
  ...

# SF-2: 仅冻结 g
python navsim/planning/script/run_training.py \
  agent=mom_driver_agent \
  experiment_name=ablation_sf_g0 \
  "agent.config.use_state_frozen"=false \
  "agent.config.freeze_beta_only"=false \
  "agent.config.freeze_g_only"=true \
  ...

# SF-3: State-Frozen（完整）
python navsim/planning/script/run_training.py \
  agent=mom_driver_agent \
  experiment_name=ablation_sf_frozen \
  "agent.config.use_state_frozen"=true \
  "agent.config.freeze_beta_only"=false \
  "agent.config.freeze_g_only"=false \
  ...
```

#### 推荐运行方式（强制使用本地 FLA）

```bash
PYTHONPATH=/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim/flash-linear-attention:/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim \
python navsim/planning/script/run_training.py \
  agent=mom_driver_agent \
  experiment_name=ablation_sf_beta0 \
  "agent.config.use_state_frozen"=false \
  "agent.config.freeze_beta_only"=true \
  "agent.config.freeze_g_only"=false \
  ...
```

#### 当前代码逻辑说明

```python
# transformer_decoder.py: cross-attn 中的优先级
if self.use_state_frozen:
    frozen_from = kv_len
    freeze_beta_from = -1
    freeze_g_from = -1
elif self.freeze_beta_only:
    freeze_beta_from = kv_len
elif self.freeze_g_only:
    freeze_g_from = kv_len
else:
    frozen_from = -1
```

FLA 侧已经把单一 `frozen_from` 扩展成了独立接口：

```python
# fla/layers/mom.py
def forward(..., frozen_from=-1, freeze_beta_from=-1, freeze_g_from=-1):
    ...
```

其中：

- `frozen_from`：完整 State-Frozen，等价于同时设置 `freeze_beta_from` 和 `freeze_g_from`
- `freeze_beta_from`：只对 query 段清零 `beta`
- `freeze_g_from`：只对 query 段清零 `g`

#### 运行前检查

```bash
PYTHONPATH=/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim/flash-linear-attention:/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim \
python -c "import inspect; from fla.layers.mom import MomAttention; print(inspect.signature(MomAttention.forward))"
```

#### 论文表格模板（LaTeX）

```latex
\begin{table}[t]
\centering
\caption{Ablation on State-Frozen cross-attention. We separately disable query-state writing and query-induced decay to study their effect in concatenation-based recurrent cross-attention.}
\label{tab:state_frozen_ablation}
\setlength{\tabcolsep}{6pt}
\resizebox{0.95\columnwidth}{!}{%
\begin{tabular}{lcccccc}
\toprule
ID & $\beta_q{=}0$ & $g_q{=}0$ & Query Write & Query Decay & Interpretation & EPDMS$\uparrow$ \\
\midrule
SF-0 &  &  & Yes & Yes & LICA baseline & \\
SF-1 & \checkmark &  & No & Yes & remove query contamination only & \\
SF-2 &  & \checkmark & Yes & No & remove extra decay only & \\
SF-3 & \checkmark & \checkmark & No & No & State-Frozen & \\
\bottomrule
\end{tabular}}
\end{table}
```

如果不想使用 `\checkmark`，可以替换成 `Yes/No` 版本：

```latex
\begin{table}[t]
\centering
\caption{Ablation on State-Frozen cross-attention.}
\label{tab:state_frozen_ablation}
\setlength{\tabcolsep}{6pt}
\resizebox{0.9\columnwidth}{!}{%
\begin{tabular}{lccccc}
\toprule
ID & $\beta_q{=}0$ & $g_q{=}0$ & Main Effect Removed & Interpretation & EPDMS$\uparrow$ \\
\midrule
SF-0 & No  & No  & None                  & LICA baseline                  & \\
SF-1 & Yes & No  & query contamination   & beta-only frozen               & \\
SF-2 & No  & Yes & extra decay           & g-only frozen                  & \\
SF-3 & Yes & Yes & both                  & State-Frozen                   & \\
\bottomrule
\end{tabular}}
\end{table}
```

---

## 7. 实验执行说明

### 7.1 实验命名规范

```
{消融组}_{变量描述}
例：ablation_seq_t4_e12  /  ablation_enc_mem8k4  /  ablation_sf_frozen
```

所有实验 checkpoint 存储于：
```
$NAVSIM_EXP_ROOT/{experiment_name}/lightning_logs/version_0/checkpoints/
```

### 7.2 训练脚本模板

```bash
#!/bin/bash
# scripts/training/run_ablation.sh

export NAVSIM_EXP_ROOT="/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim/exp"
export NUPLAN_MAPS_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1"
export NAVSIM_DEVKIT_ROOT="/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT/flash-linear-attention:$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

EXP_NAME=$1          # 实验名，作为第一个参数传入
EXTRA_ARGS="${@:2}"  # 额外的 Hydra 覆盖参数

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
  agent=mom_driver_agent \
  experiment_name=$EXP_NAME \
  "dataloader.params.batch_size"=1 \
  "dataloader.params.num_workers"=16 \
  "train_test_split.scene_filter.num_history_frames"=10 \
  "trainer.params.strategy"=ddp \
  train_test_split=navtrain \
  cache_path=$NAVSIM_EXP_ROOT/momdriver_cache \
  use_cache_without_dataset=true \
  $EXTRA_ARGS
```

使用示例：
```bash
# 消融 Encoder Memory（ME-4）
bash scripts/training/run_ablation.sh ablation_enc_mem8k2 \
  "agent.config.mom_temporal.num_memories"=8 \
  "agent.config.mom_temporal.topk"=2

# 消融 State-Frozen（SF-3）
bash scripts/training/run_ablation.sh ablation_sf_frozen \
  "agent.config.use_state_frozen"=true
```

### 7.3 评估脚本模板

```bash
#!/bin/bash
# scripts/evaluation/run_ablation_eval.sh

export NAVSIM_EXP_ROOT="/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim/exp"
export NUPLAN_MAPS_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1"
export NAVSIM_DEVKIT_ROOT="/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim"
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT/flash-linear-attention:$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

EXP_NAME=$1
CHECKPOINT=$2
EVAL_FRAMES=${3:-10}    # 默认评估 10 帧
EXTRA_ARGS="${@:4}"
CACHE_PATH="/mnt/workspace/nanyi/navsim_workspace/exp/metric_cache"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_rwkv.py \
  train_test_split=navtest \
  worker=ray_distributed \
  metric_cache_path=$CACHE_PATH \
  agent=mom_driver_agent \
  agent.checkpoint_path=$CHECKPOINT \
  "train_test_split.scene_filter.num_history_frames"=$EVAL_FRAMES \
  enable_padding=true \
  experiment_name=eval_$EXP_NAME \
  $EXTRA_ARGS
```

使用示例：
```bash
# 标准评估
bash scripts/evaluation/run_ablation_eval.sh \
  ablation_enc_mem8k2 $CKPT 10

# 跨帧数评估（Train=4, Eval=12）
bash scripts/evaluation/run_ablation_eval.sh \
  ablation_seq_t4_e12 $CKPT_T4 12 \
  "agent.config.cam_seq_len"=12
```

### 7.4 代码修改优先级与依赖关系

```
优先级 P0（必须，阻塞其他实验）：
  └── 消融一：确认 valid_frame_len > cam_seq_len 时 _build_attention_mask 的正确性
              文件：mom_backbone.py:485-510

优先级 P1（必须）：
  ├── 消融四 SF-1/SF-2：确保训练 / 评估脚本始终优先加载本地 flash-linear-attention
  │   文件：scripts/* 或 shell 环境变量
  └── 消融三：明确压缩策略互斥关系，避免 Pooling / Decoder Compress / Registers 混用
      文件：mom_backbone.py 或 agent config 校验

优先级 P2（可选，增强实验）：
  ├── 消融一：流式评估模式支持
  └── 消融三：No compression 动态 token 支持（需重构 tokens_per_frame 与 pos_emb）
```

---

## 8. 结果汇总模板

### 8.1 消融一：时序长度泛化

| Train \ Eval | 4 | 8 | 12 | 20 | ∞ |
|---|---|---|---|---|---|
| **4** | NC / DAC / EP / EPDMS | — | — | — | — |
| **12** | — | — | — | — | — |
| **20** | — | — | — | — | — |

### 8.2 消融二：Memory Slots

| Encoder Ablation ID | Enc Mem/TopK | EPDMS | Decoder Ablation ID | Dec Mem/TopK | EPDMS |
|------|---|---|---|---|---|
| MS-1 | 1/1 | | MS-6 | 1/1 | |
| MS-2 | 2/1 | | MS-7 | 2/1 | |
| MS-3 | 2/2 | | MS-8 | 2/2 | |
| **MS-0$^\dagger$** | **4/2** | **基线** | **MS-0$^\dagger$** | **4/2** | **基线** |
| MS-4 | 8/2 | | MS-9 | 8/2 | |
| MS-5 | 8/4 | | MS-10 | 8/4 | |

### 8.3 消融三：图像主干

| 编号 | 压缩方式 | 约束 | Cam tokens | Scene tokens | Img. Enc. training | Params Optim | Params Total | EPDMS |
|---|---|---|---|---|---|---|---|---|
| (a) | Pooling | `use_feature_pooling=True` | 16 | 64 | Frozen | | | |
| (b) | Pooling | `use_feature_pooling=True` | 16 | 64 | LoRA | | | |
| (c) | Decoder Compress | `compress_fc=True, focus_front_cam=True` | 16(front)+48(other) | 64 | Frozen | | | |
| (d) | Decoder Compress | `compress_fc=True, focus_front_cam=True` | 16(front)+48(other) | 64 | LoRA | | | |
| (e) | Registers | `num_scene_tokens=16` | 16 | 64 | Full ft. | | | |
| (f) | Registers | `num_scene_tokens=16` | 16 | 64 | Frozen | | | |
| **(g)** | **Registers** | **num_scene_tokens=16** | **16** | **64** | **LoRA** | | | **基线** |

| 主干规模 | 模型 | Params Total | EPDMS |
|---|---|---|---|
| ViT-S（基线） | vit_small_patch14_reg4_dinov2 | 41.8M | |
| ViT-B | vit_base_patch14_reg4_dinov2 | ~107M | |
| ViT-L | vit_large_patch14_reg4_dinov2 | ~328M | |

### 8.4 消融四：State-Frozen 解耦

| 编号 | `use_state_frozen` | `freeze_beta_only` | `freeze_g_only` | Query Write | Query Decay | Main Effect Removed | EPDMS |
|---|---|---|---|---|---|---|---|
| SF-0 | false | false | false | Yes | Yes | None | |
| SF-1 | false | true | false | No | Yes | query contamination | |
| SF-2 | false | false | true | Yes | No | extra decay | |
| SF-3 | true | false | false | No | No | both | |

---

*文档维护：每次补充实验结果后更新对应表格，并在 git commit 中注明实验编号。*
