# 模型架构对比：DrivoR / MoM / RWKV7_mf

> 本文档对比 `DrivoR`、`MoM_driver`、`rwkv7_mf` 三个自动驾驶规划模型的数据流程与架构差异。

---

## 一、MoM Driver — 数据流程与架构

### 数据流程（端到端）

```
输入 (AgentInput)
  ├─ 4路摄像头 (f0, b0, l0, r0)  × 10帧
  └─ EgoStatus (pose+vel+acc+cmd = 11维) × 10帧
         │
         ▼ MoMFeatureBuilder
  features["image"]           [B, 10, 4, 3, H, W]  (左填充至10帧)
  features["ego_status"]      [B, 10, 11]
  features["valid_frame_len"] [B]  (实际帧数)
         │
         ▼ MoMModel.forward()
  ┌──────────────────────────────────────────┐
  │  Step 1: MoMBackbone                      │
  │   ├─ reshape → [B*10, 4, 3, H, W]         │
  │   ├─ DINOv2+LoRA + 可学习scene_embeds      │
  │   │   → [B*10, 4*16, 256]                  │
  │   ├─ reshape → [B, 10, 64, 256]            │
  │   ├─ 加逐帧位置编码 pos_emb               │
  │   ├─ reshape → [B, 640, 256]               │
  │   └─ 2层 MomBlock (gated_deltanet)        │
  │       (含valid_frame_len注意力mask)        │
  │   → scene_features [B, 640, 256]           │
  │                                            │
  │  Step 2: Ego Token + 轨迹初始化            │
  │   ├─ ego_status[:,-1] → Linear → [B,1,256] │
  │   └─ + init_feature.weight → [B,64,256]    │
  │                                            │
  │  Step 3: 初始提案                          │
  │   └─ traj_head[0] → proposals [B,64,8,3]   │
  │                                            │
  │  Step 4: 迭代精化 (4轮)                    │
  │   └─ MoMTransformerDecoder                 │
  │       ├─ query_attn: MomBlock (自注意力)   │
  │       └─ cross_attn: concat[kv,q]→MomBlock │
  │           → 每轮输出 proposals [B,64,8,3]  │
  │                                            │
  │  Step 5: Scorer                            │
  │   ├─ pos_embed(proposals) → [B,64,256]     │
  │   ├─ MoMTransformerDecoderScorer (4轮)     │
  │   ├─ + ego_token → [B,64,256]              │
  │   └─ Scorer → pred_logit (6个PDM子分)     │
  │                                            │
  │  Step 6: 选择轨迹                          │
  │   ├─ pdm_score = noc+dac+ddc+(ttc+ep+com) │
  │   └─ argmax → trajectory [B,8,3]           │
  └──────────────────────────────────────────┘
```

### 核心架构组件

| 组件 | 实现 |
|------|------|
| 图像编码器 | DINOv2 ViT-S/14 + LoRA (rank=32) |
| 时序融合 | 2层 `MomBlock` (fla库, gated_deltanet) |
| 轨迹解码器 | 4层 `MoMDoubleSelfAttnBlock` (MomBlock) |
| 打分解码器 | 4层 `MoMDoubleSelfAttnBlock` (MomBlock) |
| 提案数 | 64条轨迹，每条8个pose |

---

## 二、MoM vs DrivoR — 核心差异对比

### 1. 时序融合策略（最核心差异）

| | DrivoR | MoM |
|---|---|---|
| **输入帧数** | **仅最后1帧** (`cameras[-1]`) | **10帧历史** (`cam_seq_len=10`) |
| **时序建模** | **无** — 每帧独立 | **MomBlock 时序融合** — 跨帧序列建模 |
| **scene_features形状** | `[B, 4*16, 256]` = `[B, 64, 256]` | `[B, 10*64, 256]` = `[B, 640, 256]` |
| **attention mask** | 无 | 按valid_frame_len构建，屏蔽左填充帧 |

### 2. Transformer解码块（第二核心差异）

| | DrivoR | MoM |
|---|---|---|
| **解码块类型** | 标准 `Block` (MultiheadAttention) | `MoMDoubleSelfAttnBlock` (MomBlock) |
| **自注意力** | `nn.MultiheadAttention` (Softmax attn) | `MomBlock` (线性递归注意力) |
| **交叉注意力** | 独立的 Q/KV 线性映射 + Softmax | concat([KV, Q]) → MomBlock → 取末尾Q部分 |
| **State-Frozen模式** | 无 | 支持 (`use_state_frozen=True`) — 冻结beta/g消除ε_decay和ε_contam |
| **帧掩码** | 无 | `_mask_invalid_frames` 将无效帧token置零 |
| **DropPath/LayerScale** | 有 (drop_path=0.2, ls_values) | 无（依靠MomBlock内部机制） |

### 3. 特征构建（Feature Builder）

| | DrivoR | MoM |
|---|---|---|
| **摄像头帧** | 仅 `cameras[-1]`（最新帧） | `cameras[0..T-1]`（所有帧） |
| **输出image形状** | `[N_cams, 3, H, W]`（无时间维） | `[seq_len, N_cams, 3, H, W]`（有时间维） |
| **内参/外参** | 提取 cam_K, world_2_cam | **不提取**（MoM不做显式3D投影） |
| **lidar支持** | 支持（2D histogram） | 不支持 |
| **ego_status** | 仅当前帧或4帧 (full_history_status) | 最多10帧 (full_history_status) |

### 4. Agent接口与训练

| | DrivoR | MoM |
|---|---|---|
| **配置格式** | OmegaConf dict | Python dataclass (`MoMConfig`) |
| **LR调度** | LinearLR warmup + CosineAnnealingLR | AdamW 固定lr（无scheduler） |
| **Ray并行评分** | 支持 (`worker_ray_no_torch`) | 不支持（顺序执行） |
| **Metric cache** | 支持合成数据集 (5个synthetic splits) | 不支持合成数据 |
| **训练回调** | ModelCheckpoint + LRMonitor | 无（由外部管理） |

### 5. 共享组件（两者基本相同）

- **Scorer模块**：完全相同的 `Scorer` 类 + 6个PDM子分
- **PDM打分公式**：完全相同（noc·log + dac·log + ddc·log + log(ttc+ep+comfort)）
- **轨迹选择**：相同（argmax pdm_score）
- **轨迹头**：相同（`ref_num+1` 个 MLP）
- **pos_embed**：相同（Linear(8×3→ffn→d_model)）
- **DINOv2+LoRA**：相同的 `ImgEncoder` 实现（MoM直接移植自DrivoR）

---

## 三、RWKV7_mf vs MoM vs DrivoR — 三模型全面对比

### 1. 整体架构范式

| 维度 | RWKV7_mf (v3，默认) | MoM | DrivoR |
|---|---|---|---|
| **轨迹生成范式** | **扩散模型** (DDIM去噪) | **提案精化** (迭代refinement) | **提案精化** (迭代refinement) |
| **传感器模态** | **相机 + LiDAR** (双模态) | 纯相机 | 纯相机（可选LiDAR） |
| **图像编码器** | **ResNet34** (timm, pretrained=False) | **DINOv2-S ViT + LoRA** | **DINOv2-S ViT + LoRA** |
| **时序建模粒度** | **特征图级别** (4个ResNet scale内部融合) | **token序列级别** (DINOv2 token后融合) | **无时序建模** |
| **注意力核** | **RWKV7Block** (线性递归) | **MomBlock** (gated_deltanet) | **MultiheadAttention** (Softmax) |

### 2. 数据流对比

#### RWKV7_mf (v3)

```
输入:
  camera_feature [B, 10, 3, 256, 1024]  (拼接全景图)
  lidar_feature  [B, 10, 1, 256, 256]   (2D histogram)
  status_feature [B, 8]  (当前帧: cmd+vel+acc)
       │
       ▼ RWKVBackbone (4-scale ResNet + RWKV7 融合)
  ┌──────────────────────────────────────────────┐
  │ 4× {ResNet34 block (image & lidar) →          │
  │      拼接 [img_pool + lidar_pool, d_model] →   │
  │      RWKV7Block × 2 (image⊕lidar 跨帧融合) →  │
  │      双路残差加回 image/lidar 特征图}           │
  │ → bev_feature [B*10, 512, 8, 8]               │
  │ → bev_upscale [B*10, 64, 32, 64] (FPN top-down)│
  └──────────────────────────────────────────────┘
       │
       ▼ bev_downscale + reshape
  keyval: [B, 10*64+1, 256]  (BEV tokens + status token)
       │
       ▼ 3层 RWKV7_DoubleSelfAttnBlock
  agents_query [B, 30, 256]
       │
  ┌────────────────────────────────────────────────┐
  │ AgentHead → agent_states [B, 30, 5]            │
  │                                                │
  │ TrajectoryHead (扩散解码器)                     │
  │   ├─ 20条轨迹锚点 + 噪声 → noisy_traj [B,20,8,3]│
  │   ├─ 2层 RWKV_DiffusionDeCoderLayer:           │
  │   │   cross_bev_attention  (RWKV7 × bev_feature)│
  │   │   cross_agent_attention (RWKV7 × agents_query)│
  │   │   FFN + ModulationLayer (timestep调制)      │
  │   │   → poses_reg [B,20,8,3] + PDM score [B,20,6]│
  │   └─ DDIM 2步去噪 → trajectory [B,8,3]         │
  └────────────────────────────────────────────────┘
  BEV语义分割头 → bev_semantic_map [B, 7, 128, 256]
```

#### MoM

```
输入:
  image [B, 10, 4, 3, H, W]  (4路相机 × 10帧)
  ego_status [B, 10, 11]
  valid_frame_len [B]
       │
       ▼ MoMBackbone
  ┌──────────────────────────────────────────┐
  │ DINOv2-S + LoRA → scene_embeds           │
  │ [B*10, 4*16, 256]                         │
  │ → 2层 MomBlock 时序融合                  │
  │ → scene_features [B, 640, 256]            │
  └──────────────────────────────────────────┘
       │
       ▼ 64条可学习提案 + 4层 MoMDoubleSelfAttnBlock
  proposal_list: 5×[B,64,8,3]
       │
       ▼ pos_embed + 4层 MoMDoubleSelfAttnBlock (scorer)
  + PDM Scorer → pdm_score [B,64]
       │
       ▼ argmax → trajectory [B,8,3]
```

#### DrivoR

```
输入:
  image [B, N_cams, 3, H, W]  (最新1帧, N_cams≤8)
  ego_status [B, 11]
       │
       ▼ ImgEncoder (DINOv2 + LoRA)
  scene_features [B, N_cams*16, 256]
       │
       ▼ 64条可学习提案 + 4层 TransformerDecoder (Softmax Attn)
  proposal_list: 5×[B,64,8,3]
       │
       ▼ 4层 TransformerDecoderScorer + PDM Scorer
  pdm_score [B,64] → argmax → trajectory [B,8,3]
```

### 3. 时序融合策略差异（核心）

| | RWKV7_mf | MoM |
|---|---|---|
| **融合时机** | **Backbone 内部**，4个ResNet层级各融合一次 | **Backbone 之后**，DINOv2 token序列统一融合 |
| **融合粒度** | 特征图 (8×8 到 32×64 的 spatial map) | token序列 (64 tokens/帧 × 10帧 = 640) |
| **融合方式** | 相机+LiDAR 拼接 → RWKV7Block (跨帧+跨模态同时) | 纯相机 tokens → MomBlock (仅跨帧) |
| **Block类型** | RWKV7Block（单注意力） | MomBlock（gated_deltanet，含MoE memory） |
| **帧掩码** | attention_mask 按 valid_frame_len | attention_mask 按 valid_frame_len |
| **解码端** | 3层 RWKV7_DoubleSelfAttnBlock | 4层 MoMDoubleSelfAttnBlock |

### 4. 解码器注意力块对比

| | RWKV7_DoubleSelfAttnBlock | MoMDoubleSelfAttnBlock | DrivoR Block |
|---|---|---|---|
| **底层实现** | `RWKV7Block` (fla库, RWKV7) | `MomBlock` (fla库, gated_deltanet + MoE memory) | `nn.MultiheadAttention` (Softmax) |
| **交叉注意力** | concat([kv, q]) → RWKV7Block | concat([kv, q]) → MomBlock | 独立 Q/KV 线性映射 + Softmax |
| **State-Frozen** | 无 | 支持 (`use_state_frozen=True`) | 无 |
| **v_first 传递** | 显式传递 `v_first` 状态 | 不需要（MomBlock内部管理） | 无状态 |
| **DropPath/LayerScale** | 无 | 无 | 有 (drop_path=0.2) |

### 5. 轨迹生成范式差异（最关键）

| | RWKV7_mf | MoM | DrivoR |
|---|---|---|---|
| **提案来源** | **20条固定锚点** (从文件加载 `plan_anchors_100.npy`) | **64条可学习embedding** (`nn.Embedding`) | **64条可学习embedding** (`nn.Embedding`) |
| **生成机制** | **扩散去噪** (DDIM, 训练50步/推理2步) | **迭代MLP精化** (无噪声，直接回归) | **迭代MLP精化** (无噪声，直接回归) |
| **噪声注入** | ✅ 训练加高斯噪声，推理从锚点+截断噪声出发 | ❌ 无噪声 | ❌ 无噪声 |
| **时间步条件** | ✅ SinusoidalPosEmb + Mish + ModulationLayer | ❌ 无 | ❌ 无 |
| **选择方式** | PDM score argmax | PDM score argmax | PDM score argmax |
| **Scorer架构** | `DiffMotionPlanningRefinementModule` (MLP) | `Scorer` 类（与DrivoR共享） | `Scorer` 类 |

### 6. 输出 Head 对比

| 输出 | RWKV7_mf | MoM | DrivoR |
|---|---|---|---|
| trajectory | [B, 8, 3] | [B, 8, 3] | [B, 8, 3] |
| proposals | [B, 20, 8, 3] | [B, 64, 8, 3] | [B, 64, 8, 3] |
| bev_semantic_map | ✅ [B, 7, 128, 256] | 可选 | 可选 |
| agent_states | [B, 30, 5] | 可选 | 可选 |
| agent_labels | [B, 30] | 可选 | 可选 |
| denoised_trajectories | ✅ 扩散去噪中间结果 | ❌ | ❌ |

---

## 四、三模型关系与演进路径

```
DrivoR
  │
  │  保留: DINOv2+LoRA + 提案精化 + Scorer
  │  替换: Softmax Attn → RWKV7 递归线性注意力
  │  增加: LiDAR 双模态融合 + 扩散轨迹生成 + 多帧历史
  │  替换: 图像编码器 DINOv2 → ResNet34
  ▼
RWKV7_mf
  │
  │  保留: 多帧10帧历史 + 线性递归注意力解码
  │  替换: RWKV7Block → MomBlock (gated_deltanet)
  │  替换: Backbone内融合 → token层级融合
  │  替换: 扩散轨迹 → 提案精化（继承自DrivoR）
  │  恢复: 图像编码器 ResNet34 → DINOv2+LoRA
  │  去掉: LiDAR
  ▼
MoM
  │
  │  保留: DINOv2+LoRA + 提案精化 + Scorer（来自DrivoR）
  │  保留: 多帧历史时序建模（来自RWKV7_mf）
  │  创新: 时序融合从 Backbone 内部移至 token 序列层级
  │  创新: State-Frozen 交叉注意力（消除ε_decay/ε_contam）
  │  去掉: LiDAR 与扩散去噪
```

### 设计理念对比

| 视角 | RWKV7_mf | MoM | DrivoR |
|---|---|---|---|
| **核心创新** | 扩散+RWKV双模态时序规划 | MomBlock token级时序融合 + State-Frozen | DINOv2+LoRA 单帧提案精化 |
| **计算复杂度** | 较高（ResNet×10帧×4scale + RWKV7融合） | 中（DINOv2×10帧 + MomBlock融合） | 低（DINOv2×1帧 + Softmax解码） |
| **模态灵活性** | 相机+LiDAR耦合 | 纯相机 | 相机为主，可选LiDAR |
| **时序信息利用** | Backbone级别跨帧融合 | Token序列级别跨帧融合 | 无时序 |
