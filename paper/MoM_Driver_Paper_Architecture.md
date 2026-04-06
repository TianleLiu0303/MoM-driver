# MoM-Driver: Mixture-of-Memories Linear Attention for Streaming End-to-End Autonomous Driving

## Current Paper Architecture

---

## 论文定位

- **标题**: MoM-Driver: Mixture-of-Memories Linear Attention for Streaming End-to-End Autonomous Driving
- **投稿会议**: NeurIPS 2026
- **当前版本定位**: 主文 + 附录的结构化整理稿
- **核心主线**: 用 Mixture-of-Memories 线性注意力解决自动驾驶长时序流式推理中的容量瓶颈与遗忘问题

---

## 当前稿件的核心贡献

### 1. Multi-slot recurrent memory for driving

- 将 MoM 引入端到端自动驾驶时序融合。
- 用多个 memory slots 替代单一 recurrent state，配合 top-$K$ routing 做选择性读写。
- 论文中强调的收益是：增加有效记忆容量、减少异构驾驶信息之间的干扰、缓解长时序遗忘。

### 2. Streaming inference with constant complexity

- 保留线性注意力的 recurrent 推理形式。
- 推理时仅跨帧传递固定大小 memory state，不保存完整历史帧。
- 论文中明确强调部署属性：constant-complexity inference、streaming deployment、online deployment。

### 3. State-Frozen cross-attention in the decoder

- 解码器中的 query 只读 scene memory，不向感知 memory 写入内容。
- 目标是避免 query contamination 和额外 decay 带来的 cross-attention 偏差。
- 这部分在主文中作为关键设计，在附录中给出理论化解释。

### 4. Comprehensive evaluation across planning settings

- NavSim v2: open-loop / pseudo-simulation planning。
- Bench2Drive: closed-loop CARLA benchmark。
- 额外包含 long-horizon、efficiency、memory-slot 和 State-Frozen 等 ablations。

---

## 不应过度宣称的部分

### 可以说

- 首次将 MoM-style multi-slot recurrent memory 系统性用于当前这类 streaming end-to-end driving pipeline。
- 在 open-loop、closed-loop、long-horizon 三类设置下验证了多槽记忆的价值。

### 建议弱化

- “首个支持流式推理的多槽位记忆自动驾驶系统” 这种绝对化表述可以保留为内部表述，但论文里最好避免过强 claim。
- “4x state capacity” 可以作为直观解释，但正文里更稳妥的表述是 “total capacity expands to $Md^2$”。
- “slot specialization” 目前更适合作为可视化观察或 future analysis，不应写成已经被充分验证的主结论。

---

## 主文结构对齐

### Abstract

当前摘要结构已经比较完整，逻辑是：

1. 长时序建模对端到端自动驾驶重要。
2. 现有方法在 Transformer temporal fusion 和 recurrent linear attention 之间存在效率与容量权衡。
3. 提出 MoM-Driver，用 multi-slot memory + dynamic top-$K$ routing 替代 single-state recurrence。
4. 在 open-loop、closed-loop、long-horizon 上验证 temporal robustness，同时保持 streaming inference efficiency。

### 1. Introduction

当前引言主线已经成型：

1. 端到端自动驾驶发展迅速，但 motion planning 仍是部署瓶颈。
2. 时序建模对遮挡推理、交互理解、行为一致性至关重要。
3. Transformer temporal fusion 成本随序列长度增长，不利于长时序流式部署。
4. 线性注意力虽然适合 streaming inference，但 single shared state 容量有限，容易产生 interference 和 forgetting。
5. MoM-Driver 用多槽记忆和 routing 解决这个容量瓶颈。

### 2. Related Work

当前稿件的 related work 分为两部分，结构合理：

1. End-to-End Autonomous Driving
2. Temporal Modeling for Driving

建议在最终版本里保持以下定位：

- 不重新发明视觉编码，而是沿用 DrivoR 风格 scene compression。
- 论文真正聚焦的是 temporal backbone 和 memory architecture。
- MoM 与 MoE 的联系应点到为止，不要展开到语言模型太多篇幅。

### 3. Method

主文方法部分当前已经是完整版本，可以稳定采用以下结构：

1. Overview
2. Perception Encoder
3. MoM Temporal Fusion
4. Trajectory Decoder
5. Scoring and Selection
6. Training and Inference

#### 3.1 Overview

- 输入：多相机图像 + ego state。
- 输出：轨迹 proposal + PDM scorer 选择最终轨迹。
- 四阶段流水线描述清晰，图 `Figures/Overview.pdf` 已对应这部分。

#### 3.2 Perception Encoder

- 采用 DINOv2-Small + LoRA。
- 每个 camera 使用 $N_s$ 个可学习 register / scene tokens 压缩视觉内容。
- 每帧输出 $C \times N_s$ 个 scene tokens，供后续 temporal fusion 使用。
- 与 DrivoR 的关系已经讲清楚：沿用 scene compression，但扩展到 multi-frame temporal fusion。

#### 3.3 MoM Temporal Fusion

这是全文的技术核心，当前版本写法基本正确：

- 先回顾 single-state recurrent linear attention 的容量瓶颈。
- 再引出 MoM 的 $M$ 个并行 memory slots。
- 通过 router 对 token 进行 top-$k$ 选择性读写。
- 输出时用 routing weight 对多个 slots 聚合。
- 强调训练时可以并行 scan，推理时保持增量更新。

建议注意两点：

1. 正文里要统一符号。
当前有些地方用 $\mathbf{S}_t = \boldsymbol{\Lambda}_t \odot \mathbf{S}_{t-1} + \mathbf{k}_t^\top \mathbf{v}_t$，有些地方写法是 $\mathbf{k}_t \mathbf{v}_t^\top$，最终版本最好固定一种矩阵乘法约定。

2. shared slot 的描述建议更明确。
你们正文里写了 “supplemented by a shared slot updated by every token for global context”，但公式里没有单独展开。最好在正文或附录补一句实现上 shared slot 是否计入 $M$，以及和 routed slots 的组合方式。

#### 3.4 Trajectory Decoder

当前结构清晰：

- learnable trajectory queries；
- ego-state embedding 注入；
- self-attention + state-frozen cross-attention；
- per-query MLP 输出 trajectory proposals；
- WTA trajectory loss。

这里最重要的是保持“State-Frozen 是 decoder cross-attention 设计，而不是 temporal encoder 的默认机制”这一本质区别，避免读者误解。

#### 3.5 Scoring and Selection

这一节与 DrivoR 风格对齐，合理：

- proposal geometry 重新嵌入为 score queries；
- stop-gradient 保持生成与评分解耦；
- scorer decoder 预测六个 PDMS-related sub-scores；
- inference 时聚合 sub-scores 选最终 proposal。

建议补充一小句：最终聚合权重是否固定、是否和 benchmark convention 一致、是否在所有实验中保持不变。

#### 3.6 Training and Inference

主文当前给出：

- 总损失 $\mathcal{L}=\mathcal{L}_{traj}+\lambda_s\mathcal{L}_{score}$；
- 训练设置：100 epochs，Adam，lr $1\times 10^{-4}$，batch size 4，4x RTX 4090；
- 推理 persistent state 大小约 1 MB。

这部分已经足够支撑主文，但附录需要进一步补齐 implementation-specific 细节。

---

## 实验结构对齐

### 4. Experiments

当前实验章节划分为：

1. Benchmarks
2. NavSim v2 results
3. Bench2Drive results
4. Long-time Streaming Inference
5. Efficiency of On-Device Deployment
6. Ablation Studies

这个结构是合理的，但有几处明显还未完成。

### 已有结果表

#### NavSim v2

- 有完整大表。
- 但需要核查你们 `MoM-Driver (ours)` 行目前和 `ZTRS` 行数值完全相同，这在投稿里会非常危险。
- 如果这是占位复制，必须替换为真实结果；如果确实相同，也要解释 backbone 或 setting 差异，否则 reviewer 会直接质疑表格可信度。

#### Bench2Drive

- 表格结构清楚。
- `MoM-driver(Ours)` 的 DS / SR 提升很醒目，适合作为主结果表。
- 建议补一个简单文字解释：为什么 Efficiency 和 Comfortness 没有同步最佳，避免 reviewer 只盯住 trade-off。

#### Efficiency on Thor

- 表格与图的 narrative 合理。
- 但当前 `MoM-driver` 与 `LADY` 数值完全一致，也需要确认这是测得结果还是占位继承。
- 如果 MoM 的 latency 确实与 LADY 几乎相同，建议写 “matches LADY while improving temporal capacity”，不要只给相同数字而不解释。

### 尚未完成的实验部分

以下内容现在仍是空表或标题，投稿前必须补全：

1. `Long-time Streaming Inference` 小节正文几乎为空。
2. `Effect of Training and Evaluation History Length` 表格是空的。
3. `Number of Memory Slots` 表格是空的。
4. `State-Frozen` ablation 表格是空的。
5. `Impact of Linear Attention Variants` 表格是空的。

如果篇幅紧张，建议优先级如下：

1. 必补：memory slot ablation。
2. 必补：State-Frozen ablation。
3. 强烈建议：train/eval horizon generalization。
4. 可选：linear-attention variants。

---

## 附录结构对齐

### Appendix A: Implementation Details

当前附录 A 的结构应该固定为：

1. Model Architecture Details
2. Training Details

其中应覆盖：

- 输入和 tokenization。
- 完整 layer config table。
- 参数量拆分。
- 初始化策略。
- 数据预处理和 augmentation。
- optimizer、learning rate、precision、DDP。
- loss 公式和各项权重。
- proposal ranking 的 score aggregation weights。

### Appendix B: Cross-Attention Design Analysis

这个部分当前很合适放附录，不建议升格为主贡献。

建议定位：

- 解释为什么 concatenation-based recurrent cross-attention 会引入额外 decay 和 query contamination。
- 说明 State-Frozen 与 standard cross-attention / linear cross-attention 的关系。
- 给设计动机，而不是包装成完整理论贡献。

### Additional Experiments

目前附录里已经列出：

- NavsimV1
- Bench2Drive multi-ability
- Linear-attention variants
- Image backbone ablation
- Visualizations and limitations

其中最大问题不是结构，而是若干表仍然明显未填满或存在 placeholder citation，比如 `\cite{...}`。这些在最终提交前必须清理。

---

## 当前版本中最需要修正的不一致

### 1. State-Frozen 的默认设置表述不一致

- 主文方法里看起来像默认采用 State-Frozen。
- 附录实现表格里写的是 `State-Frozen: off, implemented in code`。

这两种说法不能同时成立。建议二选一并统一：

1. 如果论文主结果都启用了 State-Frozen，就把附录改成 `enabled by default`。
2. 如果 released code 默认关闭，但论文实验开启，就明确写成 “paper experiments enable State-Frozen; released code retains a switch”。

### 2. 训练配置不一致

主文里是：

- Adam
- batch size 4
- 4x RTX 4090

当前附录文件里是：

- AdamW
- batch size 1
- workers 16
- mixed precision 16-mixed

这不一定矛盾，但需要解释清楚是“论文实验配置”还是“released training script default”。

### 3. 损失定义层级不一致

- 主文只写 `L_traj + L_score`。
- 附录写了更完整的总损失并说明 auxiliary branches 默认关闭。

这本身可以接受，但建议在附录显式写一句：论文主实验中 auxiliary branches disabled，因此有效损失退化为主文公式。

### 4. benchmark naming 和指标命名还需统一

- `NavSim`, `NAVSIM`, `NasimV1`, `NavsimV1` 混用。
- `PDMS`, `PDM Score`, `EPDMS` 混用。

建议全文统一：

- 数据集名用 `NAVSIM v1` / `NAVSIM v2`。
- 指标名首次全称，后文统一简称。

### 5. 部分表格存在明显占位或复用数值

这是当前稿件最影响可信度的地方，比文字问题更严重。

---

## 投稿前建议补充内容

### 必补

1. 完整的 long-horizon ablation 数字。
2. `#memory slots / top-k` 的主消融结果。
3. State-Frozen 消融结果。
4. 训练与推理配置统一说明。
5. 所有表格的真实数值与 citation 清理。

### 强烈建议补

1. 训练和测试使用的历史长度、是否 left-padding、valid length masking。
2. shared slot 的具体实现说明。
3. route command 的编码方式。
4. proposal scoring aggregation 的固定权重。
5. 与 LADY 的 fair comparison setting 说明：相同视觉 backbone、相同数据、相同 decoder capacity、只改 temporal memory。

### 若篇幅允许可补

1. slot utilization / routing entropy 分析。
2. 不同场景类型上的 gains 分解。
3. failure case 可视化。
4. 训练成本与额外参数量分析。

---

## 对当前论文内容的总体判断

### 合理的部分

- 问题定义清楚。
- 方法主线完整且自洽。
- MoM + streaming inference + State-Frozen 的组合有明确动机。
- 主文和附录的章节组织基本符合 NeurIPS 写法。

### 目前的主要风险

- 若干实验表格未完成。
- 多处配置存在前后不一致。
- 部分结果表出现可疑重复数字。
- checklist 中个别回答目前并不严谨，尤其是 statistical significance、open access、new assets、human subjects 等问题，最终提交前建议逐项重写。

### 结论

这篇论文的技术叙事已经比较像可投稿版本，但还不是“可直接投”的状态。现在最需要做的不是再扩展 method，而是：

1. 统一实现细节。
2. 补完关键 ablation。
3. 清理所有占位结果和不一致表述。

---

## 建议的最终文档分工

### 主文保留

- 方法主线。
- 两个主 benchmark 结果。
- efficiency。
- 2 到 3 个最关键 ablations。

### 附录承接

- 完整实现配置。
- cross-attention 分析。
- 更细的 benchmark 拆解。
- 视觉化和 failure cases。
- reproducibility instructions。

---

## 当前最建议立即执行的收尾项

1. 核对 `State-Frozen` 在论文实验里是否默认打开，并统一主文/附录表述。
2. 核对 optimizer、batch size、GPU 配置，区分论文实验设置和代码默认设置。
3. 补齐 3 个空 ablation 表：history length、memory slots、state-frozen。
4. 逐表检查是否有复制粘贴导致的重复结果。
5. 重写 checklist，避免 reviewer 一眼看出模板式回答。
