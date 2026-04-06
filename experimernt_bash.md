# MoM-Driver 实验执行手册（目录规范版）

目标：
- 所有实验产物统一放在 `NAVSIM_EXP_ROOT`
- 数据缓存统一放在子目录 `Dataset_caching`
- 训练结果统一放在子目录 `Train_results`
- 评估结果统一放在子目录 `Evaluation_results`
- 各类消融实验放在对应子目录下，命名清晰易读

---

## 1. 终端传参

你执行的命令都是 Hydra 覆盖参数：

```bash
python run_xxx.py key1=value1 key2=value2 ...
```

- `key=value`：覆盖默认配置
- `run_train EXP_NAME TRAIN_H ...`：前两个是固定参数，后面是额外覆盖参数
- `run_eval EXP_NAME CKPT EVAL_H ...`：前三个是固定参数，后面是额外覆盖参数

以基线为例：

```bash
run_train base 10
BASE_CKPT="$(find_ckpt base)"
run_eval base "$BASE_CKPT" 10
```

对应含义：
- `base`：实验名
- `10`：历史帧长度（训练或评估）
- `BASE_CKPT`：基线训练产生的 checkpoint 路径

---

## 2. 全局环境与目录根路径

```bash
export HF_ENDPOINT="https://hf-mirror.com"

export NAVSIM_EXP_ROOT="/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim/exp_new"
export NUPLAN_MAPS_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1"
export NAVSIM_DEVKIT_ROOT="/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim"

export PYTHON="/home/gpus-09/miniforge3/envs/navsim3/bin/python"
export HYDRA_FULL_ERROR=1

# State-Frozen 消融必须优先使用本地 FLA
export PYTHONPATH="$NAVSIM_DEVKIT_ROOT/flash-linear-attention:$NAVSIM_DEVKIT_ROOT:$PYTHONPATH"

mkdir -p "$NAVSIM_EXP_ROOT"
```

---

## 3. 统一目录与命名规范

### 3.1 一级目录（都在 `NAVSIM_EXP_ROOT` 下）

- `Dataset_caching/`
- `Train_results/`
- `Evaluation_results/`

### 3.2 二级目录规范

- 数据缓存
  - `Dataset_caching/train_features/h{H}/`
  - `Dataset_caching/train_metric/navtrain/`
  - `Dataset_caching/eval_metric/navtest/`
- 训练结果
  - `Train_results/baseline/`
  - `Train_results/ablation_history/`
  - `Train_results/ablation_memory_encoder/`
  - `Train_results/ablation_memory_decoder/`
  - `Train_results/ablation_backbone_strategy/`
  - `Train_results/ablation_backbone_scale/`
  - `Train_results/ablation_state_frozen/`
- 评估结果
  - `Evaluation_results/baseline/`
  - `Evaluation_results/ablation_history/`
  - `Evaluation_results/ablation_memory_encoder/`
  - `Evaluation_results/ablation_memory_decoder/`
  - `Evaluation_results/ablation_backbone_strategy/`
  - `Evaluation_results/ablation_backbone_scale/`
  - `Evaluation_results/ablation_state_frozen/`

### 3.3 实验命名规范

- 基线：`base`
- 历史长度训练：`hist_t{T}`，例如 `hist_t12`
- 历史长度评估：`hist_t{T}_e{E}`，例如 `hist_t12_e20`
- 历史长度“无限帧”评估预留名：`hist_t{T}_einf`（当前先不跑）
- memory encoder：`mem_enc_{m}k{k}`，例如 `mem_enc_8k4`
- memory decoder：`mem_dec_{m}k{k}`
- backbone 策略：`bb_pool_frozen` / `bb_reg_fullft` 等
- backbone 规模：`bb_vitb` / `bb_vitl`
- state frozen：`sf_lica` / `sf_beta0` / `sf_g0` / `sf_frozen`

---

## 4. 先创建目录（一次）

```bash
mkdir -p "$NAVSIM_EXP_ROOT/Dataset_caching/train_features"
mkdir -p "$NAVSIM_EXP_ROOT/Dataset_caching/train_metric/navtrain"
mkdir -p "$NAVSIM_EXP_ROOT/Dataset_caching/eval_metric/navtest"

mkdir -p "$NAVSIM_EXP_ROOT/Train_results/baseline"
mkdir -p "$NAVSIM_EXP_ROOT/Train_results/ablation_history"
mkdir -p "$NAVSIM_EXP_ROOT/Train_results/ablation_memory_encoder"
mkdir -p "$NAVSIM_EXP_ROOT/Train_results/ablation_memory_decoder"
mkdir -p "$NAVSIM_EXP_ROOT/Train_results/ablation_backbone_strategy"
mkdir -p "$NAVSIM_EXP_ROOT/Train_results/ablation_backbone_scale"
mkdir -p "$NAVSIM_EXP_ROOT/Train_results/ablation_state_frozen"

mkdir -p "$NAVSIM_EXP_ROOT/Evaluation_results/baseline"
mkdir -p "$NAVSIM_EXP_ROOT/Evaluation_results/ablation_history"
mkdir -p "$NAVSIM_EXP_ROOT/Evaluation_results/ablation_memory_encoder"
mkdir -p "$NAVSIM_EXP_ROOT/Evaluation_results/ablation_memory_decoder"
mkdir -p "$NAVSIM_EXP_ROOT/Evaluation_results/ablation_backbone_strategy"
mkdir -p "$NAVSIM_EXP_ROOT/Evaluation_results/ablation_backbone_scale"
mkdir -p "$NAVSIM_EXP_ROOT/Evaluation_results/ablation_state_frozen"
```

---

## 5. 构建缓存（按新目录）

### 5.1 训练特征缓存（多份，按历史帧）

```bash
for H in 4 10 12 20 40; do
  mkdir -p "$NAVSIM_EXP_ROOT/Dataset_caching/train_features/h${H}"
  NAVSIM_EVAL_EXP_ROOT="$NAVSIM_EXP_ROOT/Dataset_caching/train_features" \
  $PYTHON "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py" \
    agent=mom_driver_agent \
    train_test_split=navtrain \
    train_test_split.scene_filter.num_history_frames=$H \
    +agent.config.cam_seq_len=$H \
    cache_path="$NAVSIM_EXP_ROOT/Dataset_caching/train_features/h${H}" \
    experiment_name="cache_train_h${H}"
done
```

### 5.2 训练评分缓存

```bash
$PYTHON "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_train_metric_caching.py" \
  train_test_split=navtrain \
  cache.cache_path="$NAVSIM_EXP_ROOT/Dataset_caching/train_metric/navtrain"
```

### 5.3 适配 MoMAgent 固定训练评分缓存路径

```bash
FIXED_TRAIN_CACHE="/mnt/workspace/nanyi/navsim_workspace/exp/metric_cache_training_new"
mkdir -p "$(dirname "$FIXED_TRAIN_CACHE")"
ln -sfn "$NAVSIM_EXP_ROOT/Dataset_caching/train_metric/navtrain" "$FIXED_TRAIN_CACHE"
```

### 5.4 评测评分缓存

```bash
$PYTHON "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
  train_test_split=navtest \
  cache.cache_path="$NAVSIM_EXP_ROOT/Dataset_caching/eval_metric/navtest"
```

备注：评测评分缓存通常不需要按多帧重建（除非你改了 split/scene_filter/scoring 配置）。

---

## 6. 通用函数（已按新目录改好）

```bash
get_cam_index_list() {
  local H="$1"
  $PYTHON - <<PY
H = int("$1")
print("[" + ",".join(str(i) for i in range(H)) + "]")
PY
}

train_group_dir() {
  local EXP_NAME="$1"
  if [[ "$EXP_NAME" == "base" ]]; then
    echo "baseline"
  elif [[ "$EXP_NAME" == hist_* ]]; then
    echo "ablation_history"
  elif [[ "$EXP_NAME" == mem_enc_* ]]; then
    echo "ablation_memory_encoder"
  elif [[ "$EXP_NAME" == mem_dec_* ]]; then
    echo "ablation_memory_decoder"
  elif [[ "$EXP_NAME" == bb_pool_* || "$EXP_NAME" == bb_dec_* || "$EXP_NAME" == bb_reg_* ]]; then
    echo "ablation_backbone_strategy"
  elif [[ "$EXP_NAME" == bb_vit* ]]; then
    echo "ablation_backbone_scale"
  elif [[ "$EXP_NAME" == sf_* ]]; then
    echo "ablation_state_frozen"
  else
    echo "misc"
  fi
}

eval_group_dir() {
  train_group_dir "$1"
}

run_train() {
  local EXP_NAME="$1"
  local TRAIN_H="$2"
  shift 2

  local CAM_IDX
  CAM_IDX="$(get_cam_index_list "$TRAIN_H")"

  local CACHE_PATH="$NAVSIM_EXP_ROOT/Dataset_caching/train_features/h${TRAIN_H}"
  local GROUP
  GROUP="$(train_group_dir "$EXP_NAME")"
  local TRAIN_ROOT="$NAVSIM_EXP_ROOT/Train_results/$GROUP"

  mkdir -p "$TRAIN_ROOT"

  NAVSIM_EVAL_EXP_ROOT="$TRAIN_ROOT" \
  $PYTHON "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py" \
    agent=mom_driver_agent \
    experiment_name="$EXP_NAME" \
    train_test_split=navtrain \
    cache_path="$CACHE_PATH" \
    use_cache_without_dataset=True \
    train_test_split.scene_filter.num_history_frames="$TRAIN_H" \
    trainer.params.max_epochs=10 \
    dataloader.params.prefetch_factor=1 \
    dataloader.params.batch_size=1 \
    dataloader.params.num_workers=4 \
    agent.lr=0.0002 \
    +agent.config.refiner_ls_values=0.0 \
    +agent.config.one_token_per_traj=true \
    +agent.config.refiner_num_heads=1 \
    +agent.config.tf_d_model=256 \
    +agent.config.tf_d_ffn=1024 \
    +agent.config.ref_num=4 \
    +agent.config.prev_weight=0.0 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    +agent.config.cam_seq_len="$TRAIN_H" \
    +agent.config.cam_f0="$CAM_IDX" \
    +agent.config.cam_b0="$CAM_IDX" \
    +agent.config.cam_l0="$CAM_IDX" \
    +agent.config.cam_r0="$CAM_IDX" \
    seed=2 \
    "$@"
}

find_ckpt() {
  local EXP_NAME="$1"
  local GROUP
  GROUP="$(train_group_dir "$EXP_NAME")"
  ls -1t "$NAVSIM_EXP_ROOT/Train_results/$GROUP/$EXP_NAME"/*/lightning_logs/version_0/checkpoints/*.ckpt | head -n 1
}

run_eval() {
  local EXP_NAME="$1"
  local CKPT="$2"
  local EVAL_H="$3"
  shift 3

  local CAM_IDX
  CAM_IDX="$(get_cam_index_list "$EVAL_H")"

  local GROUP
  GROUP="$(eval_group_dir "$EXP_NAME")"
  local EVAL_ROOT="$NAVSIM_EXP_ROOT/Evaluation_results/$GROUP"

  mkdir -p "$EVAL_ROOT"

  NAVSIM_EVAL_EXP_ROOT="$EVAL_ROOT" \
  $PYTHON "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_rwkv.py" \
    train_test_split=navtest \
    worker=ray_distributed \
    metric_cache_path="/mnt/workspace/nanyi/navsim_workspace/exp/metric_cache" \
    agent=mom_driver_agent \
    "agent.checkpoint_path='$CKPT'" \
    train_test_split.scene_filter.num_history_frames="$EVAL_H" \
    +agent.config.cam_seq_len="$EVAL_H" \
    +agent.config.cam_f0="$CAM_IDX" \
    +agent.config.cam_b0="$CAM_IDX" \
    +agent.config.cam_l0="$CAM_IDX" \
    +agent.config.cam_r0="$CAM_IDX" \
    enable_padding=true \
    experiment_name="eval_$EXP_NAME" \
    "$@"
}
```
    # metric_cache_path="$NAVSIM_EXP_ROOT/Dataset_caching/eval_metric/navtest" \
---

## 7. 基线实验

```bash
run_train base 10
BASE_CKPT="$(find_ckpt base)"
run_eval base "$BASE_CKPT" 10
```

---

## 8. 消融 A：历史长度（Train x Eval）

```bash
for T in 4 12 20; do
  run_train "hist_t${T}" "$T"
done

for T in 4 12 20; do
  CKPT="$(find_ckpt hist_t${T})"
  for E in 4 8 12 20; do
    NAME="hist_t${T}_e${E}"
    run_eval "$NAME" "$CKPT" "$E"
  done
done
```

无限帧评测先留空（后续改完代码再补）：

```bash
# TODO:
```

---

## 9. 消融 B：Memory Slots

```bash
# Encoder
run_train mem_enc_1k1 10 +agent.config.mom_temporal.num_memories=1 +agent.config.mom_temporal.topk=1
run_train mem_enc_2k1 10 +agent.config.mom_temporal.num_memories=2 +agent.config.mom_temporal.topk=1
run_train mem_enc_2k2 10 +agent.config.mom_temporal.num_memories=2 +agent.config.mom_temporal.topk=2
run_train mem_enc_8k2 10 +agent.config.mom_temporal.num_memories=8 +agent.config.mom_temporal.topk=2
run_train mem_enc_8k4 10 +agent.config.mom_temporal.num_memories=8 +agent.config.mom_temporal.topk=4

for EXP in mem_enc_1k1 mem_enc_2k1 mem_enc_2k2 mem_enc_8k2 mem_enc_8k4; do
  CKPT="$(find_ckpt "$EXP")"
  run_eval "$EXP" "$CKPT" 10
done

# Decoder
run_train mem_dec_1k1 10 +agent.config.mom_decoder.num_memories=1 +agent.config.mom_decoder.topk=1
run_train mem_dec_2k1 10 +agent.config.mom_decoder.num_memories=2 +agent.config.mom_decoder.topk=1
run_train mem_dec_2k2 10 +agent.config.mom_decoder.num_memories=2 +agent.config.mom_decoder.topk=2
run_train mem_dec_8k2 10 +agent.config.mom_decoder.num_memories=8 +agent.config.mom_decoder.topk=2
run_train mem_dec_8k4 10 +agent.config.mom_decoder.num_memories=8 +agent.config.mom_decoder.topk=4

for EXP in mem_dec_1k1 mem_dec_2k1 mem_dec_2k2 mem_dec_8k2 mem_dec_8k4; do
  CKPT="$(find_ckpt "$EXP")"
  run_eval "$EXP" "$CKPT" 10
done
```

---

## 10. 消融 C/D/E（主干策略、主干规模、State-Frozen）

```bash
# C: 主干策略
run_train bb_pool_frozen 10 +agent.config.num_scene_tokens=16 +agent.config.image_backbone_use_feature_pooling=true +agent.config.image_backbone_use_lora=false +agent.config.image_backbone_finetune=false
run_train bb_pool_lora   10 +agent.config.num_scene_tokens=16 +agent.config.image_backbone_use_feature_pooling=true +agent.config.image_backbone_use_lora=true +agent.config.image_backbone_lora_rank=32
run_train bb_dec_frozen  10 +agent.config.num_scene_tokens=16 +agent.config.image_backbone_focus_front_cam=true +agent.config.image_backbone_compress_fc=true +agent.config.image_backbone_use_lora=false +agent.config.image_backbone_finetune=false
run_train bb_dec_lora    10 +agent.config.num_scene_tokens=16 +agent.config.image_backbone_focus_front_cam=true +agent.config.image_backbone_compress_fc=true +agent.config.image_backbone_use_lora=true
run_train bb_reg_fullft  10 +agent.config.num_scene_tokens=16 +agent.config.image_backbone_use_feature_pooling=false +agent.config.image_backbone_compress_fc=false +agent.config.image_backbone_focus_front_cam=false +agent.config.image_backbone_use_lora=false +agent.config.image_backbone_finetune=true
run_train bb_reg_frozen  10 +agent.config.num_scene_tokens=16 +agent.config.image_backbone_use_feature_pooling=false +agent.config.image_backbone_compress_fc=false +agent.config.image_backbone_focus_front_cam=false +agent.config.image_backbone_use_lora=false +agent.config.image_backbone_finetune=false

for EXP in bb_pool_frozen bb_pool_lora bb_dec_frozen bb_dec_lora bb_reg_fullft bb_reg_frozen; do
  CKPT="$(find_ckpt "$EXP")"
  run_eval "$EXP" "$CKPT" 10
done

# D: 主干规模
run_train bb_vitb 10 +agent.config.image_backbone_model_name=timm/vit_base_patch14_reg4_dinov2.lvd142m  agent.config.image_backbone_model_weights=weights/vit_base_patch14_reg4_dinov2.lvd142m/model.safetensors
run_train bb_vitl 10 +agent.config.image_backbone_model_name=timm/vit_large_patch14_reg4_dinov2.lvd142m agent.config.image_backbone_model_weights=weights/vit_large_patch14_reg4_dinov2.lvd142m/model.safetensors

for EXP in bb_vitb bb_vitl; do
  CKPT="$(find_ckpt "$EXP")"
  run_eval "$EXP" "$CKPT" 10
done

# E: State-Frozen
run_train sf_lica   10 +agent.config.use_state_frozen=false +agent.config.freeze_beta_only=false +agent.config.freeze_g_only=false
run_train sf_beta0  10 +agent.config.use_state_frozen=false +agent.config.freeze_beta_only=true  +agent.config.freeze_g_only=false
run_train sf_g0     10 +agent.config.use_state_frozen=false +agent.config.freeze_beta_only=false +agent.config.freeze_g_only=true
run_train sf_frozen 10 +agent.config.use_state_frozen=true  +agent.config.freeze_beta_only=false +agent.config.freeze_g_only=false

for EXP in sf_lica sf_beta0 sf_g0 sf_frozen; do
  CKPT="$(find_ckpt "$EXP")"
  run_eval "$EXP" "$CKPT" 10
done
```

---

## 11. 推荐执行顺序

1. 先构建第 5 节全部缓存。
2. 运行第 7 节基线，验证链路。
3. 跑第 8 节历史长度消融（最关键）。
4. 跑第 9 节 memory slots。
5. 跑第 10 节 state-frozen 与 backbone 消融。

---

## 12. 实验目录结构与说明（最终规范）

建议最终目录结构如下：

```text
${NAVSIM_EXP_ROOT}/
├── Dataset_caching/
│   ├── train_features/
│   │   ├── h4/
│   │   ├── h10/
│   │   ├── h12/
│   │   ├── h20/
│   │   └── h40/
│   ├── train_metric/
│   │   └── navtrain/
│   └── eval_metric/
│       └── navtest/
├── Train_results/
│   ├── baseline/
│   │   └── base/
│   ├── ablation_history/
│   │   ├── hist_t4/
│   │   ├── hist_t12/
│   │   └── hist_t20/
│   ├── ablation_memory_encoder/
│   │   ├── mem_enc_1k1/
│   │   ├── mem_enc_2k1/
│   │   ├── mem_enc_2k2/
│   │   ├── mem_enc_8k2/
│   │   └── mem_enc_8k4/
│   ├── ablation_memory_decoder/
│   │   ├── mem_dec_1k1/
│   │   ├── mem_dec_2k1/
│   │   ├── mem_dec_2k2/
│   │   ├── mem_dec_8k2/
│   │   └── mem_dec_8k4/
│   ├── ablation_backbone_strategy/
│   │   ├── bb_pool_frozen/
│   │   ├── bb_pool_lora/
│   │   ├── bb_dec_frozen/
│   │   ├── bb_dec_lora/
│   │   ├── bb_reg_fullft/
│   │   └── bb_reg_frozen/
│   ├── ablation_backbone_scale/
│   │   ├── bb_vitb/
│   │   └── bb_vitl/
│   └── ablation_state_frozen/
│       ├── sf_lica/
│       ├── sf_beta0/
│       ├── sf_g0/
│       └── sf_frozen/
└── Evaluation_results/
    ├── baseline/
    │   └── eval_base/
    ├── ablation_history/
    │   ├── eval_hist_t4_e4/
    │   ├── eval_hist_t4_e8/
    │   ├── eval_hist_t4_e12/
    │   ├── eval_hist_t4_e20/
    │   ├── eval_hist_t12_e4/
    │   ├── eval_hist_t12_e8/
    │   ├── eval_hist_t12_e12/
    │   ├── eval_hist_t12_e20/
    │   ├── eval_hist_t20_e4/
    │   ├── eval_hist_t20_e8/
    │   ├── eval_hist_t20_e12/
    │   ├── eval_hist_t20_e20/
    │   ├── eval_hist_t4_einf/      # 预留，当前不跑
    │   ├── eval_hist_t12_einf/     # 预留，当前不跑
    │   └── eval_hist_t20_einf/     # 预留，当前不跑
    ├── ablation_memory_encoder/
    │   ├── eval_mem_enc_1k1/
    │   ├── eval_mem_enc_2k1/
    │   ├── eval_mem_enc_2k2/
    │   ├── eval_mem_enc_8k2/
    │   └── eval_mem_enc_8k4/
    ├── ablation_memory_decoder/
    │   ├── eval_mem_dec_1k1/
    │   ├── eval_mem_dec_2k1/
    │   ├── eval_mem_dec_2k2/
    │   ├── eval_mem_dec_8k2/
    │   └── eval_mem_dec_8k4/
    ├── ablation_backbone_strategy/
    │   ├── eval_bb_pool_frozen/
    │   ├── eval_bb_pool_lora/
    │   ├── eval_bb_dec_frozen/
    │   ├── eval_bb_dec_lora/
    │   ├── eval_bb_reg_fullft/
    │   └── eval_bb_reg_frozen/
    ├── ablation_backbone_scale/
    │   ├── eval_bb_vitb/
    │   └── eval_bb_vitl/
    └── ablation_state_frozen/
        ├── eval_sf_lica/
        ├── eval_sf_beta0/
        ├── eval_sf_g0/
        └── eval_sf_frozen/
```

解释：
- `Dataset_caching`：只放缓存，不放训练/评估输出
- `Train_results`：每个实验名对应一个目录，里面有 hydra 配置、日志、checkpoint
- `Evaluation_results`：每次评测输出 csv，目录名与实验名一一对应
- `einf` 目录先作为命名预留，待你改完无限帧代码后再实际生成
- 这种结构便于后期论文对表、复现实验、批量汇总脚本读取
