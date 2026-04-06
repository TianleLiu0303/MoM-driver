# Training Notes (MoM_driver)

This file summarizes the current training setup and key paths used in this repo.

## What We Ran

- Agent: `mom_driver_agent`
- GPU: `CUDA_VISIBLE_DEVICES=1`
- Epochs: 50
- Batch size: 4
- Learning rate: 1e-4
- Precision: 16-mixed
- Cache usage: feature cache + metric cache

## Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=1
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1/maps"
export NAVSIM_EXP_ROOT="/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim/exp"
export NAVSIM_DEVKIT_ROOT="/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim"
export OPENSCENE_DATA_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1"
```

## Training Command (Reference)

```bash
python navsim/planning/script/run_training.py \
    agent=mom_driver_agent \
    experiment_name=mom_training_gpu1_v2 \
    train_test_split=navtrain \
    cache_path=/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim/exp/training_cache \
    use_cache_without_dataset=true \
    force_cache_computation=false \
    trainer.params.max_epochs=50 \
    trainer.params.accelerator=gpu \
    +trainer.params.devices=1 \
    trainer.params.strategy=auto \
    trainer.params.precision=16-mixed \
    dataloader.params.batch_size=4 \
    dataloader.params.num_workers=2 \
    agent.lr=1e-4 \
    seed=42
```

## Cache Locations

- Feature cache: `exp/training_cache/`
  - Train tokens: `exp/training_cache/debug/train_cache_paths.pkl`
  - Val tokens: `exp/training_cache/debug/val_cache_paths.pkl`
- Metric cache: `exp/train_metric_cache/metadata/cache_metadata.csv`

## Outputs

- Training log: `training_mom_final_v4.log`
- Checkpoints: `exp/mom_training_gpu1_v2/2026.03.06.00.25.43/lightning_logs/version_0/checkpoints/`
- Latest checkpoint (example): `epoch=49-step=750.ckpt`

## Notes

- 60 training samples with batch size 4 means 15 batches per epoch.
- Metric caching is for PDM scorer loss and is required even when using feature caches.

## DrivoR Comparison (Training)

- Cache usage: DrivoR does NOT use feature cache (`cache_path=null`, `use_cache_without_dataset=false`), but DOES generate metric cache with `run_train_metric_caching.py` for PDM scorer loss.
- Data scale: DrivoR trains on full navtrain (85k+ samples), we used 60 train / 16 val cached samples.
- GPU: DrivoR uses 4 GPUs (`agent.num_gpus=4`), we used single GPU.
- Batch size: DrivoR uses 16, we used 4.
- Optimizer: DrivoR uses AdamW with `base_lr=2e-4`, we used `lr=1e-4` (default optimizer).
- Epochs: DrivoR 10–25 epochs, we used 50 epochs.

### DrivoR Commands (Reference)

Metric cache (PDM cache):

```bash
python navsim/planning/script/run_train_metric_caching.py
```

NAVSIM-v1 training (DrivoR README):

```bash
export HYDRA_FULL_ERROR=1
EXPERIMENT=training_drivoR_Nav1_traj_long_25epochs
AGENT=drivoR
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training_full.py \
    agent=$AGENT \
    experiment_name=$EXPERIMENT \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.max_epochs=25 \
    dataloader.params.prefetch_factor=1 \
    dataloader.params.batch_size=16 \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.num_gpus=4 \
    agent.progress_bar=false \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.loss.prev_weight=0.0 \
    agent.config.long_trajectory_additional_poses=2 \
    seed=2
```

NAVSIM-v2 training (DrivoR README):

```bash
export HYDRA_FULL_ERROR=1
EXPERIMENT=training_drivoR_Nav2_10epochs
AGENT=drivoR
python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
    agent=$AGENT \
    experiment_name=$EXPERIMENT \
    train_test_split=navtrain \
    cache_path=null \
    use_cache_without_dataset=false \
    trainer.params.max_epochs=10 \
    dataloader.params.prefetch_factor=1 \
    dataloader.params.batch_size=16 \
    agent.lr_args.name=AdamW \
    agent.lr_args.base_lr=0.0002 \
    agent.num_gpus=4 \
    agent.progress_bar=false \
    agent.config.refiner_ls_values=0.0 \
    agent.config.image_backbone.focus_front_cam=false \
    agent.config.one_token_per_traj=true \
    agent.config.refiner_num_heads=1 \
    agent.config.tf_d_model=256 \
    agent.config.tf_d_ffn=1024 \
    agent.config.area_pred=false \
    agent.config.agent_pred=false \
    agent.config.ref_num=4 \
    agent.loss.prev_weight=0.0 \
    seed=2
```

### Side-by-Side Summary

| Item | MoM_driver | DrivoR |
| --- | --- | --- |
| Feature cache | `use_cache_without_dataset=true` | `cache_path=null` + `use_cache_without_dataset=false` |
| Metric cache | `cache_tokens.py` | `run_train_metric_caching.py` |
| GPUs | 1 | 4 |
| Batch size | 4 | 16 |
| Epochs | 50 | 10–25 |
| Optimizer | Adam (default) | AdamW |
| LR | 1e-4 | 2e-4 |
