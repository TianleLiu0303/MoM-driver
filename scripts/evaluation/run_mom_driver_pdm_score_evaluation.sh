export HF_ENDPOINT=https://hf-mirror.com

export NAVSIM_EXP_ROOT="/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim/exp"
export NUPLAN_MAPS_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1"
export NAVSIM_DEVKIT_ROOT="/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim"

TRAIN_TEST_SPLIT=navtest
CHECKPOINT=$NAVSIM_DEVKIT_ROOT/exp/mom_train/2026.03.19.18.57.57/lightning_logs/version_0/checkpoints/epoch_driver.ckpt
CACHE_PATH="/mnt/workspace/nanyi/navsim_workspace/exp/metric_cache"

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_rwkv.py \
train_test_split=$TRAIN_TEST_SPLIT \
worker=ray_distributed \
metric_cache_path=$CACHE_PATH \
agent=mom_driver_agent \
agent.checkpoint_path=$CHECKPOINT \
"train_test_split.scene_filter.num_history_frames"=10 \
enable_padding=true \
experiment_name=evaluating_mom_driver_agent_0320


