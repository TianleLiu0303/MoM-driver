export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0,1,2,3

export NAVSIM_EVAL_EXP_ROOT="/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim/exp"
export NAVSIM_EXP_ROOT="/mnt/workspace/nanyi/navsim_workspace/exp"
export NUPLAN_MAPS_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1"
export NAVSIM_DEVKIT_ROOT="/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim"

TRAIN_TEST_SPLIT=navtest
CHECKPOINT=/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim/ckpts/mom_epoch_14.ckpt
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_rwkv.py \
train_test_split=$TRAIN_TEST_SPLIT \
worker=ray_distributed \
metric_cache_path=$CACHE_PATH \
agent=mom_agent \
+agent.config.shared_mem=false \
agent.checkpoint_path=$CHECKPOINT \
"train_test_split.scene_filter.num_history_frames"=8 \
enable_padding=true \
experiment_name=evaluating_mom_agent


