export HF_ENDPOINT=https://hf-mirror.com

export NAVSIM_EXP_ROOT="/mnt/workspace/nanyi/navsim_workspace/exp"
export NUPLAN_MAPS_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1"
export NAVSIM_DEVKIT_ROOT="/mnt/workspace/jihao/RWKV-navsim"
export MODEL_TYPE="v2"

TRAIN_TEST_SPLIT=navtest
CHECKPOINT=$NAVSIM_DEVKIT_ROOT/base_line/rwkv_navtrain.ckpt
CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_rwkv.py \
train_test_split=$TRAIN_TEST_SPLIT \
worker=ray_distributed \
metric_cache_path=$CACHE_PATH \
agent=rwkv7_mf_agent \
agent.checkpoint_path=$CHECKPOINT \
"train_test_split.scene_filter.num_history_frames"=10 \
enable_padding=true \
experiment_name=evaluating_rwkv_mf_agent_0920_new_diffusion


