export HF_ENDPOINT=https://hf-mirror.com

export NAVSIM_EXP_ROOT="/mnt/workspace/nanyi/navsim_workspace/exp"
export NUPLAN_MAPS_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1"
export NAVSIM_DEVKIT_ROOT="/mnt/workspace/jihao/RWKV-navsim"

TRAIN_TEST_SPLIT=navtrain
CACHE_PATH=$NAVSIM_EXP_ROOT/training_cache_multi_10_past_status

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
agent=transfuser_mf_agent \
experiment_name=training_past_status_test \
train_test_split=$TRAIN_TEST_SPLIT \
enable_padding=True \
cache_path=$CACHE_PATH \
"train_test_split.scene_filter.num_history_frames"=10