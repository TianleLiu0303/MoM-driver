export HF_ENDPOINT=https://hf-mirror.com

export NAVSIM_EVAL_EXP_ROOT="/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim/exp"
export NAVSIM_EXP_ROOT="/mnt/workspace/nanyi/navsim_workspace/exp"
export NUPLAN_MAPS_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1"
export NAVSIM_DEVKIT_ROOT="/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim"

TRAIN_TEST_SPLIT=navtrain
CACHE_PATH=$NAVSIM_EXP_ROOT/training_cache_multi_10_scorer

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=mom_agent \
experiment_name=training_mom_agent \
+agent.config.shared_mem=false \
"dataloader.params.batch_size"=4 \
"dataloader.params.num_workers"=16 \
"train_test_split.scene_filter.num_history_frames"=10 \
"trainer.params.strategy"=ddp_find_unused_parameters_true \
train_test_split=$TRAIN_TEST_SPLIT \
cache_path=$CACHE_PATH \
use_cache_without_dataset=true \
