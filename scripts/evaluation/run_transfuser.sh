export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1/maps"
export NAVSIM_EXP_ROOT="/mnt/pai-pdc-nas/tianle_DPR/workspace/exp"
export NAVSIM_DEVKIT_ROOT="/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim"
export OPENSCENE_DATA_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1"
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

export PYTHONPATH=/mnt/pai-pdc-nas/tianle_DPR/RWKV-navsim:$PYTHONPATH

TRAIN_TEST_SPLIT=navtest
CHECKPOINT="/mnt/pai-pdc-nas/tianle_DPR/workspace/navsim/ckpts/transfuser_seed_2.ckpt"
CACHE_PATH="/mnt/pai-pdc-nas/tianle_DPR/workspace/exp/metric_cache"
# SYNTHETIC_SENSOR_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/sensor_blobs
# SYNTHETIC_SCENES_PATH=$OPENSCENE_DATA_ROOT/navhard_two_stage/synthetic_scene_pickles


python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_rwkv.py \
train_test_split=$TRAIN_TEST_SPLIT \
agent=transfuser_agent \
worker=ray_distributed \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=transfuser_agent \
metric_cache_path=$CACHE_PATH \
# synthetic_sensor_path=$SYNTHETIC_SENSOR_PATH \
# synthetic_scenes_path=$SYNTHETIC_SCENES_PATH \
