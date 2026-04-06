export NAVSIM_EXP_ROOT="/mnt/workspace/nanyi/navsim_workspace/exp"
export NUPLAN_MAPS_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export OPENSCENE_DATA_ROOT="/mnt/pai-pdc-nas/nanyi/openscene-v1.1"
export NAVSIM_DEVKIT_ROOT="/mnt/workspace/jihao/RWKV-navsim"
export MODEL_TYPE="v2"

TEAM_NAME="wmy_test"
AUTHORS="xiaoming"
EMAIL="giggle98@163.com"
INSTITUTION="ZJU"
COUNTRY="China"
CHECKPOINT=$NAVSIM_DEVKIT_ROOT/base_line/best.ckpt

TRAIN_TEST_SPLIT=navtest

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_create_submission_pickle.py \
train_test_split=$TRAIN_TEST_SPLIT \
"train_test_split.scene_filter.num_history_frames"=10 \
agent=rwkv7_mf_agent \
agent.checkpoint_path=$CHECKPOINT \
experiment_name=submission_rwkv_agent \
enable_padding=true \
team_name=$TEAM_NAME \
authors=$AUTHORS \
email=$EMAIL \
institution=$INSTITUTION \
country=$COUNTRY \
