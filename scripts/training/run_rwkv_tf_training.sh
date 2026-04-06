TRAIN_TEST_SPLIT=navtrain
CACHE_PATH=$NAVSIM_EXP_ROOT/training_cache_multi_10

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=rwkv7_tfdecoder_agent \
experiment_name=training_rwkv_agent_multi_10 \
"dataloader.params.batch_size"=8 \
"dataloader.params.num_workers"=4 \
train_test_split=$TRAIN_TEST_SPLIT \
"train_test_split.scene_filter.num_history_frames"=10 \
cache_path=$CACHE_PATH \
use_cache_without_dataset=true \