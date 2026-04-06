TRAIN_TEST_SPLIT=navtrain
CACHE_PATH=$NAVSIM_EXP_ROOT/training_cache_mixed

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=rwkv7_mixed_agent \
experiment_name=training_rwkv_mixed_agent \
"dataloader.params.batch_size"=8 \
"dataloader.params.num_workers"=0 \
"dataloader.params.prefetch_factor"=null \
train_test_split=$TRAIN_TEST_SPLIT \
"train_test_split.scene_filter.num_history_frames"=10 \
cache_path=$CACHE_PATH \
use_cache_without_dataset=true \