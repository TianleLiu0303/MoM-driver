TRAIN_TEST_SPLIT=navtrain
CACHE_PATH=$NAVSIM_EXP_ROOT/training_cache_multi

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=transfuser_mf_agent \
experiment_name=training_transfuser_agent_multi \
"dataloader.params.batch_size"=16 \
train_test_split=$TRAIN_TEST_SPLIT \
cache_path=$CACHE_PATH \