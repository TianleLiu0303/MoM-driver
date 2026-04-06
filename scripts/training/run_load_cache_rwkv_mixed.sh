TRAIN_TEST_SPLIT=navtrain
CACHE_PATH=$NAVSIM_EXP_ROOT/training_cache_mixed

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
agent=transfuser_mf_agent \
experiment_name=cache_exp_log \
train_test_split=$TRAIN_TEST_SPLIT \
cache_path=$CACHE_PATH \