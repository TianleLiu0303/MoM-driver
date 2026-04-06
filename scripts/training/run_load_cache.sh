TRAIN_TEST_SPLIT=navtrain
CACHE_PATH=$NAVSIM_EXP_ROOT/training_cache_multi_4

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching.py \
agent=transfuser_mf_agent \
experiment_name=training_transfuser_agent_multi \
train_test_split=$TRAIN_TEST_SPLIT \
cache_path=$CACHE_PATH