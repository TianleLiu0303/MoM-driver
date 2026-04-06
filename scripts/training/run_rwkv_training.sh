TRAIN_TEST_SPLIT=navtrain

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_training.py \
agent=rwkv_agent \
experiment_name=training_rwkv_agent \
train_test_split=$TRAIN_TEST_SPLIT