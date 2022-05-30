#!/bin/bash

#DATASET_DIR='../../../data_experiment/'
#WORKSPACE='../../../experiment_workspace/baseline_cnn/'
DATASET_DIR='../../dataset1/'
WORKSPACE='../../experiment_workspace/Conformer-S6/'

DEV_DIR='models_dev'
TEST_DIR='models_test'

ITERATION_MAX=30000

############ Development ############
# Train model
python3 main_pytorch.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda


# Evaluate
python3 main_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda

############ Test ############
# Train model
#python3 main_pytorch.py train --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --cuda

# Evaluate
#python3 main_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --cuda
