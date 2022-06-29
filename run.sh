#!/bin/bash

#DATASET_DIR='../../../data_experiment/'
#WORKSPACE='../../../experiment_workspace/baseline_cnn/'
DATASET_DIR='../../dataset1/'
WORKSPACE='../../experiment_workspace/Conformer-S9/'

DEV_DIR='models_dev'
TEST_DIR='models_test'

MODEL_TYPE='Conformer'
AUDIO_NUM=3
AUDIO_1='breathing-deep'
AUDIO_2='counting-normal'
AUDIO_3='vowel-o'
AUDIO_4='vowel-e'
ITERATION_MAX=30000

############ Development ############
# Train model
python3 main_pytorch.py train --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda --model_type=$MODEL_TYPE --audio_num=$AUDIO_NUM --audio_1=$AUDIO_1 --audio_2=$AUDIO_2 --audio_3=$AUDIO_3 --audio_4=$AUDIO_4


# Evaluate
python3 main_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$DEV_DIR --workspace=$WORKSPACE --validate --iteration=$ITERATION_MAX --cuda --model_type=$MODEL_TYPE --audio_num=$AUDIO_NUM --audio_1=$AUDIO_1 --audio_2=$AUDIO_2 --audio_3=$AUDIO_3 --audio_4=$AUDIO_4

############ Test ############
# Train model
#python3 main_pytorch.py train --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --cuda --model_type=$MODEL_TYPE --audio_num=$AUDIO_NUM --audio_1=$AUDIO_1 --audio_2=$AUDIO_2 --audio_3=$AUDIO_3 --audio_4=$AUDIO_4

# Evaluate
#python3 main_pytorch.py inference_validation_data --dataset_dir=$DATASET_DIR --subdir=$TEST_DIR --workspace=$WORKSPACE --iteration=$ITERATION_MAX --cuda --model_type=$MODEL_TYPE --audio_num=$AUDIO_NUM --audio_1=$AUDIO_1 --audio_2=$AUDIO_2 --audio_3=$AUDIO_3 --audio_4=$AUDIO_4
