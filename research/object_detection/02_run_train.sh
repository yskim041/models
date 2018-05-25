#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"

# DATA_HOME='/mnt/hard_data/Data/foods'
DATA_HOME='/mnt/hard_data/Data/rcnn/demo_data'

python train.py \
    --logtostderr \
    --pipeline_config_path=$DATA_HOME/models/model/demo_faster_rcnn_resnet101.config \
    --train_dir=$DATA_HOME/models/model/train

