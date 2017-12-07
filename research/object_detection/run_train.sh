#!/bin/bash

DEMO_DATA_HOME=${HOME}/Data/rcnn/demo_data

python train.py \
    --logtostderr \
    --pipeline_config_path=$DEMO_DATA_HOME/models/model/demo_faster_rcnn_resnet101.config \
    --train_dir=$DEMO_DATA_HOME/models/model/train

