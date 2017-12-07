#!/bin/bash

DEMO_DATA_HOME=${HOME}/Data/rcnn/demo_data

python eval.py \
    --logtostderr \
    --pipeline_config_path=$DEMO_DATA_HOME/models/model/demo_faster_rcnn_resnet101.config \
    --checkpoint_dir=$DEMO_DATA_HOME/models/model/train \
    --eval_dir=$DEMO_DATA_HOME/models/model/eval

