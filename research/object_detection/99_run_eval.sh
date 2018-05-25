#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"

DATA_HOME='/mnt/hard_data/Data/foods'

python eval.py \
    --logtostderr \
    --pipeline_config_path=$DATA_HOME/models/model/food_faster_rcnn_inception_v2.config \
    --checkpoint_dir=$DATA_HOME/models/model/train \
    --eval_dir=$DATA_HOME/models/model/eval

