#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"

# DATA_HOME='/mnt/hard_data/Data/foods'
DATA_HOME='/mnt/hard_data/Data/rcnn/demo_data'

python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path $DATA_HOME/models/model/demo_faster_rcnn_resnet101.config \
    --trained_checkpoint_prefix $DATA_HOME/models/model/train/model.ckpt-5310 \
    --output_directory $DATA_HOME/graph

