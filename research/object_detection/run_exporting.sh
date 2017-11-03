#!/bin/bash

DEMO_DATA_HOME='/home/yskim/Data/demo/rcnn/demo_data'

python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path $DEMO_DATA_HOME/models/model/demo_faster_rcnn_resnet101.config \
    --trained_checkpoint_prefix $DEMO_DATA_HOME/models/model/train/model.ckpt-14214 \
    --output_directory $DEMO_DATA_HOME/graph

