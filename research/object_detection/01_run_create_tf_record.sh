#!/bin/bash

export CUDA_VISIBLE_DEVICES="1"

# DATA_HOME='/mnt/hard_data/Data/foods'
DATA_HOME='/mnt/hard_data/Data/rcnn/demo_data'

python create_food_tf_record.py \
    --data_dir=$DATA_HOME/data/ \
    --output_dir=$DATA_HOME/data/ \
    --label_map_path=$DATA_HOME/data/demo_label_map.pbtxt

