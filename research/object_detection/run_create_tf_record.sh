#!/bin/bash

DEMO_DATA_HOME='/home/yskim/Data/demo/rcnn/demo_data'

python create_demo_tf_record.py \
    --data_dir=$DEMO_DATA_HOME/data/ \
    --output_dir=$DEMO_DATA_HOME/data/ \
    --label_map_path=$DEMO_DATA_HOME/data/demo_label_map.pbtxt

