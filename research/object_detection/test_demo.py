#!/usr/bin/env python

from __future__ import division

import numpy as np
import os
import sys
import tensorflow as tf

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed to display the images.
# %matplotlib inline

# This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("..")

from utils import label_map_util

from utils import visualization_utils as vis_util

MODEL_BASE_DIR = '/mnt/hard_data/Data/foods/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(MODEL_BASE_DIR, 'graph', 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(MODEL_BASE_DIR, 'data', 'food_label_map.pbtxt')

NUM_CLASSES = 47

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
#TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'apple_%d.jpg' % i) for i in range(1, 6) ]
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'test_%d.jpg' % i) for i in range(1, 6) ]
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


tf_config = tf.ConfigProto()
tf_config.gpu_options.visible_device_list = '1'
tf_config.gpu_options.allow_growth = False

with detection_graph.as_default():
  with tf.Session(graph=detection_graph, config=tf_config) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    plt.figure(figsize=IMAGE_SIZE)
    rst_idx = 0

    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})

      threshold = 0.7
      valid_scores = [score for score in scores[0] if score > threshold]
      midpoints = []

      for val in range(0, len(valid_scores)):
        bounding_box = boxes[0][val]
        # [ymin, xmin, ymax, xmax]
        midpoint_y = (bounding_box[0] + bounding_box[2]) / 2.0  * image_np.shape[0]
        midpoint_x = (bounding_box[1] + bounding_box[3]) / 2.0  * image_np.shape[1]

        # import IPython; IPython.embed()

        image_np[int(midpoint_y) - 5: int(midpoint_y) + 5,
                 int(midpoint_x) - 5:int(midpoint_x) + 5, :3] = [0, 0, 255.0]
        midpoints.append([midpoint_x, midpoint_y])

      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=len(valid_scores),
          min_score_thresh=threshold,
          line_thickness=4)

      plt.clf()
      plt.imshow(image_np)
      plt.tight_layout()
      plt.savefig('rst_%03d.png' % rst_idx)
      rst_idx += 1

plt.close()


# End of script


