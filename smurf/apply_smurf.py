# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Produces videos of flow predictions from a directory of ordered images.

Run with a directory of images using apply_smurf \
  --data_dir=<directory with images> \
  --plot_dir=<directory to output results> \
  --checkpoint_dir=<directory to restore model from>
"""

from absl import app
from absl import flags

# pylint:disable=g-bad-import-order
import os
import sys
import gin

import tensorflow as tf
import numpy as np

import smurf_flags  # pylint:disable=unused-import
import smurf_plotting
import smurf_evaluator
from smurf_plotting import _FLOW_SCALING_FACTOR, flow_to_rgb


from src import data
import output
import output_640_480
import pre_trained_models
from utils.data_viz import display_flow
from utils.file_io import save_flow_image
import flowiz as fz


try:
  import cv2  # pylint:disable=g-import-not-at-top
except:  # pylint:disable=bare-except
  print('Missing cv2 dependency. Please install opencv-python.')

MODEL_DIR = os.path.join(os.path.dirname(pre_trained_models.__file__), "kitti-smurf")

DATA_PATH = os.path.dirname(data.__file__)
FRAMES_DIR = os.path.join(DATA_PATH, "frames")
#OUTPUT_DIR = os.path.dirname(output.__file__)
OUTPUT_DIR = os.path.dirname(output_640_480.__file__)


flags.DEFINE_string('data_dir', '', 'Directory with images to run on. Images '
                    'should be named numerically, e.g., 1.png, 2.png.')

FLAGS = flags.FLAGS


def get_image_iterator(image_dir):
  """Iterate through images in the image_dir."""
  
  images = os.listdir(image_dir)
  images = sorted(images)
  images = [os.path.join(image_dir, i) for i in images]
    
  images = zip(images[:-1], images[1:])
  for image1, image2 in images:
    image1 = cv2.imread(image1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1 = tf.image.convert_image_dtype(image1, tf.float32)
    image2 = cv2.imread(image2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image2 = tf.image.convert_image_dtype(image2, tf.float32)
    yield (image1, image2)

def main(unused_argv):
  gin.parse_config_files_and_bindings(FLAGS.config_file, FLAGS.gin_bindings)
  smurf = smurf_evaluator.build_network(batch_size=1)
  smurf.update_checkpoint_dir(MODEL_DIR)
  smurf.restore()
  
  image_dirs = []
  for parent_dir, _, images in os.walk(FRAMES_DIR):
    if len(images) == 0:
      pass
    else:
      image_dirs.append(parent_dir)
  
  image_dirs = sorted(image_dirs)
  
  for dir in image_dirs:
    for idx, (image1, image2) in enumerate(get_image_iterator(dir)):
      sys.stdout.write(':')
      sys.stdout.flush()
      flow_forward, occlusion, flow_backward = smurf.infer(
          image1, image2, input_height=FLAGS.height, input_width=FLAGS.width,
          infer_occlusion=True, infer_bw=True)
      occlusion = 1. - occlusion
      
      
      #image1_arr = image1.numpy()
      #flow_as_image = flow_to_rgb(flow_forward[:, :, ::-1]).numpy()
      #display_flow(-flow_forward.numpy(), flow_as_image, image1_arr)
      
      # smurf_plotting.complete_paper_plot(plot_dir=FLAGS.plot_dir, index=i,
      #                                   image1=image1, image2=image2,
      #                                   flow_uv=flow_forward,
      #                                   ground_truth_flow_uv=None,
      #                                   flow_valid_occ=None,
      #                                   predicted_occlusion=occlusion,
      #                                   ground_truth_occlusion=None)
      
      if "sintel" not in dir:
        dir_name_for_frame_src = os.path.basename(dir)
      else:
        dir_name_for_frame_src = "sintel/market_2/final"
      output_path = os.path.join(OUTPUT_DIR, dir_name_for_frame_src, "SMURF_kitti-smurf")
      
      save_flow_image(-flow_forward.numpy(), idx, output_path, model="SMURF", res=(640, 480))


if __name__ == '__main__':
  app.run(main)
