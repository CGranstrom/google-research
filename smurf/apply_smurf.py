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

import pprofile
profiler = pprofile.Profile()
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np

import smurf_flags  # pylint:disable=unused-import
#import smurf_plotting
import smurf_evaluator
from smurf_plotting import _FLOW_SCALING_FACTOR, flow_to_rgb
import yappi
import itertools
import smurf_net

from src import data
from output import opt_flow_output
from output import opt_flow_output_640_480
import pre_trained_models
from utils.data_viz import display_flow
from utils.file_io import save_flow_image
import flowiz as fz
import pandas as pd
print(pd.__version__)

try:
  import cv2  # pylint:disable=g-import-not-at-top
except:  # pylint:disable=bare-except
  print('Missing cv2 dependency. Please install opencv-python.')

MODEL_DIR = os.path.join(os.path.dirname(pre_trained_models.__file__), "sintel-smurf")
# MODEL_PATHS = [MODEL_DIR]#, os.path.join(os.path.dirname(pre_trained_models.__file__), "sintel-smurf")]
MODEL_PATHS = [os.path.join(os.path.dirname(pre_trained_models.__file__), "sintel-smurf"),
               os.path.join(os.path.dirname(pre_trained_models.__file__), "kitti-smurf")]


DATA_PATH = os.path.dirname(data.__file__)
FRAMES_DIR = os.path.join(DATA_PATH, "frames_640_480/ir_video_of_compartment_fire_extra")
#OUTPUT_DIR = os.path.dirname(output.__file__)
OUTPUT_DIR = os.path.dirname(opt_flow_output_640_480.__file__)
METRICS_REPORT_PATH = os.path.join(DATA_PATH, "../system_resource_metrics")

DIRS_TO_SKIP = ("34539_fire_helmet_cam_2014_extra", "sintel")
PATHS_TO_SKIP = tuple(os.path.join(FRAMES_DIR, _dir) for _dir in DIRS_TO_SKIP)

#RUN_MODE = "perf_testing"
RUN_MODE = "gen_images"
NUM_IMAGES_IN_DIR = 200  # 208
clock_types = ["wall"]#, "cpu"]


flags.DEFINE_string('data_dir', '', 'Directory with images to run on. Images '
                    'should be named numerically, e.g., 1.png, 2.png.')

FLAGS = flags.FLAGS


def get_image_iterator(image_dir: str, end_idx: int):
  """Iterate through images in the image_dir."""
  
  images = os.listdir(image_dir)[:end_idx]
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
  for model_path, CLOCK_TYPE in itertools.product(MODEL_PATHS, clock_types):
    print(f"NOW ON MODEL {model_path}, clock type {CLOCK_TYPE}\n\n\n\n\n")
    yappi.set_clock_type(CLOCK_TYPE)
    
    experiment_label = f"TOTAL for {os.path.basename(model_path)} {CLOCK_TYPE} time"
    dir_names = [experiment_label]

    gin.parse_config_files_and_bindings(FLAGS.config_file, FLAGS.gin_bindings)
    smurf = smurf_evaluator.build_network(batch_size=1)
    smurf.update_checkpoint_dir(MODEL_DIR)
    smurf.restore()
    
    image_dirs = []
    for parent_dir, _, images in sorted(os.walk(FRAMES_DIR)):
      if not images or parent_dir in PATHS_TO_SKIP:
        continue
      elif "sintel" in parent_dir:
        dir_names.append("sintel")
        image_dirs.append(parent_dir)
      else:
        dir_names.append(os.path.basename(parent_dir))
        image_dirs.append(parent_dir)
    
    image_dirs = sorted(image_dirs)
    
    run_stats = pd.DataFrame(
        [[0.0] * 5 + [(1, 1)]] * len(dir_names),
        columns=[
            "dir_agg_runtime",
            "dir_mean_runtime",
            "dir_median_runtime",
            "dir_std_dev_runtime",
            "dir_num_trials",
            "image_res",
            ],
        index=dir_names, dtype=np.float
        )
    total_runtimes = []
    
    TRIAL_CTR=0
    
    if RUN_MODE == "perf_testing":
      for dir in image_dirs:
        (dir_num_trials,dir_agg_runtime,dir_mean_runtime,dir_median_runtime,dir_std_dev_runtime) = (0, 0, 0, 0, 0)
        dir_runtimes = []
        
        for idx, (image1, image2) in enumerate(get_image_iterator(dir, NUM_IMAGES_IN_DIR)):
            sys.stdout.write(':')
            sys.stdout.flush()
            
            image_res = image1.shape[:2]
            
            TRIAL_CTR += 1
            print(f"TRIAL_CTR={TRIAL_CTR}\n\n\n\n\n")
            
            #yappi.start()
            #with profiler:
            start = timer()
            # flow_forward, occlusion, flow_backward = smurf.infer(
            #     image1, image2, input_height=FLAGS.height, input_width=FLAGS.width,
            #     infer_occlusion=True, infer_bw=True)
            # flow_forward, occlusion, flow_backward = smurf.infer(
            #     image1, image2, infer_occlusion=False, infer_bw=False)  # 296 x 696, 480 x 640
            
            # faster version
            # flow_forward = smurf.infer(
            #     image1, image2, infer_occlusion=False, infer_bw=False, resize_flow_to_img_res=False)
            
            # just RAFT
            images = tf.stack([image1, image2])[None]
            feature_dict = smurf._feature_model(images[:,0], images[:,1], bidirectional=False)
            flow_forward = smurf._flow_model(feature_dict, training=False)[0]
            end = timer()
            elapsed_time=end-start
            #print(f"end-start={end-start}")
            #profiler.print_stats()
            #yappi.stop()
            # a = yappi.get_func_stats()
            # b = sorted([a._as_dict[list(a._as_dict.keys())[b]][0] for b in range(len(list(a._as_dict.keys())))])
            # print(b)
            
            # get total runtime for model()
            # stats = yappi.get_func_stats(
            #     filter_callback=lambda x: yappi.func_matches(
            #         x, [smurf.tf__infer_no_tf_function]
            #     )
            # )
            # stats = yappi.get_func_stats(
            #     filter_callback=lambda x: yappi.func_matches(
            #         x, [smurf_net.SMURFNet.infer]
            #     )
            # )
            # stats = yappi.get_func_stats()
            # #stats = list(range(20))
            # stat_list = stats._as_list[14]
            # tf__infer_entry = stat_list[6]
            # print(f"func_name is {stat_list[0]}\n\n\n\n\n")
            # print(f"time is {tf__infer_entry}\n\n\n\n\n")
            # dir_runtimes.append(tf__infer_entry)
            # yappi.get_func_stats().print_all()
            # print("\n\n\n\n\n")
            # yappi.get_thread_stats().print_all()
            #dir_runtimes.append(list(stats._as_dict.keys())[0][6])
            #yappi.clear_stats()
            dir_runtimes.append(elapsed_time)

            
            # SMURFNet.tf__infer
            #occlusion = 1. - occlusion
            
            
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
            #output_path = os.path.join(OUTPUT_DIR, dir_name_for_frame_src, "SMURF_kitti-smurf")
            
            #save_flow_image(-flow_forward.numpy(), idx, output_path, model="SMURF", res=(640, 480))
        dir_agg_runtime = sum(dir_runtimes)
        dir_mean_runtime = np.mean(dir_runtimes)
        dir_median_runtime = np.median(dir_runtimes)
        dir_std_dev_runtime = np.std(dir_runtimes)
        dir_num_trials = len(dir_runtimes)

        if "sintel" in dir_name_for_frame_src:
            dir_name_for_frame_src = "sintel"

        # run_stats.loc[dir_name_for_frame_src] = {
        #     "dir_agg_runtime": dir_agg_runtime,
        #     "dir_mean_runtime": dir_mean_runtime,
        #     "dir_median_runtime": dir_median_runtime,
        #     "dir_std_dev_runtime": dir_std_dev_runtime,
        #     "dir_num_trials": dir_num_trials,
        #     "image_res": image_res,
        # }
        tmp_dict = {"dir_agg_runtime": dir_agg_runtime, "dir_mean_runtime": dir_mean_runtime, "dir_median_runtime": dir_median_runtime, "dir_std_dev_runtime":dir_std_dev_runtime, "dir_num_trials": dir_num_trials, "image_res": image_res}
        
        run_stats.loc[dir_name_for_frame_src, tmp_dict.keys()] = tmp_dict.values()
        total_runtimes.extend(dir_runtimes)
        
        #yappi.clear_stats()

        
        
    elif RUN_MODE == "gen_images":
      for dir in image_dirs:
        for idx, (image1, image2) in enumerate(get_image_iterator(dir, NUM_IMAGES_IN_DIR)):


          sys.stdout.write(':')
          sys.stdout.flush()
          flow_forward, occlusion, flow_backward = smurf.infer(
              image1, image2,
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
          
    total_runtime = sum(total_runtimes)
    total_mean_runtime = np.mean(total_runtimes)
    total_median_runtime = np.median(total_runtimes)
    total_std_dev_runtime = np.std(total_runtimes)
    total_num_trials = len(total_runtimes)

    tmp_dict = {"dir_agg_runtime": total_runtime, "dir_mean_runtime": total_mean_runtime, "dir_median_runtime": total_median_runtime, "dir_std_dev_runtime":total_std_dev_runtime, "dir_num_trials": total_num_trials, "image_res": np.nan}
        
    run_stats.loc[experiment_label, tmp_dict.keys()] = tmp_dict.values()
    # run_stats.loc[experiment_label] = {
    #     "dir_agg_runtime": total_runtime,
    #     "dir_mean_runtime": total_mean_runtime,
    #     "dir_median_runtime": total_median_runtime,
    #     "dir_std_dev_runtime": total_std_dev_runtime,
    #     "dir_num_trials": total_num_trials,
    #     "image_res": np.nan,
    #     }
    
    
    # run_stats.to_csv(
    #     os.path.join(
    #         METRICS_REPORT_PATH,
    #         f"{os.path.basename(model_path)}_{CLOCK_TYPE}_runtimes_640_480_no_occlusion_or_bkwrd_more_images_justRAFT.csv",
    #     )
    #     )    


if __name__ == '__main__':
  app.run(main)
