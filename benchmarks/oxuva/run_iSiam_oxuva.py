#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 WR Tan     National Tsing Hua University
#
# Distributed under terms of the MIT license.

r"""Support integration with OxUvA benchmark"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import sys
import time
import cv2
import tensorflow as tf

# Code root absolute path
CODE_ROOT = '/home/william/iSiam-TF'

# Checkpoint for evaluation
CHECKPOINT = '/home/william/iSiam-TF/Logs/SiamFC/track_model_checkpoints/iSiam'

sys.path.insert(0, CODE_ROOT)

from utils.misc_utils import auto_select_gpu, load_cfgs
from inference import inference_wrapper_oxuva as inference_wrapper
from inference.tracker_oxuva import Tracker
from utils.infer_utils import Rectangle

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.logging.set_verbosity(tf.logging.DEBUG)


class iSiam_tracker():
  def __init__(self,):
    checkpoint_path = CHECKPOINT
    logging.info('Evaluating {}...'.format(checkpoint_path))

    # Read configurations from json
    self.model_config, _, self.track_config = load_cfgs(checkpoint_path)

    self.track_config['log_level'] = 1  # Skip verbose logging for speed
    self.track_config['scale_step'] = 1.025  # 1.025
    self.track_config['scale_damp'] = 1.
    #self.track_config['window_influence'] = 0.125  # 0.176
    
    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
      self.model = inference_wrapper.InferenceWrapper()
      restore_fn = self.model.build_graph_from_config(self.model_config, self.track_config, checkpoint_path)
    g.finalize()

    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    self.sess = tf.Session(graph=g, config=sess_config)
    
    # Load the model from checkpoint.
    restore_fn(self.sess)
    self.tracker = Tracker(self.model, self.model_config, self.track_config)
    self.vi = 0
    
  def init(self, im, rec):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    self.dir_name = os.path.join('/home/william/long-term-tracking-benchmark/results/samples', str(self.vi))
    if not os.path.exists(self.dir_name):
       os.mkdir(self.dir_name)
    self.vi += 1
    return self.tracker.init(self.sess, im, rec, self.dir_name)

  def update(self, im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return self.tracker.track(self.sess, im, self.dir_name)
