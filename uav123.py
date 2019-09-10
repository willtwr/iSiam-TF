#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 WR Tan     National Tsing Hua University
#
# Distributed under terms of the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib 
matplotlib.use('Agg')

from got10k.trackers import Tracker
from got10k.experiments import ExperimentUAV123

import logging
import os
import sys
import time
import cv2
import numpy as np
import tensorflow as tf

# Code root absolute path
CODE_ROOT = '/home/william/SiamFC-TensorFlow'

# Checkpoint for evaluation
CHECKPOINT = '/home/william/iSiam-TF/Logs/SiamFC/track_model_checkpoints/SiamFC-iSiam'

sys.path.insert(0, CODE_ROOT)

from utils.misc_utils import load_cfgs
from inference import inference_wrapper_uav as inference_wrapper
from inference.tracker_uav import ISiamTrackerTF
from utils.infer_utils import Rectangle

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.logging.set_verbosity(tf.logging.DEBUG)


class ISiamTracker(Tracker):
    def __init__(self):
        super(ISiamTracker, self).__init__(name='ISiamTracker')
        checkpoint_path = CHECKPOINT
        logging.info('Evaluating {}...'.format(checkpoint_path))
    
        # Read configurations from json
        self.model_config, _, self.track_config = load_cfgs(checkpoint_path)
    
        self.track_config['log_level'] = 1  # Skip verbose logging for speed
        self.track_config['scale_step'] = 1.025  # 1.025
        self.track_config['scale_damp'] = 1.
        self.track_config['window_influence'] = 0.2  # 0.176
        
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
        self.tracker = ISiamTrackerTF(self.model, self.model_config, self.track_config)
        
    def init(self, image, box):
        vid = image.filename.split('/')[-2]
        im = np.array(image)
        self.dir_name = os.path.join('/home/william/uav123_benchmark/results/samples', vid)
        if not os.path.exists(self.dir_name):
            os.mkdir(self.dir_name)
        return self.tracker.init(self.sess, im, box, self.dir_name)
    
    def update(self, image):
        im = np.array(image)
        return self.tracker.track(self.sess, im, self.dir_name)


if __name__ == '__main__':
    # setup tracker
    tracker = ISiamTracker()

    # run experiments
    experiment = ExperimentUAV123('/home/william/dataset/uav123/UAV123', version='UAV123')
    experiment.run(tracker, visualize=False)

    # report performance
    experiment.report([tracker.name,
                       #'ISiamTracker_base',
                       #'ISiamTracker_nodft',
                       #'ISiamTracker_nofbgs_nodft',
                       #'ISiamTracker_noibgs_nodft',
                       #'ISiamTracker_nobgs_nodft',
                       #'ISiamTracker_dense_nobgs_nodft',
                       ])
