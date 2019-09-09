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

# Code root absolute path
CODE_ROOT = '/home/william/SiamFC-TensorFlow'

sys.path.insert(0, CODE_ROOT)


if __name__ == '__main__':
    # run experiments
    experiment = ExperimentUAV123('/home/william/dataset/uav123/UAV123', version='UAV123')
    
    # report performance
    experiment.report(['ISiamTracker',
                       'i-Siam',
                       'DaSiamRPN',
                       'SiamRPN',
                       'ECO',
                       'SAMF',
                       'SRDCF',
                       'Struck',
                       'TLD',
                       'MEEM',
                       'OAB'
                       ])
