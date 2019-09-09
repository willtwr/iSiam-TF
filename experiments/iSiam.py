#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 WR Tan     National Tsing Hua University
# 
# Distributed under terms of the MIT license.

"""Train the color model in the iSiam paper from scratch with Data Augmentation"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..'))

from trains.siamese_2base.configuration import LOG_DIR
from trains.siamese_2base.train_siamese_model import ex

if __name__ == '__main__':
  RUN_NAME = 'iSiam'
  ex.run(config_updates={'train_config': {'train_dir': osp.join(LOG_DIR, 'track_model_checkpoints', RUN_NAME), },
                         'track_config': {'log_dir': osp.join(LOG_DIR, 'track_model_inference', RUN_NAME), }
                         },
         options={'--name': RUN_NAME,
                  '--force': True,
                  '--enforce_clean': False,
                  })
