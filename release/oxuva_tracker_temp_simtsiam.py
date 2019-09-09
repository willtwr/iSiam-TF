#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.
"""Class for tracking using a track model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import logging
import os
import os.path as osp
from scipy import signal
import numpy as np
from scipy.misc import imresize
import cv2
from cv2 import imwrite, imread
from PIL import Image

from utils.infer_utils import convert_bbox_format, Rectangle
from utils.misc_utils import get_center, get


class TargetState(object):
  """Represent the target state."""
  def __init__(self, bbox, search_pos, scale_idx):
    self.bbox = bbox  # (cx, cy, w, h) in the original image
    self.search_pos = search_pos  # target center position in the search image
    self.scale_idx = scale_idx  # scale index in the searched scales


class Tracker(object):
  """Tracker based on the siamese model."""
  def __init__(self, siamese_model, model_config, track_config):
    self.siamese_model = siamese_model
    self.model_config = model_config
    self.track_config = track_config

    self.num_scales = track_config['num_scales']
    logging.info('track num scales -- {}'.format(self.num_scales))
    scales = np.arange(self.num_scales) - get_center(self.num_scales)
    self.search_factors_init = [self.track_config['scale_step'] ** x for x in scales]
    scales_5 = np.arange(self.num_scales + 2) - get_center(self.num_scales + 2)
    self.search_factors_init5 = [self.track_config['scale_step'] ** x for x in scales_5]    

    self.x_image_size_init = track_config['x_image_size']  # Search image size
    self.window = None  # Cosine window
    self.log_level = track_config['log_level']

  def init(self, sess, frame, first_bbox, logdir='/tmp'):
    # Get initial target bounding box and convert to center based
    self.i = 0
    first_bbox = Rectangle(first_bbox[0], first_bbox[1], first_bbox[2], first_bbox[3])
    bbox = convert_bbox_format(first_bbox, 'center-based')

    # Feed in the first frame image to set initial state.
    bbox_feed = [bbox.y, bbox.x, bbox.height, bbox.width]
    input_feed = [frame, bbox_feed, self.x_image_size_init, self.search_factors_init]
    frame2crop_scale, self.image_z = self.siamese_model.initialize(sess, input_feed)
    imwrite(osp.join(logdir, 'aimagez.jpg'), cv2.cvtColor(self.image_z, cv2.COLOR_RGB2BGR))
    
    # Storing target state
    self.original_target_height = bbox.height
    self.original_target_width = bbox.width
    self.search_center = np.array([get_center(self.x_image_size_init),
                                   get_center(self.x_image_size_init)])
    self.current_target_state = TargetState(bbox=bbox,
                                            search_pos=self.search_center,
                                            scale_idx=int(get_center(self.num_scales)))
    
    self.store_thresh = 0.9
    self.conf_thresh = 0.7
    self.bound_thresh = 0.5
    self.sup_thresh = 0.1
    
    self.mem_count = 0
    self.update_delay = 0
    self.lost = 0
    self.x_image_size = self.x_image_size_init
    self.image_c = None
    self.moved2border = False
    self.prev_score = self.conf_thresh + 0.01
    return True

  def track(self, sess, frame, logdir='/tmp'):
    """Runs tracking on a single image."""
    i = self.i = self.i + 1
    current_target_state = self.current_target_state
    original_target_height = self.original_target_height
    original_target_width = self.original_target_width
    search_center = self.search_center
    
    mem_count = self.mem_count
    moved2border = self.moved2border
    update_delay = self.update_delay + 1
    lost = self.lost + 1
    image_c = self.image_c
    x_image_size = self.x_image_size
    search_factors = self.search_factors_init
    conf_thresh = self.conf_thresh
    bound_thresh = self.bound_thresh
    sup_thresh = self.sup_thresh
    prev_score = self.prev_score
    
    hi, wi, _ = frame.shape
    h_ratio = current_target_state.bbox.height / hi
    w_ratio = current_target_state.bbox.width / wi
    t_i_ratio = max([h_ratio, w_ratio])
    
    if prev_score < conf_thresh:
      x_image_size += 100
      #x_image_size = min(x_image_size, ((1. - t_i_ratio) * 1.8 + 1.) * self.x_image_size_init)
      if t_i_ratio < 0.05:
        x_image_size = min(x_image_size, 555)
      elif t_i_ratio > 0.6:
        x_image_size = min(x_image_size, 255)
      elif t_i_ratio > 0.4:
        x_image_size = min(x_image_size, 355)      
      else:
        x_image_size = min(x_image_size, 455)
    else:
      x_image_size = self.x_image_size_init
      
    num_scales = len(search_factors)
    bbx = current_target_state.bbox.x
    bby = current_target_state.bbox.y
    bbw = current_target_state.bbox.width
    bbh = current_target_state.bbox.height
    bbox_feed = [bby, bbx, bbh, bbw]
    
    if i > 1:
      top = (current_target_state.bbox.y - (current_target_state.bbox.height / 2) < 10)
      left = (current_target_state.bbox.x - (current_target_state.bbox.width / 2) < 10)
      bottom = (current_target_state.bbox.y + (current_target_state.bbox.height / 2) > hi - 10)
      right = (current_target_state.bbox.x + (current_target_state.bbox.width / 2) > wi - 10)
      if top or left or bottom or right:
        if not prev_score < bound_thresh:
          moved2border = True
        if not moved2border:
          current_target_state.bbox = Rectangle(wi / 2, hi / 2, 
                                                current_target_state.bbox.width, 
                                                current_target_state.bbox.height)
          bbox_feed = [current_target_state.bbox.y, current_target_state.bbox.x,
                       current_target_state.bbox.height, current_target_state.bbox.width]              
      else:
        if not prev_score < bound_thresh:
          moved2border = False    
    
    if t_i_ratio < 0.3 and lost > 5:
      lost = 0
      diffy = hi * 0.5 - bbox_feed[0]
      diffx = wi * 0.5 - bbox_feed[1]
      bbox_feed = [diffy * 0.25 + bbox_feed[0], diffx * 0.25 + bbox_feed[1], 
                   bbox_feed[2], bbox_feed[3]]
    
    current_target_state.bbox = Rectangle(bbox_feed[1], bbox_feed[0], bbox_feed[3], bbox_feed[2])
    
    input_feed = [frame, bbox_feed, x_image_size, search_factors]
    outputs, metadata = self.siamese_model.inference_step(sess, input_feed)
    search_scale_list = outputs['scale_xs']
    response = outputs['response']
    response_size = response.shape[1]
    re_out = np.around(1 / (1 + np.exp(-response)), 2)
    
    if np.max(re_out) < conf_thresh:
      x_image_sizeb4 = x_image_size
      x_image_size += 100
      #x_image_size_l = ((1. - t_i_ratio) * 1.8 + 1.) * self.x_image_size_init
      if t_i_ratio < 0.05:
        x_image_size_l = 555
      elif t_i_ratio > 0.6:
        x_image_size_l = 255
      elif t_i_ratio > 0.4:
        x_image_size_l = 355
      else:
        x_image_size_l = 455
      
      if not x_image_size > x_image_size_l:
        input_feed = [frame, bbox_feed, x_image_size, search_factors]
        outputs, metadata = self.siamese_model.inference_step(sess, input_feed)
        search_scale_list = outputs['scale_xs']
        response = outputs['response']
        response_size = response.shape[1]
        re_out = np.around(1 / (1 + np.exp(-response)), 2)
      else:
        x_image_size = x_image_sizeb4
      
    # Choose the scale whole response map has the highest peak
    if num_scales > 1:
      response_max = np.max(response * (re_out > sup_thresh), axis=(1, 2))
      penalties = self.track_config['scale_penalty'] * np.ones((num_scales))
      current_scale_idx = int(get_center(num_scales))
      penalties[current_scale_idx] = 1.0
      response_penalized = response_max * penalties
      if max(response_penalized) == 0.:
        best_scale = 1
      else:
        best_scale = np.argmax(response_penalized)
    else:
      best_scale = 0
    
    response = response[best_scale]
    re_out = re_out[best_scale]
    
    with np.errstate(all='raise'):  # Raise error if something goes wrong
      response = response - np.min(response)
      response = response / np.sum(response)
      response = response * (re_out > sup_thresh)
      
    window = np.dot(np.expand_dims(np.hanning(response_size), 1),
                    np.expand_dims(np.hanning(response_size), 0))
    self.window = window / np.sum(window)  # normalize window
    window_influence = self.track_config['window_influence']
    response = (1 - window_influence) * response + window_influence * self.window

    # Find maximum response
    r_max, c_max = np.unravel_index(response.argmax(),
                                    response.shape)
    prev_score = re_out[r_max, c_max]
    
    # Convert from crop-relative coordinates to frame coordinates
    p_coor = np.array([r_max, c_max])
    # displacement from the center in instance final representation ...
    disp_instance_final = p_coor - get_center(response_size)
    # ... in instance feature space ...
    upsample_factor = self.track_config['upsample_factor']
    disp_instance_feat = disp_instance_final / upsample_factor
    # ... Avoid empty position ...
    r_radius = int(response_size / upsample_factor / 2)
    disp_instance_feat = np.maximum(np.minimum(disp_instance_feat, r_radius), -r_radius)
    # ... in instance input ...
    disp_instance_input = disp_instance_feat * self.model_config['embed_config']['stride']
    # ... in instance original crop (in frame coordinates)
    disp_instance_frame = disp_instance_input / search_scale_list[best_scale]
    # Position within frame in frame coordinates
    y = current_target_state.bbox.y
    x = current_target_state.bbox.x
    y += disp_instance_frame[0]
    x += disp_instance_frame[1]
    y = np.round(y)
    x = np.round(x)
    
    # Target scale damping and saturation
    target_scale = current_target_state.bbox.height / original_target_height
    search_factor = search_factors[best_scale]
    scale_damp = self.track_config['scale_damp']  # damping factor for scale update
    target_scale *= ((1 - scale_damp) * 1.0 + scale_damp * search_factor)
    
    # Some book keeping
    search_center = np.array([get_center(x_image_size),
                              get_center(x_image_size)])        
    height = original_target_height * target_scale
    width = original_target_width * target_scale
    current_target_state.bbox = Rectangle(x, y, width, height)
    current_target_state.scale_idx = best_scale
    current_target_state.search_pos = search_center + disp_instance_input
    
    assert 0 <= current_target_state.search_pos[0] < x_image_size, \
      'target position in feature space should be no larger than input image size'
    assert 0 <= current_target_state.search_pos[1] < x_image_size, \
      'target position in feature space should be no larger than input image size'

    if self.log_level > 0:
      # Select the image with the highest score scale and convert it to uint8
      image_cropped = outputs['image_cropped'][best_scale].astype(np.uint8)

      y_search, x_search = current_target_state.search_pos
      search_scale = search_scale_list[best_scale]
      target_height_search = height * search_scale
      target_width_search = width * search_scale
      bbox_search = Rectangle(x_search, y_search, target_width_search, target_height_search)
      bbox_search = convert_bbox_format(bbox_search, 'top-left-based')
      
      xmin = bbox_search.x.astype(np.int32)
      ymin = bbox_search.y.astype(np.int32)
      xmax = xmin + bbox_search.width.astype(np.int32)
      ymax = ymin + bbox_search.height.astype(np.int32)
      cv2.rectangle(image_cropped, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
      text = str(prev_score)
      cv2.putText(image_cropped, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), lineType=cv2.LINE_AA)
      imwrite(osp.join(logdir, 'image_cropped{}.jpg'.format(i)),
              cv2.cvtColor(image_cropped, cv2.COLOR_RGB2BGR))

    if prev_score > self.store_thresh:
      bbox_feed = [current_target_state.bbox.y, current_target_state.bbox.x,
                   current_target_state.bbox.height, current_target_state.bbox.width]      
      self.siamese_model.update_mem(sess, [frame, bbox_feed, self.x_image_size_init, 
                                           self.search_factors_init, mem_count])
      mem_count += 1
      
    if mem_count > 4 or (mem_count > 0 and update_delay > 5):
      self.siamese_model.update(sess)
      mem_count = 0
      update_delay = 0
    
    if prev_score > bound_thresh:
      lost = 0
    
    self.mem_count = mem_count
    self.update_delay = update_delay
    self.moved2border = moved2border
    self.lost = lost
    self.x_image_size = x_image_size
    self.prev_score = prev_score
    self.current_target_state = current_target_state
    reported_bbox = convert_bbox_format(current_target_state.bbox, 'top-left-based')
    #return prev_score>0.4, reported_bbox, prev_score
    return prev_score>0.4, reported_bbox
