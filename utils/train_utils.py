#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
# Modified by william 2018  National Tsing Hua University
# Distributed under terms of the MIT license.

"""Utilities for model construction"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import numpy as np
import tensorflow as tf
from scipy import io as sio
import torch
from utils.misc_utils import get_center


def create_mask(image, target_size=127):
  """Randomly create a rectangle mask. 
     Assume target is at the center of the image.
  todo: ignore this function for now..
  Args:
    image: image [height, width, channel]
    target_size: Size of the target. Default to 127.
    
  Return:
    A float tensor mask of shape [batch_size, height, width, 1]
  """
  im_size = tf.shape(image)
  im_shape = tf.cast(im_size, tf.float32)
  
  cx = im_size / 2
  cy = im_size / 2
  w = target_size
  h = target_size

  randh = tf.random_uniform([], 0.2, 0.5)
  randw = tf.random_uniform([], 0.2, 0.5)
  ymin = tf.random_uniform([], 0., 1. - randh) * h + next_bbox[0, 0]
  xmin = tf.random_uniform([], 0., 1. - randw) * w + next_bbox[0, 1]
  ymax = ymin + randh * h
  xmax = xmin + randw * w

  ymin = tf.cast(ymin, tf.int32)
  xmin = tf.cast(xmin, tf.int32)
  ymax = tf.cast(ymax, tf.int32)
  xmax = tf.cast(xmax, tf.int32)

  w = xmax - xmin
  h = ymax - ymin
  indicesh = tf.tile(tf.range(ymin, ymax), [w])
  indicesw = tf.tile(tf.expand_dims(tf.range(xmin, xmax), 1), [1, h])
  indicesw = tf.reshape(indicesw, [-1])

  indices = tf.cast(tf.stack([indicesh, indicesw], 1), tf.int64)
  mask = tf.SparseTensor(indices, tf.ones([h * w]), tf.cast(im_size[:2], tf.int64))
  mask = tf.sparse_reorder(mask)
  mask = tf.sparse_tensor_to_dense(mask)
  mask = tf.expand_dims(mask, -1)
  return mask


def construct_gt_score_maps(response_size, batch_size, stride, gt_config=None, n_out=1):
  """Construct a batch of groundtruth score maps

  Args:
    response_size: A list or tuple with two elements [ho, wo]
    batch_size: An integer e.g., 16
    stride: Embedding stride e.g., 8
    gt_config: Configurations for groundtruth generation

  Return:
    A float tensor of shape [batch_size] + response_size
  """
  with tf.name_scope('construct_gt'):
    ho = response_size[0]
    wo = response_size[1]
    y = tf.cast(tf.range(0, ho), dtype=tf.float32) - get_center(ho)
    x = tf.cast(tf.range(0, wo), dtype=tf.float32) - get_center(wo)
    [Y, X] = tf.meshgrid(y, x)

    def _logistic_label(X, Y, rPos, rNeg):
      # dist_to_center = tf.sqrt(tf.square(X) + tf.square(Y))  # L2 metric
      dist_to_center = tf.abs(X) + tf.abs(Y)  # Block metric
      Z = tf.where(dist_to_center <= rPos,
                   tf.ones_like(X),
                   tf.where(dist_to_center < rNeg,
                            0.5 * tf.ones_like(X),
                            tf.zeros_like(X)))
      return Z

    rPos = gt_config['rPos'] / stride
    rNeg = gt_config['rNeg'] / stride
    gt = _logistic_label(X, Y, rPos, rNeg)

    # Duplicate a batch of maps
    if n_out > 1:
      gt_expand = tf.reshape(gt, [1] + response_size + [1])
      gt = tf.tile(gt_expand, [batch_size, 1, 1, n_out])
    else:
      gt_expand = tf.reshape(gt, [1] + response_size)
      gt = tf.tile(gt_expand, [batch_size, 1, 1])
    return gt


def construct_seg_score_maps(response_size, bboxes, im_size):
  """Construct a batch of groundtruth score maps

  Args:
    response_size: A list or tuple with two elements [ho, wo]
    bboxes: Labels for bounding boxes
    im_size: Image size

  Return:
    A float tensor of shape [batch_size] + response_size
  """
  with tf.name_scope('construct_gt'):
    ho = response_size[0]
    wo = response_size[1]
    y = tf.cast(tf.range(0, ho), dtype=tf.float32) - get_center(ho)
    x = tf.cast(tf.range(0, wo), dtype=tf.float32) - get_center(wo)
    [X, Y] = tf.meshgrid(x, y)
    
    def _logistic_label(Y, X, H, W):
      Y = tf.abs(Y)
      X = tf.abs(X)
      Z = tf.where(Y <= H * ho / im_size[0] / 2,
                   tf.ones_like(Y),
                   tf.zeros_like(Y))
      Z = tf.where(X <= W * wo / im_size[1] / 2,
                   Z,
                   tf.zeros_like(X))
      return Z
    
    gt = tf.map_fn(lambda x: _logistic_label(Y, X, tf.to_float(x[0]), tf.to_float(x[1])),
                   bboxes,
                   dtype=tf.float32)
    return gt
  
  
def construct_xseg_score_maps(response_size, bboxes, im_size):
  """Construct a batch of groundtruth score maps

  Args:
    response_size: A list or tuple with two elements [ho, wo]
    bboxes: Labels for bounding boxes
    im_size: Image size

  Return:
    A float tensor of shape [batch_size] + response_size
  """
  with tf.name_scope('construct_gt'):
    ho = response_size[0]
    wo = response_size[1]
    y = tf.cast(tf.range(0, ho), dtype=tf.float32) - get_center(ho)
    x = tf.cast(tf.range(0, wo), dtype=tf.float32) - get_center(wo)
    [X, Y] = tf.meshgrid(x, y)
    
    def _logistic_label(Y, X, H, W):
      Y = tf.abs(Y)
      X = tf.abs(X)
      ZH = tf.where(Y <= H * ho / im_size[0] / 2,
                    tf.ones_like(Y),
                    tf.zeros_like(Y))
      ZH = tf.where(X <= 0.2 * W * wo / im_size[1] / 2,
                    ZH,
                    tf.zeros_like(X))
      ZW = tf.where(Y <= 0.2 * H * ho / im_size[0] / 2,
                    tf.ones_like(Y),
                    tf.zeros_like(Y))
      ZW = tf.where(X <= W * wo / im_size[1] / 2,
                    ZW,
                    tf.zeros_like(X))
      Z = ZH + ZW
      Z = tf.clip_by_value(Z, 0., 1.)
      return Z
    
    gt = tf.map_fn(lambda x: _logistic_label(Y, X, tf.to_float(x[0]), tf.to_float(x[1])),
                   bboxes,
                   dtype=tf.float32)
    return gt


def get_params_from_mat(matpath):
  """Get parameter from .mat file into parms(dict)"""

  def squeeze(vars_):
    # Matlab save some params with shape (*, 1)
    # However, we don't need the trailing dimension in TensorFlow.
    if isinstance(vars_, (list, tuple)):
      return [np.squeeze(v, 1) for v in vars_]
    else:
      return np.squeeze(vars_, 1)

  netparams = sio.loadmat(matpath)["net"]["params"][0][0]
  params = dict()

  for i in range(netparams.size):
    param = netparams[0][i]
    name = param["name"][0]
    value = param["value"]
    value_size = param["value"].shape[0]

    match = re.match(r"([a-z]+)([0-9]+)([a-z]+)", name, re.I)
    if match:
      items = match.groups()
    elif name == 'adjust_f':
      params['detection/weights'] = squeeze(value)
      continue
    elif name == 'adjust_b':
      params['detection/biases'] = squeeze(value)
      continue
    else:
      raise Exception('unrecognized layer params')

    op, layer, types = items
    layer = int(layer)
    if layer in [1, 3]:
      if op == 'conv':  # convolution
        if types == 'f':
          params['conv%d/weights' % layer] = value
        elif types == 'b':
          value = squeeze(value)
          params['conv%d/biases' % layer] = value
      elif op == 'bn':  # batch normalization
        if types == 'x':
          m, v = squeeze(np.split(value, 2, 1))
          params['conv%d/BatchNorm/moving_mean' % layer] = m
          params['conv%d/BatchNorm/moving_variance' % layer] = np.square(v)
        elif types == 'm':
          value = squeeze(value)
          params['conv%d/BatchNorm/gamma' % layer] = value
        elif types == 'b':
          value = squeeze(value)
          params['conv%d/BatchNorm/beta' % layer] = value
      else:
        raise Exception
    elif layer in [2, 4]:
      if op == 'conv' and types == 'f':
        b1, b2 = np.split(value, 2, 3)
      else:
        b1, b2 = np.split(value, 2, 0)
      if op == 'conv':
        if types == 'f':
          params['conv%d/b1/weights' % layer] = b1
          params['conv%d/b2/weights' % layer] = b2
        elif types == 'b':
          b1, b2 = squeeze(np.split(value, 2, 0))
          params['conv%d/b1/biases' % layer] = b1
          params['conv%d/b2/biases' % layer] = b2
      elif op == 'bn':
        if types == 'x':
          m1, v1 = squeeze(np.split(b1, 2, 1))
          m2, v2 = squeeze(np.split(b2, 2, 1))
          params['conv%d/b1/BatchNorm/moving_mean' % layer] = m1
          params['conv%d/b2/BatchNorm/moving_mean' % layer] = m2
          params['conv%d/b1/BatchNorm/moving_variance' % layer] = np.square(v1)
          params['conv%d/b2/BatchNorm/moving_variance' % layer] = np.square(v2)
        elif types == 'm':
          params['conv%d/b1/BatchNorm/gamma' % layer] = squeeze(b1)
          params['conv%d/b2/BatchNorm/gamma' % layer] = squeeze(b2)
        elif types == 'b':
          params['conv%d/b1/BatchNorm/beta' % layer] = squeeze(b1)
          params['conv%d/b2/BatchNorm/beta' % layer] = squeeze(b2)
      else:
        raise Exception

    elif layer in [5]:
      if op == 'conv' and types == 'f':
        b1, b2 = np.split(value, 2, 3)
      else:
        b1, b2 = squeeze(np.split(value, 2, 0))
      assert op == 'conv', 'layer5 contains only convolution'
      if types == 'f':
        params['conv%d/b1/weights' % layer] = b1
        params['conv%d/b2/weights' % layer] = b2
      elif types == 'b':
        params['conv%d/b1/biases' % layer] = b1
        params['conv%d/b2/biases' % layer] = b2

  return params


def load_mat_model(matpath, embed_scope, detection_scope=None):
  """Restore SiameseFC models from .mat model files"""
  params = get_params_from_mat(matpath)

  assign_ops = []

  def _assign(ref_name, params, scope=embed_scope):
    var_in_model = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope + ref_name)[0]
    var_in_mat = params[ref_name]
    op = tf.assign(var_in_model, var_in_mat)
    assign_ops.append(op)

  for l in range(1, 6):
    if l in [1, 3]:
      _assign('conv%d/weights' % l, params)
      # _assign('conv%d/biases' % l, params)
      _assign('conv%d/BatchNorm/beta' % l, params)
      _assign('conv%d/BatchNorm/gamma' % l, params)
      _assign('conv%d/BatchNorm/moving_mean' % l, params)
      _assign('conv%d/BatchNorm/moving_variance' % l, params)
    elif l in [2, 4]:
      # Branch 1
      _assign('conv%d/b1/weights' % l, params)
      # _assign('conv%d/b1/biases' % l, params)
      _assign('conv%d/b1/BatchNorm/beta' % l, params)
      _assign('conv%d/b1/BatchNorm/gamma' % l, params)
      _assign('conv%d/b1/BatchNorm/moving_mean' % l, params)
      _assign('conv%d/b1/BatchNorm/moving_variance' % l, params)
      # Branch 2
      _assign('conv%d/b2/weights' % l, params)
      # _assign('conv%d/b2/biases' % l, params)
      _assign('conv%d/b2/BatchNorm/beta' % l, params)
      _assign('conv%d/b2/BatchNorm/gamma' % l, params)
      _assign('conv%d/b2/BatchNorm/moving_mean' % l, params)
      _assign('conv%d/b2/BatchNorm/moving_variance' % l, params)
    elif l in [5]:
      # Branch 1
      _assign('conv%d/b1/weights' % l, params)
      _assign('conv%d/b1/biases' % l, params)
      # Branch 2
      _assign('conv%d/b2/weights' % l, params)
      _assign('conv%d/b2/biases' % l, params)
    else:
      raise Exception('layer number must below 5')

  if detection_scope:
    _assign(detection_scope + 'biases', params, scope='')

  initialize = tf.group(*assign_ops)
  return initialize


def load_pretrained_model(pth, embed_scope):
  """Load pretrained model from .pth model files"""
  model = torch.load(pth)
  assign_ops = []

  def _assign(ref_name, params, scope=embed_scope):
    var_in_model = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope + ref_name)[0]
    var_in_mat = params
    op = tf.assign(var_in_model, var_in_mat)
    assign_ops.append(op)

  for l_name, param in model.iteritems():
    params = param.data.numpy()
    l_int = int(l_name.split('.')[2])
    
    if l_int == 0:
      if 'weight' in l_name:
        _assign('conv1_pre/weights', np.transpose(params, [2, 3, 1, 0]))
      elif 'bias' in l_name:
        #continue
        _assign('conv1_pre/biases', params)
      else:
        raise Exception('layer name NOT available!')
    elif l_int == 1:
      if 'bias' in l_name:
        _assign('conv1_pre/BatchNorm/beta', params)
      elif 'weight' in l_name:
        _assign('conv1_pre/BatchNorm/gamma', params)
      elif 'running_mean' in l_name:
        _assign('conv1_pre/BatchNorm/moving_mean', params)
      elif 'running_var' in l_name:
        _assign('conv1_pre/BatchNorm/moving_variance', params)
      else:
        raise Exception('layer name NOT available!')    
    elif l_int == 4:
      if 'weight' in l_name:
        _assign('conv2_pre/weights', np.transpose(params, [2, 3, 1, 0]))
      elif 'bias' in l_name:
        #continue
        _assign('conv2_pre/biases', params)
      else:
        raise Exception('layer name NOT available!')    
    elif l_int == 5:
      if 'bias' in l_name:
        _assign('conv2_pre/BatchNorm/beta', params)
      elif 'weight' in l_name:
        _assign('conv2_pre/BatchNorm/gamma', params)
      elif 'running_mean' in l_name:
        _assign('conv2_pre/BatchNorm/moving_mean', params)
      elif 'running_var' in l_name:
        _assign('conv2_pre/BatchNorm/moving_variance', params)
      else:
        raise Exception('layer name NOT available!')
    elif l_int == 8:
      if 'weight' in l_name:
        _assign('conv3_pre/weights', np.transpose(params, [2, 3, 1, 0]))
      elif 'bias' in l_name:
        #continue
        _assign('conv3_pre/biases', params)
      else:
        raise Exception('layer name NOT available!')    
    elif l_int == 9:
      if 'bias' in l_name:
        _assign('conv3_pre/BatchNorm/beta', params)
      elif 'weight' in l_name:
        _assign('conv3_pre/BatchNorm/gamma', params)
      elif 'running_mean' in l_name:
        _assign('conv3_pre/BatchNorm/moving_mean', params)
      elif 'running_var' in l_name:
        _assign('conv3_pre/BatchNorm/moving_variance', params)
      else:
        raise Exception('layer name NOT available!')
    elif l_int == 11:
      if 'weight' in l_name:
        _assign('conv4_pre/weights', np.transpose(params, [2, 3, 1, 0]))
      elif 'bias' in l_name:
        #continue
        _assign('conv4_pre/biases', params)
      else:
        raise Exception('layer name NOT available!')    
    elif l_int == 12:
      if 'bias' in l_name:
        _assign('conv4_pre/BatchNorm/beta', params)
      elif 'weight' in l_name:
        _assign('conv4_pre/BatchNorm/gamma', params)
      elif 'running_mean' in l_name:
        _assign('conv4_pre/BatchNorm/moving_mean', params)
      elif 'running_var' in l_name:
        _assign('conv4_pre/BatchNorm/moving_variance', params)
      else:
        raise Exception('layer name NOT available!')
    elif l_int == 14:
      if 'weight' in l_name:
        _assign('conv5_pre/weights', np.transpose(params, [2, 3, 1, 0]))
      elif 'bias' in l_name:
        #continue
        _assign('conv5_pre/biases', params)
      else:
        raise Exception('layer name NOT available!')    
    elif l_int == 15:
      if 'bias' in l_name:
        _assign('conv5_pre/BatchNorm/beta', params)
      elif 'weight' in l_name:
        _assign('conv5_pre/BatchNorm/gamma', params)
      elif 'running_mean' in l_name:
        _assign('conv5_pre/BatchNorm/moving_mean', params)
      elif 'running_var' in l_name:
        _assign('conv5_pre/BatchNorm/moving_variance', params)
      else:
        raise Exception('layer name NOT available!')
    else:
      raise Exception('No Match!')
      
  initialize = tf.group(*assign_ops)
  return initialize
