#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.

"""Miscellaneous Utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import errno
import json
import logging
import os
import re
import sys
from struct import unpack
from os import path as osp
import tensorflow as tf

import tfplot
import seaborn.apionly as sns

try:
  import pynvml  # nvidia-ml provides utility for NVIDIA management

  HAS_NVML = True
except:
  HAS_NVML = False


def auto_select_gpu():
  """Select gpu which has largest free memory"""
  if HAS_NVML:
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    largest_free_mem = 0
    largest_free_idx = 0
    for i in range(deviceCount):
      handle = pynvml.nvmlDeviceGetHandleByIndex(i)
      info = pynvml.nvmlDeviceGetMemoryInfo(handle)
      if info.free > largest_free_mem:
        largest_free_mem = info.free
        largest_free_idx = i
    pynvml.nvmlShutdown()
    largest_free_mem = largest_free_mem / 1024. / 1024.  # Convert to MB

    idx_to_gpu_id = {}
    for i in range(deviceCount):
      idx_to_gpu_id[i] = '{}'.format(i)

    gpu_id = idx_to_gpu_id[largest_free_idx]
    logging.info('Using largest free memory GPU {} with free memory {}MB'.format(gpu_id, largest_free_mem))
    return gpu_id
  else:
    logging.info('nvidia-ml-py is not installed, automatically select gpu is disabled!')
    return '0'


def heatmap_overlay(data, overlay_image=None, cmap='jet',
                    cbar=False, show_axis=False, alpha=0.5, **kwargs):
    fig, ax = tfplot.subplots(figsize=(5, 4) if cbar else (4, 4))
    fig.subplots_adjust(0, 0, 1, 1)  # use tight layout (no margins)
    ax.axis('off')

    if overlay_image is None: alpha = 1.0
    sns.heatmap(data, ax=ax, alpha=alpha, cmap=cmap, cbar=cbar, **kwargs)

    if overlay_image is not None:
        h, w = data.shape
        ax.imshow(overlay_image, extent=[0, h, 0, w])

    if show_axis:
        ax.axis('on')
        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
    return fig


def get_center(x):
  return (x - 1.) / 2.


def get(config, key, default):
  """Get value in config by key, use default if key is not set

  This little function is useful for dynamical experimental settings.
  For example, we can add a new configuration without worrying compatibility with older versions.
  You can also achieve this by just calling config.get(key, default), but add a warning is even better : )
  """
  val = config.get(key)
  if val is None:
    logging.warning('{} is not explicitly specified, using default value: {}'.format(key, default))
    val = default
  return val


def mkdir_p(path):
  """mimic the behavior of mkdir -p in bash"""
  try:
    os.makedirs(path)
  except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


def tryfloat(s):
  try:
    return float(s)
  except:
    return s


def alphanum_key(s):
  """ Turn a string into a list of string and number chunks.
      "z23a" -> ["z", 23, "a"]
  """
  return [tryfloat(c) for c in re.split('([0-9.]+)', s)]


def sort_nicely(l):
  """Sort the given list in the way that humans expect."""
  return sorted(l, key=alphanum_key)


class Tee(object):
  """Mimic the behavior of tee in bash

  From: http://web.archive.org/web/20141016185743/https://mail.python.org/pipermail/python-list/2007-May/460639.html
  Usage:
    tee=Tee('logfile', 'w')
    print 'abcdefg'
    print 'another line'
    tee.close()
    print 'screen only'
    del tee # should do nothing
  """

  def __init__(self, name, mode):
    self.file = open(name, mode)
    self.stdout = sys.stdout
    sys.stdout = self

  def close(self):
    if self.stdout is not None:
      sys.stdout = self.stdout
      self.stdout = None
    if self.file is not None:
      self.file.close()
      self.file = None

  def write(self, data):
    self.file.write(data)
    self.stdout.write(data)

  def flush(self):
    self.file.flush()
    self.stdout.flush()

  def __del__(self):
    self.close()


def save_cfgs(train_dir, model_config, train_config, track_config):
  """Save all configurations in JSON format for future reference"""
  with open(osp.join(train_dir, 'model_config.json'), 'w') as f:
    json.dump(model_config, f, indent=2)
  with open(osp.join(train_dir, 'train_config.json'), 'w') as f:
    json.dump(train_config, f, indent=2)
  with open(osp.join(train_dir, 'track_config.json'), 'w') as f:
    json.dump(track_config, f, indent=2)


def load_cfgs(checkpoint):
  if osp.isdir(checkpoint):
    train_dir = checkpoint
  else:
    train_dir = osp.dirname(checkpoint)

  with open(osp.join(train_dir, 'model_config.json'), 'r') as f:
    model_config = json.load(f)
  with open(osp.join(train_dir, 'train_config.json'), 'r') as f:
    train_config = json.load(f)
  with open(osp.join(train_dir, 'track_config.json'), 'r') as f:
    track_config = json.load(f)
  return model_config, train_config, track_config


def PSR(x, k_size=[5, 5]):
  """
  Peak to Sidelobe Ratio for response map
  """
  #assert tf.rank(x) == 3, 'x must have rank 3'
  
  #x = tf.expand_dims(x, -1)
  #x_shape = x.get_shape().as_list()
  patches = tf.extract_image_patches(x, 
                                     [1, k_size[0], k_size[1], 1], 
                                     [1, 1, 1, 1], [1, 1, 1, 1], 'SAME')  # [b, h, w, fh*fw]  
  mu, var = tf.nn.moments(patches, [3], keep_dims=True)
  return (x - mu) / (tf.sqrt(var) + 1e-8)


def GetWidthAndHeight(stream):
  """ Extract the dimensions from a jpeg file as a tuple (width, height)

  Keyword arguments:
  stream -- a stream (eg: open()ed). Must be opened in binary mode

  Author: Andrew Stephens http://sandfly.net.nz/
  License: use in any way you wish
  """
  while (True):
    data = stream.read(2)
    marker, = unpack(">H", data)

    # some markers stand alone
    if (marker == 0xffd8):  # start of image
      continue
    if (marker == 0xffd9):  # end of image
      return;
    if ((marker >= 0xffd0) and (marker <= 0xffd7)): # restart markers
      continue
    if (marker == 0xff01): # private marker:
      continue

    # all other markers specify chunks with lengths
    data = stream.read(2)
    length, = unpack(">H", data)

    if (marker == 0xffc0):  # baseline DCT chunk, has the info we want
      data = stream.read(5)
      t, height, width = unpack(">BHH", data)
      return (width, height)

    # not the chunk we want, skip it
    stream.seek(length - 2, 1)


def check_image(image):
  assertion = tf.assert_equal(tf.shape(image)[-1], 3, message='image must have 3 color channels')
  with tf.control_dependencies([assertion]):
    image = tf.identity(image)

  if image.get_shape().ndims not in (3, 4):
    raise ValueError('image must be either 3 or 4 dimensions')

  # make the last dimension 3 so that you can unstack the colors
  shape = list(image.get_shape())
  shape[-1] = 3
  image.set_shape(shape)
  return image


def rgb_to_lab(srgb):
  with tf.name_scope("rgb_to_lab"):
    #srgb = check_image(srgb)
    srgb_pixels = tf.reshape(srgb, [-1, 3])

    with tf.name_scope("srgb_to_xyz"):
      linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
      exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
      rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
      rgb_to_xyz = tf.constant([
              #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
      xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

    # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
    with tf.name_scope("xyz_to_cielab"):
      # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

      # normalize for D65 white point
      xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

      epsilon = 6/29
      linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
      exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
      fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

      # convert to lab
      fxfyfz_to_lab = tf.constant([
              #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
      lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

    return tf.reshape(lab_pixels, tf.shape(srgb))


def rgb_to_opp(rgb):
  with tf.name_scope("rgb_to_opp"):
    o1 = (rgb[...,0] - rgb[...,1]) / tf.sqrt(2.)
    o2 = (rgb[...,0] + rgb[...,1] - 2 * rgb[...,2]) / tf.sqrt(6.)
    o3 = (rgb[...,0] + rgb[...,1] + rgb[...,2]) / tf.sqrt(3.)
    
    return tf.stack([o1, o2, o3], -1)