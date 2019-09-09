#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 bily     Huazhong University of Science and Technology
#
# Distributed under terms of the MIT license.


"""Various transforms for video and image augmentation"""

import numbers

import tensorflow as tf


class Compose(object):
  """Composes several transforms together."""

  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, example, ann=None):
    for t in self.transforms:
      example = t(example, ann)
    return example


class RandomGray(object):
  def __init__(self, gray_ratio=0.25):
    self.gray_ratio = gray_ratio

  def __call__(self, img_sequence):
    def rgb_to_gray():
      gray_images = tf.image.rgb_to_grayscale(img_sequence)
      return tf.concat([gray_images] * 3, axis=3)

    def identity():
      return tf.identity(img_sequence)

    return tf.cond(tf.less(tf.random_uniform([], 0, 1), self.gray_ratio), rgb_to_gray, identity)


class RandomStretch(object):
  def __init__(self, max_stretch=0.05, interpolation='bilinear'):
    self.max_stretch = max_stretch
    self.interpolation = interpolation

  def __call__(self, img, ann=None):
    scale = 1.0 + tf.random_uniform([], -self.max_stretch, self.max_stretch)
    img_shape = tf.shape(img)
    ts = tf.to_int32(tf.round(tf.to_float(img_shape[:2]) * scale))
    resize_method_map = {'bilinear': tf.image.ResizeMethod.BILINEAR,
                         'bicubic': tf.image.ResizeMethod.BICUBIC}
    return tf.image.resize_images(img, ts, method=resize_method_map[self.interpolation])


class CenterCrop(object):
  def __init__(self, size):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size

  def __call__(self, img, ann=None):
    th, tw = self.size
    return tf.image.resize_image_with_crop_or_pad(img, th, tw)


class RandomCrop(object):
  def __init__(self, size, return_offset=False):
    self.return_offset = return_offset
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size

  def __call__(self, img, ann=None):
    img_shape = tf.shape(img)
    th, tw = self.size
    
    y1 = tf.random_uniform([], 0, img_shape[0] - th, dtype=tf.int32)
    x1 = tf.random_uniform([], 0, img_shape[1] - tw, dtype=tf.int32)
    
    return tf.image.crop_to_bounding_box(img, y1, x1, th, tw)
  
  
class RandomCropBG(object):
  def __init__(self, size):
    self.size = size

  def __call__(self, img, ann=None):
    img_shape = tf.shape(img)
    size = self.size
    rand_list = tf.constant([[0, 0], 
                             [0, 271 - size],
                             [271 - size, 0], 
                             [271 - size, 271 - size]], 
                            dtype=tf.int32)
    
    get_rand = tf.split(tf.random.shuffle(rand_list), 4, 0)[0]
    return tf.image.crop_to_bounding_box(img, get_rand[0,0], get_rand[0,1], size, size)
  
  
class BboxCrop(object):
  def __init__(self, size):
    if isinstance(size, numbers.Number):
      self.size = (int(size), int(size))
    else:
      self.size = size

  def __call__(self, img, ann):
    th, tw = self.size
    ann = tf.cast(ann, tf.float32)
    ratio = ann[1] / ann[0]
    offsets = tf.cond(tf.logical_or(tf.greater(ratio, 1.5), tf.less(ratio, 0.66)), 
                      lambda: 1.,
                      lambda: 0.8)
    img = tf.image.resize_image_with_crop_or_pad(img, 
                                                 tf.cast(ann[1] * offsets, tf.int32), 
                                                 tf.cast(ann[0] * offsets, tf.int32))
    return tf.image.resize_image_with_crop_or_pad(img, th, tw)


class Bbox_correction(object):
  def __init__(self,):
    pass
  
  def __call__(self, img, ann):
    ann = [ann[1] - (ann[3] / 2),
           ann[0] - (ann[2] / 2),
           ann[1] - (ann[3] / 2) + ann[3],
           ann[0] - (ann[2] / 2) + ann[2]]
    ann = tf.squeeze(tf.stack([ann]))
    return img, ann


class Resize(object):
  def __init__(self, size):
    self.size = size

  def __call__(self, img, ann=None):
    return tf.image.resize_images(img, self.size)


class RandomFlip(object):
  def __init__(self,):
    pass

  def __call__(self, img, ann=None):
    return tf.image.random_flip_left_right(img)
  
  
class RandomScale(object):
  def __init__(self, crop_size, resize, scale=1.04):
    self.crop_size = crop_size
    self.resize = resize
    self.scale = scale

  def __call__(self, img, ann=None):
    th, tw = self.crop_size
    th = tf.cast(th, tf.float32)
    tw = tf.cast(tw, tf.float32)
    rs = tf.random.uniform([], tf.pow(self.scale, -1), self.scale)
    th = tf.cast(th * rs, tf.int32)
    tw = tf.cast(tw * rs, tf.int32)
    img = tf.image.resize_image_with_crop_or_pad(img, th, tw)
    return tf.image.resize_images(img, self.resize), tf.pow(rs, -1)
  
  
class RandomBrightness(object):
  def __init__(self, delta=0.25, proc=0.5):
    self.delta = delta
    self.proc = proc

  def __call__(self, img, ann=None):
    boo = tf.random.uniform([], 0., 1., dtype=tf.float32)
    return tf.cond(tf.less(boo, self.proc), lambda: tf.image.random_brightness(img, self.delta), lambda: img)
  
  
class RandomSaturation(object):
  def __init__(self, proc=0.5):
    self.proc = proc

  def __call__(self, img, ann=None):
    boo = tf.random.uniform([], 0., 1., dtype=tf.float32)
    return tf.cond(tf.less(boo, self.proc), lambda: tf.image.random_saturation(img, 0.5, 1.5), lambda: img)
  
  
class RandomHue(object):
  def __init__(self, proc=0.5):
    self.proc = proc

  def __call__(self, img, ann=None):
    boo = tf.random.uniform([], 0., 1., dtype=tf.float32)
    return tf.cond(tf.less(boo, self.proc), lambda: tf.image.random_hue(img, 0.2), lambda: img)
  
  
class RandomBlur(object):
  def __init__(self, kernel_size=5):
    han_dist = tf.contrib.signal.hamming_window(kernel_size)
    han_dist = tf.matmul(tf.expand_dims(han_dist, 1), tf.expand_dims(han_dist, 0))
    han_dist = han_dist / tf.reduce_max(han_dist)
    han_dist = tf.expand_dims(tf.expand_dims(han_dist, -1), -1)
    self.han_dist = tf.tile(han_dist, [1, 1, 3, 1])

  def __call__(self, img):
    if len(img.get_shape().as_list()) < 4:
      img = tf.expand_dims(img, 0)
    img = tf.cond(tf.equal(tf.random_uniform([], 0, 1, dtype=tf.int32), 1),
                  lambda: tf.nn.depthwise_conv2d(img, self.han_dist, strides=[1, 1, 1, 1], padding='SAME'),
                  lambda: img)
    if len(img.get_shape().as_list()) < 4:
      return img[0]
    else:
      return img
