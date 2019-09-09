#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 WR Tan     National Tsing Hua University
#
# Distributed under terms of the MIT license.

"""Model Wrapper class for performing inference with a SiameseModel"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import logging
import os
import os.path as osp

import numpy as np
import tensorflow as tf

from embeddings.convolutional_alexnet_multi_fnonlinear import convolutional_alexnet_arg_scope
from embeddings.convolutional_alexnet_multi_fnonlinear import convolutional_alexnet
from utils.infer_utils import get_exemplar_images
from utils.misc_utils import get_center

slim = tf.contrib.slim


class InferenceWrapper():
  """Model wrapper class for performing inference with a siamese model."""
  def __init__(self):
    self.image = None
    self.target_bbox_feed = None
    self.search_images = None
    self.embeds = None
    self.templates = None
    self.init = None
    self.model_config = None
    self.track_config = None
    self.response_up = None

  def build_graph_from_config(self, model_config, track_config, checkpoint_path):
    """Build the inference graph and return a restore function."""
    self.build_model(model_config, track_config)
    ema = tf.train.ExponentialMovingAverage(0)
    variables_to_restore = ema.variables_to_restore(moving_avg_variables=[])
    
    # Filter out State variables
    variables_to_restore_filterd = {}
    for key, value in variables_to_restore.items():
      if key.split('/')[1] != 'State':
        variables_to_restore_filterd[key] = value

    saver = tf.train.Saver(variables_to_restore_filterd)

    if osp.isdir(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
      if not checkpoint_path:
        raise ValueError("No checkpoint file found in: {}".format(checkpoint_path))

    def _restore_fn(sess):
      logging.info("Loading model from checkpoint: %s", checkpoint_path)
      saver.restore(sess, checkpoint_path)
      logging.info("Successfully loaded checkpoint: %s", os.path.basename(checkpoint_path))

    return _restore_fn

  def build_model(self, model_config, track_config):
    self.model_config = model_config
    self.track_config = track_config

    self.build_inputs()
    self.build_search_images()
    self.build_template()
    self.build_detection()
    self.build_upsample()
    self.dumb_op = tf.no_op('dumb_operation')

  def build_inputs(self):
    filename = tf.placeholder(tf.string, [], name='filename')
    image_file = tf.read_file(filename)
    image = tf.image.decode_jpeg(image_file, channels=3, dct_method="INTEGER_ACCURATE")
    image = tf.to_float(image)
    self.image = image
    self.target_bbox_feed = tf.placeholder(dtype=tf.float32,
                                           shape=[4],
                                           name='target_bbox_feed')  # center's y, x, height, width  
    self.size_x_feed = tf.placeholder(dtype=tf.int32,
                                      shape=[],
                                      name='size_x_feed')
    self.scale_feed = tf.placeholder(dtype=tf.float32,
                                     shape=[3],
                                     name='scale_feed')      
    
  def build_search_images(self):
    """Crop search images from the input image based on the last target position
    1. The input image is scaled such that the area of target&context takes up to (scale_factor * z_image_size) ^ 2
    2. Crop an image patch as large as x_image_size centered at the target center.
    3. If the cropped image region is beyond the boundary of the input image, mean values are padded.
    """
    model_config = self.model_config
    track_config = self.track_config
    ratio = self.target_bbox_feed[2] / self.target_bbox_feed[3]  

    size_z = model_config['z_image_size']
    size_x = self.size_x_feed
    context_amount = 0.3
    
    num_scales = track_config['num_scales']
    scales = np.arange(num_scales) - get_center(num_scales)
    assert np.sum(scales) == 0, 'scales should be symmetric'
    search_factors = tf.split(self.scale_feed, 3)

    frame_sz = tf.shape(self.image)
    target_yx = self.target_bbox_feed[0:2]
    target_size = self.target_bbox_feed[2:4]
    avg_chan = tf.reduce_mean(self.image, axis=(0, 1), name='avg_chan')
    
    # Compute base values
    base_z_size = target_size
    base_z_context_size = base_z_size + context_amount * tf.reduce_sum(base_z_size)
    base_s_z = tf.sqrt(tf.reduce_prod(base_z_context_size))  # Canonical size
    base_scale_z = tf.div(tf.to_float(size_z), base_s_z)
    d_search = tf.div(tf.to_float(size_x) - tf.to_float(size_z), 2.0)
    base_pad = tf.div(d_search, base_scale_z)
    base_s_x = base_s_z + 2 * base_pad
    base_scale_x = tf.div(tf.to_float(size_x), base_s_x)
    
    boxes = []
    for factor in search_factors:
      s_x = factor * base_s_x
      frame_sz_1 = tf.to_float(frame_sz[0:2] - 1)
      topleft = tf.div(target_yx - get_center(s_x), frame_sz_1)
      bottomright = tf.div(target_yx + get_center(s_x), frame_sz_1)
      box = tf.concat([topleft, bottomright], axis=0)
      boxes.append(box)
    boxes = tf.stack(boxes)
    
    self.target_size = target_size * 127. / base_s_z    
    
    scale_xs = []
    for factor in search_factors:
      scale_x = base_scale_x / factor
      scale_xs.append(scale_x)
    self.scale_xs = tf.stack(scale_xs)

    # Pad with average value of the image
    image_minus_avg = tf.expand_dims(self.image - avg_chan, 0)
    image_cropped = tf.image.crop_and_resize(image_minus_avg, boxes,
                                             box_ind=tf.zeros((track_config['num_scales']), tf.int32),
                                             crop_size=[size_x, size_x])
    self.search_images = image_cropped + avg_chan
    
  def get_image_embedding(self, images, reuse=None):
    config = self.model_config['embed_config']
    self.arg_scope = convolutional_alexnet_arg_scope(config,
                                                     trainable=config['train_embedding'],
                                                     is_training=False)

    @functools.wraps(convolutional_alexnet)
    def embedding_fn(images, reuse=False):
      with slim.arg_scope(self.arg_scope):
        return convolutional_alexnet(images, reuse=reuse)
    
    im = images
    outputs = embedding_fn(im, reuse)
    return outputs[0], outputs[1]
  
  def build_template(self):
    model_config = self.model_config
    track_config = self.track_config
    size_z = model_config['z_image_size']
    ratio = self.target_size[0] / self.target_size[1]
    
    # Exemplar image lies at the center of the search image in the first frame
    center_scale = int(get_center(track_config['num_scales']))
    search_images = tf.expand_dims(self.search_images[center_scale], 0)
    exemplar_images = get_exemplar_images(search_images, [size_z, size_z], 
                                          np.array([[get_center(track_config['x_image_size']), 
                                                     get_center(track_config['x_image_size'])]]))
    
    def boundary_suppression(embeds, embeds2, ratio):
      offsets = tf.cond(tf.greater(ratio, 1.5), 
                        lambda: [0, 4, 0, 4], 
                        lambda: tf.cond(tf.less(ratio, 0.67), lambda: [4, 0, 4, 0], 
                                        lambda: [2, 2, 2, 2]))
      embeds = tf.image.resize_image_with_crop_or_pad(embeds, t_shape[1]-offsets[0], 
                                                      t_shape[2]-offsets[1])
      embeds = tf.image.resize_image_with_crop_or_pad(embeds, t_shape[1], t_shape[2])
      embeds2 = tf.image.resize_image_with_crop_or_pad(embeds2, t_shape2[1]-offsets[2], 
                                                       t_shape2[2]-offsets[3])
      embeds2 = tf.image.resize_image_with_crop_or_pad(embeds2, t_shape2[1], t_shape2[2])
      return embeds, embeds2
    
    def background_suppression(embeds, ratio):
      offsets = tf.cond(tf.greater(ratio, 1.5),  # 1.2 / 0.83; 1.5 / 0.67
                        lambda: [1., 1.2 / ratio], 
                        lambda: tf.cond(tf.less(ratio, 0.67), 
                                        lambda: [1.2 * ratio, 1.], 
                                        lambda: tf.cond(tf.greater(ratio, 1.2), 
                                                        lambda: [1., 1.1 / ratio], 
                                                        lambda: tf.cond(tf.less(ratio, 0.83), 
                                                                        lambda: [1.1 * ratio, 1.], 
                                                                        lambda: [0.7, 0.7]))))
      h = tf.cast(size_z * offsets[0], tf.int32)
      w = tf.cast(size_z * offsets[1], tf.int32)
      embeds_mean = tf.reduce_mean(embeds, axis=(0, 1), keepdims=True)
      embeds = embeds - embeds_mean
      embeds = tf.image.resize_image_with_crop_or_pad(embeds, h, w)
      embeds = tf.image.resize_image_with_crop_or_pad(embeds, size_z, size_z)
      return embeds + embeds_mean
    
    exemplar_images = tf.map_fn(lambda x: background_suppression(x[0], x[1]),
                                (exemplar_images, tf.expand_dims(ratio, 0)),
                                dtype=exemplar_images.dtype)
    self.exemplar_images = exemplar_images
    
    templates, templates2 = self.get_image_embedding(exemplar_images)
    t_shape = templates.get_shape().as_list()
    t_shape2 = templates2.get_shape().as_list()
    
    templates, templates2 = tf.map_fn(lambda x: boundary_suppression(x[0], x[1], x[2]),
                                      (templates, templates2, tf.expand_dims(ratio, 0)),
                                      dtype=(templates.dtype, templates2.dtype))
    
    with tf.variable_scope('target_template'):
      # Store template in Variable such that we don't have to feed this template every time.
      with tf.variable_scope('State'):
        state = tf.get_variable('exemplar',
                                initializer=tf.zeros(templates.get_shape().as_list(), 
                                                     dtype=templates.dtype),
                                trainable=False)
        state2 = tf.get_variable('exemplar2',
                                 initializer=tf.zeros(templates2.get_shape().as_list(), 
                                                      dtype=templates2.dtype),
                                 trainable=False)
        with tf.control_dependencies([templates, templates2]):
          self.init = tf.assign(state, templates, validate_shape=True)
          self.init2 = tf.assign(state2, templates2, validate_shape=True)
        
        self.templates = state
        self.templates2 = state2
        
        # Store Pseudo Templates
        def _euc_distance(x, z):
          z = tf.expand_dims(z, 0)
          return tf.reduce_sum(tf.abs(x - z), -1)
        
        n_mem = 3  # 3, 5
        state_pseu = []
        state_pseu2 = []
        image_pseu = []
        self.init_pseu = []
        self.init2_pseu = []
        self.init_pseu_img = []
        for i in range(n_mem):
          state_pseu.append(tf.get_variable('exemplar_pseu'+str(i),
                                            initializer=tf.zeros(templates.get_shape().as_list(), 
                                                                 dtype=templates.dtype),
                                            trainable=False))
          state_pseu2.append(tf.get_variable('exemplar2_pseu'+str(i),
                                             initializer=tf.zeros(templates2.get_shape().as_list(), 
                                                                  dtype=templates2.dtype),
                                             trainable=False))
          image_pseu.append(tf.get_variable('exemplar_pseu_image'+str(i),
                                            initializer=tf.zeros(exemplar_images.get_shape().as_list(), 
                                                                 dtype=exemplar_images.dtype),
                                            trainable=False))
          with tf.control_dependencies([templates, templates2, exemplar_images]):
            self.init_pseu.append(tf.assign(state_pseu[i], templates, validate_shape=True))
            self.init2_pseu.append(tf.assign(state_pseu2[i], templates2, validate_shape=True))
            self.init_pseu_img.append(tf.assign(image_pseu[i], exemplar_images, validate_shape=True))
        
        self.image_pseu = image_pseu
        self.pseu_temp = state_pseu
        self.pseu_temp2 = state_pseu2
        
        state_pseus = tf.concat([self.templates] + state_pseu + [templates], 0)
        sp_shape = state_pseus.get_shape().as_list()[0]
        state_pseus_c = tf.reshape(state_pseus, [sp_shape, -1])
        state_pseus_dis = tf.map_fn(
          lambda x: _euc_distance(state_pseus_c, x),
          state_pseus_c, dtype=state_pseus_c.dtype)
        state_pseus_dis = tf.reshape(state_pseus_dis, [sp_shape, sp_shape])[1:, :]
        state_pseus_dis = tf.reduce_sum(state_pseus_dis, -1)
        self.state_pseus_dis = state_pseus_dis
        _, state_pseus_idx = tf.nn.top_k(state_pseus_dis, k=len(state_pseu))
        
        image_pseu_extra = tf.concat(image_pseu + [exemplar_images], 0)
        state_pseus2 = tf.concat(state_pseu2 + [templates2], 0)
        self.up_img = []
        self.up_pseu = []
        self.up2_pseu = []
        for i in range(len(state_pseu)):
          with tf.control_dependencies([state_pseus_idx, image_pseu_extra, state_pseus, state_pseus2]):
            self.up_pseu.append(tf.assign(state_pseu[i], tf.expand_dims(state_pseus[state_pseus_idx[i]+1], 0), 
                                          validate_shape=True))
            self.up2_pseu.append(tf.assign(state_pseu2[i], tf.expand_dims(state_pseus2[state_pseus_idx[i]], 0), 
                                           validate_shape=True))
            self.up_img.append(tf.assign(image_pseu[i], tf.expand_dims(image_pseu_extra[state_pseus_idx[i]], 0), 
                                         validate_shape=True))
        
  def build_detection(self):
    self.embeds, self.embeds2 = self.get_image_embedding(self.search_images, reuse=True)
    alpha = 0.5
    
    with tf.variable_scope('detection'):
      def _translation_match(x, z):  # translation match for one example within a batch
        x = tf.expand_dims(x, 0)  # [1, in_height, in_width, in_channels]
        z = tf.expand_dims(z, -1)  # [filter_height, filter_width, in_channels, 1]
        y = tf.nn.conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID', name='translation_match')
        return y
      
      with tf.variable_scope('patch_matching'):
        template = tf.add_n(self.pseu_temp) / (len(self.pseu_temp))
        template = alpha * self.templates + (1. - alpha) * template
        
        embeds = self.embeds
        output = tf.map_fn(lambda x: _translation_match(x, template[0]),
                           embeds,
                           dtype=embeds.dtype)        
        output = tf.squeeze(output, [1])  # of shape e.g., [b, h, w, n]
        
      with tf.variable_scope('patch_matching2'):
        template2 = tf.add_n(self.pseu_temp2) / (len(self.pseu_temp2))
        template2 = alpha * self.templates2 + (1. - alpha) * template2
        
        embeds2 = self.embeds2
        output2 = tf.map_fn(lambda x: _translation_match(x, template2[0]),
                                  embeds2,
                                  dtype=embeds2.dtype)
        output2 = tf.squeeze(output2, [1])  # of shape e.g., [b, h, w, n]
        
      with slim.arg_scope(self.arg_scope):
        output = slim.batch_norm(output, scope='bn_patch')
        output2 = slim.batch_norm(output2, scope='bn_patch2')
        
      outputF_all = tf.squeeze(output * 0.5 + output2 * 0.5, -1)
      self.response = outputF_all
      
  def build_upsample(self):
    """Upsample response to obtain finer target position"""
    with tf.variable_scope('upsample'):
      responseF = self.response
      response = tf.expand_dims(responseF, 3)
      up_method = self.track_config['upsample_method']
      methods = {'bilinear': tf.image.ResizeMethod.BILINEAR,
                 'bicubic': tf.image.ResizeMethod.BICUBIC}
      up_method = methods[up_method]
      response_spatial_size = tf.shape(responseF)[1:3]
      up_size = response_spatial_size * self.track_config['upsample_factor']
      response_up = tf.image.resize_images(response,
                                           up_size,
                                           method=up_method,
                                           align_corners=True)
      response_up = tf.squeeze(response_up, [3])
      self.response_up = response_up

  def initialize(self, sess, input_feed):
    image_path, target_bbox, size_x, scale_x = input_feed
    outputs = sess.run([self.scale_xs, self.exemplar_images,
                        self.init, self.init2]
                       + self.init_pseu + self.init2_pseu + self.init_pseu_img,
                       feed_dict={'filename:0': image_path,
                                  "target_bbox_feed:0": target_bbox, 
                                  "size_x_feed:0": size_x,
                                  "scale_feed:0": scale_x})
    scale_xs = outputs[0]
    return scale_xs, outputs[1][0]
  
  def update(self, sess, input_feed):
    image_path, target_bbox, size_x, scale_x = input_feed
    outputs = sess.run(self.up_pseu + self.up2_pseu + self.up_img,
                       feed_dict={'filename:0': image_path,
                                  "target_bbox_feed:0": target_bbox, 
                                  "size_x_feed:0": size_x,
                                  "scale_feed:0": scale_x})

  def inference_step(self, sess, input_feed):
    image_path, target_bbox, size_x, scale_x = input_feed
    log_level = self.track_config['log_level']
    image_cropped_op = self.search_images if log_level > 0 else self.dumb_op
    image_cropped, scale_xs, response_output, embed, embed2 = sess.run(
      fetches=[image_cropped_op, self.scale_xs, self.response_up, self.templates, self.templates2],
      feed_dict={
        "filename:0": image_path,
        "target_bbox_feed:0": target_bbox, 
        "size_x_feed:0": size_x,
        "scale_feed:0": scale_x
      })
    output = {
      'image_cropped': image_cropped,
      'scale_xs': scale_xs,
      'response': response_output,
      'embed': embed,
      'embed2': embed2
    }
    return output, None
