#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 WR Tan     National Tsing Hua University
#
# Distributed under terms of the MIT license.

"""Construct the computational graph of siamese model for training. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf

from datasets.dataloader_reg import DataLoader
from embeddings.convolutional_alexnet_multi_fnonlinear import convolutional_alexnet_arg_scope
from embeddings.convolutional_alexnet_multi_fnonlinear import convolutional_alexnet
from metrics.track_metrics import center_dist_error, center_score_error
from utils.train_utils import construct_gt_score_maps, load_mat_model, construct_seg_score_maps

slim = tf.contrib.slim


class SiameseModel:
  def __init__(self, model_config, train_config, mode='train'):
    self.model_config = model_config
    self.train_config = train_config
    self.mode = mode
    assert mode in ['train', 'validation', 'inference']

    if self.mode == 'train':
      self.data_config = self.train_config['train_data_config']
    elif self.mode == 'validation':
      self.data_config = self.train_config['validation_data_config']

    self.dataloader = None
    self.exemplars = None
    self.instances = None
    self.response = None
    self.batch_loss = None
    self.total_loss = None
    self.init_fn = None
    self.global_step = None

  def is_training(self):
    """Returns true if the model is built for training mode"""
    return self.mode == 'train'

  def build_inputs(self):
    """Input fetching and batching [batch, h, w, 3]"""
    with tf.device("/cpu:0"):  # Put data loading and preprocessing in CPU is substantially faster
      self.dataloader = DataLoader(self.data_config, self.is_training())
      self.dataloader.build()
      outputs = self.dataloader.get_one_batch()
      exemplars, instances = outputs[0], outputs[1]
      bbox = [outputs[4], outputs[5]]
      ratio = [outputs[2], outputs[3]]
      exemplars = tf.to_float(exemplars)
      instances = tf.to_float(instances)
    self.exemplars = exemplars
    self.instances = instances
    self.ratio = ratio
    self.bbox = bbox  # [w, h]
    
  def build_image_embeddings(self, reuse=False):
    config = self.model_config['embed_config']
    size_z = self.model_config['z_image_size']
    self.arg_scope = convolutional_alexnet_arg_scope(config,
                                                     trainable=config['train_embedding'],
                                                     is_training=self.is_training())
    
    @functools.wraps(convolutional_alexnet)
    def embedding_fn(images, reuse=False):
      with slim.arg_scope(self.arg_scope):
        return convolutional_alexnet(images, reuse=reuse)
    
    def background_suppression(embeds, ratio):
      h = tf.cast(ratio[1] * 0.6, tf.int32)
      w = tf.cast(ratio[0] * 0.6, tf.int32)
      embeds_mean = tf.reduce_mean(embeds, axis=(0, 1), keepdims=True)
      embeds = embeds - embeds_mean
      embeds = tf.image.resize_image_with_crop_or_pad(embeds, h, w)
      embeds = tf.image.resize_image_with_crop_or_pad(embeds, size_z, size_z)
      return embeds + embeds_mean
    
    exemplars = self.exemplars
    instances = self.instances
    
    exemplars = tf.map_fn(lambda x: background_suppression(x[0], x[1]),
                          (exemplars, self.ratio),
                          dtype=exemplars.dtype)
    self.exemplars_show = exemplars
    
    outputs = embedding_fn(exemplars, reuse=reuse)
    self.exemplar_embeds = outputs[0]
    self.exemplar_embeds2 = outputs[1]
    outputs2 = embedding_fn(instances, reuse=True)
    self.instance_embeds = outputs2[0]
    self.instance_embeds2 = outputs2[1]
    
  def build_template(self):
    t_shape = self.exemplar_embeds.get_shape().as_list()
    t_shape2 = self.exemplar_embeds2.get_shape().as_list()
    
    def boundary_suppression(embeds, embeds2, ratio):
      offsets = tf.cond(tf.greater(ratio, 1.5), 
                        lambda: [0, 4, 0, 4], 
                        lambda: tf.cond(tf.less(ratio, 0.66), lambda: [4, 0, 4, 0], 
                                        lambda: [2, 2, 2, 2]))
      embeds = tf.image.resize_image_with_crop_or_pad(embeds, t_shape[1]-offsets[0], 
                                                      t_shape[2]-offsets[1])
      embeds = tf.image.resize_image_with_crop_or_pad(embeds, t_shape[1], t_shape[2])
      embeds2 = tf.image.resize_image_with_crop_or_pad(embeds2, t_shape2[1]-offsets[2], 
                                                       t_shape2[2]-offsets[3])
      embeds2 = tf.image.resize_image_with_crop_or_pad(embeds2, t_shape2[1], t_shape2[2])
      return embeds, embeds2
    
    exemplar_embeds = self.exemplar_embeds
    exemplar_embeds2 = self.exemplar_embeds2
    ratio = self.ratio[1] / self.ratio[0]
    exemplar_embeds, exemplar_embeds2 = tf.map_fn(lambda x: boundary_suppression(x[0], x[1], x[2]),
                                                  (exemplar_embeds, exemplar_embeds2, ratio),
                                                  dtype=(exemplar_embeds.dtype, exemplar_embeds2.dtype))
    self.templates = exemplar_embeds
    self.templates2 = exemplar_embeds2
    
  def build_detection(self, reuse=False):
    with tf.variable_scope('detection', reuse=reuse):
      def _translation_match(x, z):  # translation match for one example within a batch
        x = tf.expand_dims(x, 0)  # [1, in_height, in_width, in_channels]
        z = tf.expand_dims(z, -1)  # [filter_height, filter_width, in_channels, 1]
        y = tf.nn.conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID', name='translation_match')
        return y
      
      with tf.variable_scope('patch_matching'):
        output = tf.map_fn(lambda x: _translation_match(x[0], x[1]),
                           (self.instance_embeds, self.templates),
                           dtype=self.instance_embeds.dtype)
        output = tf.squeeze(output, [1])  # of shape e.g., [b, h, w, 1]
        
      with tf.variable_scope('patch_matching2'):
        output2 = tf.map_fn(lambda x: _translation_match(x[0], x[1]),
                            (self.instance_embeds2, self.templates2),
                            dtype=self.instance_embeds2.dtype)
        output2 = tf.squeeze(output2, [1])  # of shape e.g., [b, h, w, 1]
        
      with slim.arg_scope(self.arg_scope):
        output = slim.batch_norm(output, scope='bn_patch')
        output2 = slim.batch_norm(output2, scope='bn_patch2')
        
      output = tf.squeeze(output, -1)
      output2 = tf.squeeze(output2, -1)
      self.total_response = output * 0.5 + output2 * 0.5
      self.response = tf.concat([output, output2], 0)

  def build_loss(self):
    response = self.response
    response_size = response.get_shape().as_list()[1:3]
    gt = construct_gt_score_maps(response_size,
                                 self.data_config['batch_size'] * 2,
                                 self.model_config['embed_config']['stride'],
                                 self.train_config['gt_config'])
    
    with tf.name_scope('Loss'):
      loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=response,
                                                     labels=gt)
      
      with tf.name_scope('Balance_weights'):
        n_pos = tf.reduce_sum(tf.to_float(tf.equal(gt[0], 1)))
        n_neg = tf.reduce_sum(tf.to_float(tf.equal(gt[0], 0)))
        w_pos = 0.5 / n_pos
        w_neg = 0.5 / n_neg
        class_weights = tf.where(tf.equal(gt, 1),
                                 w_pos * tf.ones_like(gt),
                                 tf.ones_like(gt))
        class_weights = tf.where(tf.equal(gt, 0),
                                 w_neg * tf.ones_like(gt),
                                 class_weights)
        loss = loss * class_weights
        
      loss = tf.reduce_sum(loss, [1, 2])
      batch_loss = tf.reduce_mean(loss, name='batch_loss')
      tf.losses.add_loss(batch_loss)
      
      # Get total loss
      total_loss = tf.losses.get_total_loss()
      self.batch_loss = batch_loss
      self.total_loss = total_loss

      tf.summary.image('exemplar', self.exemplars_show, family=self.mode)
      tf.summary.image('instance', self.instances, family=self.mode)

      mean_total_loss, update_op2 = tf.metrics.mean(total_loss)
      with tf.control_dependencies([update_op2]):
        tf.summary.scalar('total_loss', mean_total_loss, family=self.mode)

      if self.mode == 'train':
        tf.summary.image('GT', tf.reshape(gt[0], [1] + response_size + [1]), family='GT')
      tf.summary.image('Response', tf.expand_dims(tf.sigmoid(self.total_response), -1), family=self.mode)
      tf.summary.histogram('Response', self.total_response, family=self.mode)

      # Two more metrics to monitor the performance of training
      tf.summary.scalar('center_score_error', center_score_error(self.total_response), family=self.mode)
      tf.summary.scalar('center_dist_error', center_dist_error(self.total_response), family=self.mode)

  def setup_global_step(self):
    global_step = tf.Variable(
      initial_value=0,
      name='global_step',
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
    self.global_step = global_step

  def setup_embedding_initializer(self):
    """Sets up the function to restore embedding variables from checkpoint."""
    embed_config = self.model_config['embed_config']
    if embed_config['embedding_checkpoint_file']:
      # Restore Siamese FC models from .mat model files
      initialize = load_mat_model(embed_config['embedding_checkpoint_file'],
                                  'convolutional_alexnet/', 'detection/')

      def restore_fn(sess):
        tf.logging.info("Restoring embedding variables from checkpoint file %s",
                        embed_config['embedding_checkpoint_file'])
        sess.run([initialize])

      self.init_fn = restore_fn

  def build(self, reuse=False):
    """Creates all ops for training and evaluation"""
    with tf.name_scope(self.mode):
      self.build_inputs()
      self.build_image_embeddings(reuse=reuse)
      self.build_template()
      self.build_detection(reuse=reuse)
      self.setup_embedding_initializer()

      if self.mode in ['train', 'validation']:
        self.build_loss()

      if self.is_training():
        self.setup_global_step()
