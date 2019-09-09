#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2019 WR Tan     National Tsing Hua University
#
# Distributed under terms of the MIT license.

"""Save the paths of crops from the GOT-10k dataset in pickle format"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import os.path as osp
import pickle
import sys
import numpy as np
import tensorflow as tf

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..'))

from utils.misc_utils import sort_nicely


class Config:
  ### Dataset
  # directory where curated dataset is stored
  dataset_dir = '/store_ssd/got10k-Curation'
  save_dir = 'data/'


class DataIter:
  """Container for dataset of one iteration"""
  pass


class Dataset:
  def __init__(self, config):
    self.config = config

  def dataset_iterator(self, video_dirs):
    anno_dir = '/store_ssd/got10k'
    video_num = len(video_dirs)
    iter_size = 150
    iter_num = int(np.ceil(video_num / float(iter_size)))
    for iter_ in range(iter_num):
      iter_start = iter_ * iter_size
      iter_videos = video_dirs[iter_start: iter_start + iter_size]

      data_iter = DataIter()
      num_videos = len(iter_videos)
      instance_videos = []
      instance_ann = []
      for index in range(num_videos):
        print('Processing {}/{}...'.format(iter_start + index, video_num))
        video_dir = iter_videos[index]
        
        subdirs = video_dir.split('/')
        ann_vid_dir = osp.join(anno_dir, subdirs[-2], subdirs[-1])
        with open(osp.join(ann_vid_dir, 'groundtruth.txt')) as f:
          ann_content = f.readlines()        

        instance_image_paths = glob.glob(video_dir + '/*.jpg.jpg')

        # sort image paths by frame number
        instance_image_paths = sort_nicely(instance_image_paths)
        
        assert len(instance_image_paths) == len(ann_content)
        
        try:
          # get annotations
          bboxs = []
          for ann in ann_content:
            ann = ann.strip()
            bbox = ann.split(',')
            bbox = [float(bb) for bb in bbox]  # [xmin, ymin, w, h]
            
            w = bbox[2]
            h = bbox[3]
            context_amount = 0.5
            size_z=127
            size_x=255
            
            wc_z = w + context_amount * (w + h)
            hc_z = h + context_amount * (w + h)
            s_z = np.sqrt(wc_z * hc_z)
            scale_z = size_z / s_z
            
            d_search = (size_x - size_z) / 2
            pad = d_search / scale_z
            s_x = s_z + 2 * pad
            
            wn = w / s_x * size_x * 2
            hn = h / s_x * size_x * 2
            
            if wn < 1 or hn < 1:
              ratio = wn / hn
              if ratio > 1.:
                bboxs.append([128., 128., 85., 85. / ratio])
              else:
                bboxs.append([128., 128., 85. * ratio, 85.])              
            else:
              bboxs.append([128, 128, wn, hn])
            
          # get image absolute path
          #instance_image_paths = [os.path.abspath(p) for p in instance_image_paths]
          instance_videos.append(instance_image_paths)
          instance_ann.append(bboxs)
        except:
          print('Skipped: ' + video_dir)
      data_iter.num_videos = len(instance_videos)
      data_iter.instance_videos = instance_videos
      data_iter.instance_ann = instance_ann
      yield data_iter

  def get_all_video_dirs(self, dirtype):
    ann_dir = os.path.join(self.config.dataset_dir, dirtype)
    train_dirs = os.listdir(ann_dir)
    all_video_dirs = [os.path.join(ann_dir, name) for name in train_dirs if '.txt' not in name]
    return all_video_dirs


def main():
  # Get the data.
  config = Config()
  dataset = Dataset(config)
  
  ### validation
  validation_dirs = dataset.get_all_video_dirs('val')
  validation_imdb = dict()
  validation_imdb['videos'] = []
  validation_imdb['ann'] = []
  for i, data_iter in enumerate(dataset.dataset_iterator(validation_dirs)):
    print(data_iter.instance_videos)
    validation_imdb['videos'] += data_iter.instance_videos
    validation_imdb['ann'] += data_iter.instance_ann
  validation_imdb['n_videos'] = len(validation_imdb['videos'])
  validation_imdb['image_shape'] = (255, 255, 3)

  ### train
  train_dirs = dataset.get_all_video_dirs('train')  
  train_imdb = dict()
  train_imdb['videos'] = []
  train_imdb['ann'] = []
  for i, data_iter in enumerate(dataset.dataset_iterator(train_dirs)):
    train_imdb['videos'] += data_iter.instance_videos
    train_imdb['ann'] += data_iter.instance_ann
  train_imdb['n_videos'] = len(train_imdb['videos'])
  train_imdb['image_shape'] = (255, 255, 3)

  if not tf.gfile.IsDirectory(config.save_dir):
    tf.logging.info('Creating training directory: %s', config.save_dir)
    tf.gfile.MakeDirs(config.save_dir)

  with open(os.path.join(config.save_dir, 'validation_imdb.pickle'), 'wb') as f:
    pickle.dump(validation_imdb, f)
  with open(os.path.join(config.save_dir, 'train_imdb.pickle'), 'wb') as f:
    pickle.dump(train_imdb, f)


if __name__ == '__main__':
  main()
