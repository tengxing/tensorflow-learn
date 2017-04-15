# -*- coding:utf-8 -*-
import random
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from image_pooling import *
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_image_caches(image_lists,images_dir):
    result = {}
    for label_name, label_lists in image_lists.items():
        categorylist = {}
        for category in ['training', 'testing', 'validation']:
            category_list = label_lists[category]
            tmp = []
            for index, unused_base_name in enumerate(category_list):
                image_path = get_image_path(image_lists,label_name,
                                            index,images_dir,category)
                if not gfile.Exists(image_path):
                    tf.logging.fatal('File does not exist %s', image_path)
                #image_data = gfile.FastGFile(image_path, 'rb').read()
                image = cv2.imread(image_path)
                image = cv2.resize(image, (512, 512))
                image = fast_pooling(image)
                tmp.append(image)
                #image_raw = image.tobytes()  # 将图片转化为原生bytes
                #example = tf.train.Example(
                #    features=tf.train.Features(feature={
                #        'image_raw': _bytes_feature(image_raw),
                #        'label': _int64_feature()
                #    })
                #)
            categorylist[category] = tmp
        result[label_name] = categorylist
    return result



def get_image_path(image_lists, label_name, index, image_dir, category):
  """"Returns a path to an image for a label at the given index.

  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Int offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of set to pull images from - training, testing, or
    validation.

  Returns:
    File system path string to an image that meets the requested parameters.

  """
  if label_name not in image_lists:
    tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal('Category does not exist %s.', category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal('Label %s has no images in the category %s.',
                     label_name, category)
  mod_index = index % len(category_list)
  base_name = category_list[mod_index]
  sub_dir = label_lists['dir']
  full_path = os.path.join(image_dir, sub_dir, base_name)
  return full_path


def get_train_image_batch(image_lists,batch_size):
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    for i in range(batch_size):
        label_index = random.randrange(class_count)
        label_name = list(image_lists.keys())[label_index]
        image_name_list = image_lists[label_name]
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        img = random.sample(image_name_list["training"], 1)#only training
        img = np.squeeze(img)  # qu []
        bottlenecks.append(img)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths