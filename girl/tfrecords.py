############################################################################################
# !/usr/bin/python2.7
# coding = utf-8
# Author  : created by tengxing on 2017.2
# Function: image convert to tfrecords
#############################################################################################

import tensorflow as tf
import numpy as np
import cv2
import os
import os.path
from PIL import Image

# 全局参数设置
###############################################################################################
train_file = 'train.txt'  # 训练图片
name = 'train'  # 生成train.tfrecords
output_directory = './tfrecords'
resize_height = 28  # 存储图片高度
resize_width = 28  # 存储图片宽度
classes = {'1','2','3'} #训练图片文件夹


###############################################################################################
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_file(examples_list_file):
    lines = np.genfromtxt(examples_list_file, delimiter=" ", dtype=[('col1', 'S120'), ('col2', 'i8')])
    examples = []
    labels = []
    for example, label in lines:
        examples.append(example)
        labels.append(label)
    return np.asarray(examples), np.asarray(labels), len(lines)


def extract_image(filename, resize_height, resize_width):
    image = cv2.imread(filename)
    image = cv2.resize(image, (resize_height, resize_width))
    b, g, r = cv2.split(image)
    rgb_image = cv2.merge([r, g, b])
    return rgb_image


def transform2tfrecord(train_file, name, output_directory, resize_height, resize_width):
    if not os.path.exists(output_directory) or os.path.isfile(output_directory):
        os.makedirs(output_directory)
    _examples, _labels, examples_num = load_file(train_file)
    filename = output_directory + "/" + name + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(filename)
    for i, [example, label] in enumerate(zip(_examples, _labels)):
        print('No.%d' % (i))
        image = extract_image(example, resize_height, resize_width)
        print('shape: %d, %d, %d, label: %d' % (image.shape[0], image.shape[1], image.shape[2], label))
        image_raw = image.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'height': _int64_feature(image.shape[0]),
            'width': _int64_feature(image.shape[1]),
            'depth': _int64_feature(image.shape[2]),
            'label': _int64_feature(label)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def disp_tfrecords(tfrecord_list_file):
    filename_queue = tf.train.string_input_producer([tfrecord_list_file])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    # print(repr(image))
    height = features['height']
    width = features['width']
    depth = features['depth']
    label = tf.cast(features['label'], tf.int32)
    init_op = tf.initialize_all_variables()
    resultImg = []
    resultLabel = []
    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(21):
            image_eval = image.eval()
            resultLabel.append(label.eval())
            image_eval_reshape = image_eval.reshape([height.eval(), width.eval(), depth.eval()])
            resultImg.append(image_eval_reshape)
            pilimg = Image.fromarray(np.asarray(image_eval_reshape))
            pilimg.show()
        coord.request_stop()
        coord.join(threads)
        sess.close()
    return resultImg, resultLabel


def read_tfrecord(filename_queuetemp):
    filename_queue = tf.train.string_input_producer([filename_queuetemp])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    # image
    tf.reshape(image, [256, 256, 3])
    # normalize
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    # label
    label = tf.cast(features['label'], tf.int32)
    return image, label


def test():
    transform2tfrecord(train_file, name, output_directory, resize_height, resize_width)  # 转化函数
    img, label = disp_tfrecords(output_directory + '/' + name + '.tfrecords')  # 显示函数
    img, label = read_tfrecord(output_directory + '/' + name + '.tfrecords')  # 读取函数
    print label


if __name__ == '__main__':
    test()