#####################coding=utf-8#########################################################
####################### tfrecords制作工具类 ################################################
####################### created by tengxing on 2017.3 ####################################
####################### github：github.com/tengxing ######################################
###########################################################################################

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os

import sys
reload(sys)
sys.setdefaultencoding('utf8')

cwd = os.getcwd()

#参数设置
#############################################################################################
train_dir = {'dataset/train/test', 'dataset/train/2'} #训练图片文件夹
filename='train.tfrecords'    #生成train.tfrecords
output_directory='tmp' #输出文件夹
resize_height=64 #存储图片高度
resize_width=64 #存储图片宽度
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


def transform2tfrecord(train_dir, file_name, output_directory, resize_height, resize_width):
    if not os.path.exists(output_directory) or os.path.isfile(output_directory):
        os.makedirs(output_directory)
    for index, name in enumerate(train_dir):
        class_path = cwd + "/" + name + "/"
        filename = output_directory+"/"+file_name+('-%.5d' % (index))
        writer = tf.python_io.TFRecordWriter(filename)
        for image_name in os.listdir(class_path):
            image_path = class_path + image_name
            #image = Image.open(image_path)
            #image = image.resize((resize_height, resize_width))
            print index
            image = cv2.imread(image_path)
            #image = cv2.resize(image, (resize_height, resize_width))
            b, g, r = cv2.split(image)
            image = cv2.merge([r, g, b])
            image_raw = image.tobytes()  # 将图片转化为原生bytes
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    'image_raw': _bytes_feature(image_raw),
                    'height': _int64_feature(image.shape[0]),
                    'width': _int64_feature(image.shape[1]),
                    'depth': _int64_feature(image.shape[2]),
                    'label': _int64_feature(index)
                })
            )
            writer.write(example.SerializeToString())
        writer.close()


def read_tfrecord(file_dir, filename):

    # 文件夹下所有文件
    wildfiles = file_dir+"/"+filename+"*"
    # 获取文件列表
    files = tf.train.match_filenames_once(wildfiles)

    # 输入文件队列
    filename_queue = tf.train.string_input_producer(files, shuffle=False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'width': tf.FixedLenFeature([], tf.int64),
          'height': tf.FixedLenFeature([], tf.int64),
          'depth': tf.FixedLenFeature([], tf.int64),
          'label': tf.FixedLenFeature([], tf.int64)
        }
    )

    encoded_image = tf.decode_raw(features['image_raw'], tf.uint8)
    #encoded_image.set_shape([features['height'], features['width'], features['depth']])
    # image
    encoded_image = tf.reshape(encoded_image, [resize_height*resize_width,3])
    # normalize
    image = tf.cast(encoded_image, tf.float32) * (1. / 255) - 0.5
    # label
    label = tf.cast(features['label'], tf.int32)
    return image, label






def disp_tfrecords(file_dir, filename):
    # 文件夹下所有文件
    wildfiles = file_dir + "/" + filename + "*"
    # 获取文件列表
    files = tf.train.match_filenames_once(wildfiles)

    # 输入文件队列
    filename_queue = tf.train.string_input_producer(files, shuffle=False)

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
            #print image_eval
            resultLabel.append(label.eval())
            image_eval_reshape = image_eval.reshape([height.eval(), width.eval(), depth.eval()])
            #print image_eval_reshape
            resultImg.append(image_eval_reshape)
            pilimg = Image.fromarray(np.asarray(image_eval_reshape))
            pilimg.show()
        coord.request_stop()
        coord.join(threads)
        sess.close()
    return resultImg, resultLabel

#img, label = disp_tfrecords(output_directory, filename)

#transform2tfrecord(train_dir, filename, output_directory, resize_height, resize_width)
#img, label = read_tfrecord(output_directory, filename) #读取函数