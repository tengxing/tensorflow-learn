# -*- coding:utf-8 -*-
# 快速pool
import tensorflow as tf
from convolutional_model import *


def fast_pooling(image):
    #image = image.tobytes()
    #image = tf.decode_raw(image, tf.uint8)
    #image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    #image = tf.reshape(image,[512,512, 3])
    #print image
    x_image = tf.reshape(tf.cast(image, tf.float32) * (1. / 255) - 0.5, [-1, 512, 512, 3])
    #y_image = tf.reshape(x_image, [512,512,3])
    #print x_image.shape
    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    x = max_pool(h_conv1, ksize=(10, 10), stride=(8, 8))
    x1 = max_pool(x, ksize=(3, 3), stride=(2, 2))
    x2 = max_pool(x1, ksize=(3, 3), stride=(2, 2))
    x3 = max_pool(x2, ksize=(3, 3), stride=(2, 2))

    img = tf.reshape(x3, [2048])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(img)