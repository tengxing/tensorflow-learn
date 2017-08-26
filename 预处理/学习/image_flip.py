#####################coding=utf-8######################################
####################### 图片旋转 #######################################
####################### created by tengxing on 2017.3 #################

import tensorflow as tf
import matplotlib.pyplot as plt

# read the test
image_raw_data = tf.gfile.FastGFile("test/cat.jpg", 'r').read()

with tf.Session() as sess:
    # 解码得到三维矩阵
    # img_data是一个张量
    img_data = tf.image.decode_jpeg(image_raw_data)

    # convert为实数，方便读取
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)


    resized = tf.image.flip_up_down(img_data)
    resized1 = tf.image.flip_left_right(img_data)
    #对角线
    resized2 = tf.image.transpose_image(img_data)
    plt.imshow(resized.eval())
    plt.show()
    resized3 = tf.image.random_flip_left_right(img_data, 0.5)

    plt.imshow(resized1.eval())
    plt.show()
    plt.imshow(resized2.eval())
    plt.show()
    plt.imshow(resized3.eval())
    plt.show()
