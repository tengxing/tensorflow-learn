#####################coding=utf-8######################################
####################### 图片色彩 #######################################
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

    # 亮度
    resized = tf.image.random_brightness(img_data, 0.5)
    # 对比度
    resized1 = tf.image.random_contrast(img_data, 0.1, 5)
    # 色相
    resized2 = tf.image.random_hue(img_data, 0.5)
    # 饱和度
    resized3 = tf.image.random_saturation(img_data, 0.1, 5)

    plt.imshow(resized.eval())
    plt.show()
    plt.imshow(resized1.eval())
    plt.show()
    plt.imshow(resized2.eval())
    plt.show()
    plt.imshow(resized3.eval())
    plt.show()
