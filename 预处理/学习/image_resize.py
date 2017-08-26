#####################coding=utf-8######################################
####################### 图片裁剪 #######################################
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

    # tensorflow offer four ways to resize image
    # method:
    #  0 双线性插值法
    #  test 最近邻居法
    #  test1 双三次插值法
    #  3 面积插值法
    resized = tf.image.resize_images(img_data, [1000, 1000], method=tf.image.ResizeMethod.BILINEAR)

    plt.imshow(resized.eval())
    plt.show()

    resized2 = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)
    resized3 = tf.image.resize_image_with_crop_or_pad(img_data, 3000, 3000)
    #比例缩放
    central_cropped = tf.image.central_crop(img_data, 0.5)

    #裁剪
    img = tf.image.crop_to_bounding_box(img_data, 40, 35, 110, 70)
    plt.imshow(resized2.eval())
    plt.show()
    plt.imshow(resized3.eval())
    plt.show()
