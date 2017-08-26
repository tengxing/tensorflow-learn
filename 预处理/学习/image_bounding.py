#####################coding=utf-8######################################
####################### 图片标注 #######################################
####################### created by tengxing on 2017.3 #################

import tensorflow as tf
import matplotlib.pyplot as plt

# read the test
image_raw_data = tf.gfile.FastGFile("test/cat.jpg", 'r').read()

with tf.Session() as sess:
    # 解码得到三维矩阵
    # img_data是一个张量
    img_data = tf.image.decode_jpeg(image_raw_data)

    img_data = tf.image.resize_images(img_data, [180, 267], method=1)

    # boxes
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.37, 0.47, 0.5, 0.56]]])

    # bacth
    batched = tf.expand_dims(
        tf.image.convert_image_dtype(img_data, tf.float32), 0
    )

    draw_bound = tf.image.draw_bounding_boxes(batched, boxes=boxes)

    print draw_bound.shape
    #需要减低维度 ，我不会
    #draw_bound = tf.image.convert_image_dtype(draw_bound. )
    #plt.imshow(draw_bound.eval())
    #plt.show()
