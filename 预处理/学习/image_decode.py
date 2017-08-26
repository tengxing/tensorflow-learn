#####################coding=utf-8######################################
####################### 图片编码 #######################################
####################### created by tengxing on 2017.3 #################
import tensorflow as tf
import  matplotlib.pyplot as plt

# read the test
image_raw_data = tf.gfile.FastGFile("images/cat.jpg", 'r').read()

with tf.Session() as sess:
    # 解码得到三维矩阵
    # img_data是一个张量
    img_data = tf.image.decode_jpeg(image_raw_data)

    print img_data.eval()
    #print sess.run(img_data)

    plt.imshow(img_data.eval())
    #如何直接读取图片？
    #plt.imread("test/cat.jpg")
    #plt.show()

    # 还原img
    endoded_image = tf.image.encode_jpeg(img_data)
    with tf.gfile.FastGFile("images/new_cat.jpg", 'wb') as f:
       f.write(endoded_image.eval())

    # convert为实数，方便读取
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)
    #print sess.run(img_data)
    plt.imshow(img_data.eval())
    plt.show()
