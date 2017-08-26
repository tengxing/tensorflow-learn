#####################coding=utf-8#########################################################
####对于这个例子的理解：先将images内的数据写入内存文件，再定义模型，读取文件转换为float32的矩阵#######
####################### created by tengxing on 2017.3 ####################################
####################### github：github.com/tengxing ######################################
# ========================================================================================

import tensorflow as tf

import sys
reload(sys)
sys.setdefaultencoding('utf8')


files = tf.train.match_filenames_once("output.tfrecords")
filename_queue = tf.train.string_input_producer(["output.tfrecords"])

#读取文件。

reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)

# 解析读取的样例。
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw':tf.FixedLenFeature([], tf.string),
        'pixels':tf.FixedLenFeature([], tf.int64),
        'label':tf.FixedLenFeature([], tf.int64)
    })

decoded_images = tf.decode_raw(features['image_raw'], tf.uint8)
retyped_images = tf.cast(decoded_images, tf.float32)
labels = tf.cast(features['label'], tf.int32)
#pixels = tf.cast(features['pixels'],tf.int32)
images = tf.reshape(retyped_images, [784])

print images, labels


min_after_dequeue = 1000
batch_size = 5
capacity = min_after_dequeue + 3 * batch_size

image_batch, label_batch = tf.train.shuffle_batch([images, labels],
                                                    batch_size=batch_size,
                                                    capacity=capacity,
                                                    min_after_dequeue=min_after_dequeue)
# 初始化所有的op
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    # 启动队列
    threads = tf.train.start_queue_runners(sess=sess)
    # print image_batch.shape, label_batch
    val, l = sess.run([image_batch, label_batch])
    # l = to_categorical(l, 12)
    print(val, l)
    print "load mnist tfrecords file successful!"
