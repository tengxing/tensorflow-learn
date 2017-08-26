#####################coding=utf-8######################################
####################### 数据集读取 #######################################
####################### created by tengxing on 2017.3 #################

import tensorflow as tf

# 获取文件列表
files = tf.train.match_filenames_once("tmp/data*")

# 输入文件队列
filenaem_quene = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filenaem_quene)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64)
    }

)
a = [features['i']]
print a

with tf.Session() as sess:

    tf.initialize_all_variables().run()


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(5):
        print sess.run(a)
        print sess.run([features['i'], features['j']])
    coord.request_stop()
    coord.join(threads)
