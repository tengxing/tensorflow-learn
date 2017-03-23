#####################coding=utf-8######################################
####################### 数据集制作 #######################################
####################### created by tengxing on 2017.3 #################

import tensorflow as tf

#定义类型函数
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#
num_shards = 2
instances_per_shard = 2
for i in range(num_shards):
    #
    filename = ('tmp/data.tfrecords-%.5d-of-%.5d' % (i, num_shards))
    writer = tf.python_io.TFRecordWriter(filename)
    for j in range(instances_per_shard):
        example = tf.train.Example(features=tf.train.Features(feature={
            'i': _int64_feature(i),
            'j': _int64_feature(j)
        }))

        writer.write(example.SerializeToString())
    writer.close()