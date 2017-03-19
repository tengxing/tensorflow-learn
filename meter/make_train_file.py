# encoding=utf-8
import os
import tensorflow as tf
from PIL import Image
import random
import numpy as np
from tensorflow.python.platform import gfile
import tensorflow.python.debug as tf_debug
from tensorflow.python.framework import graph_util
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

cwd = os.getcwd()

# 分类目录
classes = {'1', '2'}


# 制作二进制数据
def create_record():
    # 标签
    labels = {}
    writer = tf.python_io.TFRecordWriter("train.tfrecords")
    for index, name in enumerate(classes):
        class_path = cwd +"/"+ name+"/"
        labels[name] = index;
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((1024, 1024))
            img_raw = img.tobytes()  # 将图片转化为原生bytes
            # print index,img_raw
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                        'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                    }
                )
            )
            writer.write(example.SerializeToString())
    writer.close()
    print len(labels)
    with gfile.FastGFile(os.path.join("graph", 'label.txt'), 'w') as f:
        f.write('\n'.join(labels.keys())+'\n')
create_record()


