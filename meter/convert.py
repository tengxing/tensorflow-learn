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
        for img_name in os.listdir(class_path):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img0 = img.resize((3000, 3000)).save(class_path+"缩放"+img_name)
            #img1 = img.convert('1').save(class_path+"灰度"+img_name)
            #img2 = img.convert('1').resize((128, 128)).save(class_path+"灰缩"+img_name)
create_record()
