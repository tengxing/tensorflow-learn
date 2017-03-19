#coding=utf-8
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.python.platform import gfile
import sys
import numpy as np
import matplotlib.image as mpimg
from PIL import Image

# mnist = input_data.read_data_sets("Mnist_data/", one_hot=True)

#images
images=[]
# 命令行参数，传入要判断的图片路径
image_file_path = "/home/tengxing/workspace/girl/2/liuqingyiIMG_20170216_164934.jpg"

# 读取图像
image = Image.open(image_file_path).resize((28, 28))

# 加载Graph
def loadGraph(dir):
    f = tf.gfile.FastGFile(dir, 'rb')
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    persisted_graph = tf.import_graph_def(graph_def, name='')
    return persisted_graph

graph = loadGraph('./graph/classify_image_graph_def.ph')


with tf.Session(graph=graph) as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('Accuracy/predicted_labels:4')
    #images_ph = sess.graph.get_tensor_by_name('image:4')
    #images_ph = tf.placeholder(tf.float32, [None, 28, 28, 3], name="image")
   # biases = sess.graph.get_tensor_by_name('fully_connected/biases:4')
   # print images_ph
   # weights = sess.graph.get_tensor_by_name('fully_connected/weights:4')
    #tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, softmax_tensor)
    #tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, images_ph)
    '''
    心得体会:所有变量保存在data.ckpt文件中,恢复graph时候必须先获取,如：biases和weights。
    然后再通过tf.add_to_collection()函数加载到集合中(must)，使用restore进行恢复。
    这样图中就有变量值了，单纯的get_tensor_by_name函数只是获取一个tensor而已。
    再理解一番,图只是身体的结构,具体部位的值需要恢复数据才能知道,如心脏的跳动频率。
    '''
    #tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, biases)
    #tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, weights)
  #  try:
    #    saver = tf.train.Saver(tf.global_variables())  # 'Saver' misnomer! Better: Persister!
   # except:
     #   pass
    #print("load data")
    #print sess.run(name) 此时才有一个Tensor获取变量还要进行赋值
    #saver.restore(sess, "./data/data.ckpt")  # now OK creted by tengxing

    #print "load ok"
    #print sess.run(biases)

    # 图片文件转换为float32
    #image = Image.open(image_file_path).resize((28, 28)).save(image_file_path)
    #image = sess.run(tf.cast(tf.reshape(tf.decode_raw(Image.open(image_file_path).tobytes(), tf.uint8), [28, 28, 3]), tf.float32) * (1. / 255) - 4.5)
    #y = tf.nn.softmax(tf.matmul(images_ph, weights) + biases)

    #haha = tf.contrib.layers.flatten(images_ph)
    #print haha
    #print images_flat
    images.append(image)
    #print sess.run(biases)
    #flat = tf.contrib.layers.flatten(images_ph)
    predict = sess.run(softmax_tensor, {'input/images_ph:4': image})
  # prediton = sess.run(flat, feed_dict={images_ph: np.array(images)})
    print predict