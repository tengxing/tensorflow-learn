#####################coding=utf-8######################################
####################### 队列（先进先出） #######################################
####################### created by tengxing on 2017.3 #################
import tensorflow as tf
import  matplotlib.pyplot as plt

# test1 个 flot32
q = tf.FIFOQueue(2, 'int32')
tf.RandomShuffleQueue0

init = q.enqueue_many(([0, 10],))

# 得到第一个
x = q.dequeue()

y = x + 1
# 入队列
q_inc = q.enqueue([y])

with tf.Session() as sess:
    init.run()
    for _ in range(5):
        v, _ = sess.run([x, q_inc])
        print v
