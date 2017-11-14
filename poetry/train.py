#-*- coding:utf-8 -*-
from prepare_data import Poetry
from prepare_model import poetryModel
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    batch_size = 50
    epoch = 20
    rnn_size = 128
    num_layers = 2
    poetrys = Poetry()
    words_size = len(poetrys.word_to_id)
    inputs = tf.placeholder(tf.int32, [batch_size, None])
    targets = tf.placeholder(tf.int32, [batch_size, None])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    model = poetryModel()
    logits,probs,initial_state,last_state = model.create_model(inputs,batch_size,
                                                               rnn_size,words_size,num_layers,True,keep_prob)
    loss = model.loss_model(words_size,targets,logits)
    learning_rate = tf.Variable(0.0, trainable=False)
    optimizer = model.optimizer_model(loss,learning_rate)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.assign(learning_rate, 0.002 * 0.97 ))
        next_state = sess.run(initial_state)
        step = 0
        while True:
            x_batch,y_batch = poetrys.next_batch(batch_size)
            feed = {inputs:x_batch,targets:y_batch,initial_state:next_state,keep_prob:0.5}
            train_loss, _ ,next_state = sess.run([loss,optimizer,last_state], feed_dict=feed)
            print("step:%d loss:%f" % (step,train_loss))
            if step > 40000:
                break
            if step%1000 == 0:
                n = step/1000
                sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** n)))
            step += 1
        saver.save(sess,"poetry_model.ckpt")