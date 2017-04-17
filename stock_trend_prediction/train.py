# -*- coding:utf-8 -*-
import tensorflow as tf
import pickle
import numpy as np
import ast
from collections import defaultdict

train_data = 'train.vec'
valid_data = 'valid.vec'

word2idx, content_length, question_length, vocab_size = pickle.load(open('vocab.data', "rb"))
print(content_length, question_length, vocab_size)

batch_size = 64

train_file = open(train_data)


def get_next_batch():
    X = []
    Q = []
    A = []
    for i in range(batch_size):
        for line in train_file:
            line = ast.literal_eval(line.strip())
            X.append(line[0])
            Q.append(line[1])
            A.append(line[2][0])
            break

    if len(X) == batch_size:
        return X, Q, A
    else:
        train_file.seek(0)
        return get_next_batch()


def get_test_batch():
    with open(valid_data) as f:
        X = []
        Q = []
        A = []
        for line in f:
            line = ast.literal_eval(line.strip())
            X.append(line[0])
            Q.append(line[1])
            A.append(line[2][0])
        return X, Q, A


X = tf.placeholder(tf.int32, [batch_size, content_length])  # 洋文材料
Q = tf.placeholder(tf.int32, [batch_size, question_length])  # 问题
A = tf.placeholder(tf.int32, [batch_size])  # 答案

# drop out
keep_prob = tf.placeholder(tf.float32)


def glimpse(weights, bias, encodings, inputs):
    weights = tf.nn.dropout(weights, keep_prob)
    inputs = tf.nn.dropout(inputs, keep_prob)
    attention = tf.transpose(tf.matmul(weights, tf.transpose(inputs)) + bias)
    attention = tf.matmul(encodings, tf.expand_dims(attention, -1))
    attention = tf.nn.softmax(tf.squeeze(attention, -1))
    return attention, tf.reduce_sum(tf.expand_dims(attention, -1) * encodings, 1)


def neural_attention(embedding_dim=384, encoding_dim=128):
    embeddings = tf.Variable(tf.random_normal([vocab_size, embedding_dim], stddev=0.22), dtype=tf.float32)
    tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-4), [embeddings])

    with tf.variable_scope('encode'):
        with tf.variable_scope('X'):
            X_lens = tf.reduce_sum(tf.sign(tf.abs(X)), 1)
            embedded_X = tf.nn.embedding_lookup(embeddings, X)
            encoded_X = tf.nn.dropout(embedded_X, keep_prob)
            gru_cell = tf.contrib.rnn.core_rnn_cell.GRUCell(encoding_dim)
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(gru_cell, gru_cell, encoded_X,
                                                                     sequence_length=X_lens, dtype=tf.float32,
                                                                     swap_memory=True)
            encoded_X = tf.concat(outputs, 2)
        with tf.variable_scope('Q'):
            Q_lens = tf.reduce_sum(tf.sign(tf.abs(Q)), 1)
            embedded_Q = tf.nn.embedding_lookup(embeddings, Q)
            encoded_Q = tf.nn.dropout(embedded_Q, keep_prob)
            gru_cell = tf.contrib.rnn.core_rnn_cell.GRUCell(encoding_dim)
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(gru_cell, gru_cell, encoded_Q,
                                                                     sequence_length=Q_lens, dtype=tf.float32,
                                                                     swap_memory=True)
            encoded_Q = tf.concat(outputs, 2)

    W_q = tf.Variable(tf.random_normal([2 * encoding_dim, 4 * encoding_dim], stddev=0.22), dtype=tf.float32)
    b_q = tf.Variable(tf.random_normal([2 * encoding_dim, 1], stddev=0.22), dtype=tf.float32)
    W_d = tf.Variable(tf.random_normal([2 * encoding_dim, 6 * encoding_dim], stddev=0.22), dtype=tf.float32)
    b_d = tf.Variable(tf.random_normal([2 * encoding_dim, 1], stddev=0.22), dtype=tf.float32)
    g_q = tf.Variable(tf.random_normal([10 * encoding_dim, 2 * encoding_dim], stddev=0.22), dtype=tf.float32)
    g_d = tf.Variable(tf.random_normal([10 * encoding_dim, 2 * encoding_dim], stddev=0.22), dtype=tf.float32)

    with tf.variable_scope('attend') as scope:
        infer_gru = tf.contrib.rnn.core_rnn_cell.GRUCell(4 * encoding_dim)
        infer_state = infer_gru.zero_state(batch_size, tf.float32)
        for iter_step in range(8):
            if iter_step > 0:
                scope.reuse_variables()

            _, q_glimpse = glimpse(W_q, b_q, encoded_Q, infer_state)
            d_attention, d_glimpse = glimpse(W_d, b_d, encoded_X, tf.concat([infer_state, q_glimpse], 1 ))

            gate_concat = tf.concat([infer_state, q_glimpse, d_glimpse, q_glimpse * d_glimpse], 1)

            r_d = tf.sigmoid(tf.matmul(gate_concat, g_d))
            r_d = tf.nn.dropout(r_d, keep_prob)
            r_q = tf.sigmoid(tf.matmul(gate_concat, g_q))
            r_q = tf.nn.dropout(r_q, keep_prob)

            combined_gated_glimpse = tf.concat([r_q * q_glimpse, r_d * d_glimpse], 1)
            _, infer_state = infer_gru(combined_gated_glimpse, infer_state)

    return tf.to_float(tf.sign(tf.abs(X))) * d_attention


def train_neural_attention():
    X_attentions = neural_attention()
    loss = -tf.reduce_mean(
        tf.log(tf.reduce_sum(tf.to_float(tf.equal(tf.expand_dims(A, -1), X)) * X_attentions, 1) + tf.constant(0.00001)))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    grads_and_vars = optimizer.compute_gradients(loss)
    capped_grads_and_vars = [(tf.clip_by_norm(g, 5), v) for g, v in grads_and_vars]
    train_op = optimizer.apply_gradients(capped_grads_and_vars)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # writer = tf.summary.FileWriter()
        # 恢复前一次训练
        ckpt = tf.train.get_checkpoint_state('.')
        if ckpt != None:
            print(ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("没找到模型")

        for step in range(20000):
            train_x, train_q, train_a = get_next_batch()
            loss_, _ = sess.run([loss, train_op], feed_dict={X: train_x, Q: train_q, A: train_a, keep_prob: 0.7})
            print(loss_)

            # 保存模型并计算准确率
            if step % 1000 == 0:
                path = saver.save(sess, 'machine_reading.model', global_step=step)
                print(path)

                test_x, test_q, test_a = get_test_batch()
                test_x, test_q, test_a = np.array(test_x[:batch_size]), np.array(test_q[:batch_size]), np.array(
                    test_a[:batch_size])
                attentions = sess.run(X_attentions, feed_dict={X: test_x, Q: test_q, keep_prob: 1.})
                correct_count = 0
                for x in range(test_x.shape[0]):
                    probs = defaultdict(int)
                    for idx, word in enumerate(test_x[x, :]):
                        probs[word] += attentions[x, idx]
                    guess = max(probs, key=probs.get)
                    if guess == test_a[x]:
                        correct_count += 1
                print(correct_count / test_x.shape[0])


train_neural_attention()