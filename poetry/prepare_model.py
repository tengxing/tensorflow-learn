#-*- coding:utf-8 -*-
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

class poetryModel:
    #定义权重和偏置项
    def rnn_variable(self,rnn_size,words_size):
        with tf.variable_scope('variable'):
            w = tf.get_variable("w", [rnn_size, words_size])
            b = tf.get_variable("b", [words_size])
        return w,b

    #损失函数
    def loss_model(self,words_size,targets,logits):
        targets = tf.reshape(targets,[-1])
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)],words_size)
        loss = tf.reduce_mean(loss)
        return loss

    #优化算子
    def optimizer_model(self,loss,learning_rate):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 5)
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, tvars))
        return optimizer

    #每个字向量化
    def embedding_variable(self,inputs,rnn_size,words_size):
        with tf.variable_scope('embedding'):
            with tf.device("/cpu:0"):
                embedding = tf.get_variable('embedding', [words_size, rnn_size])
                input_data = tf.nn.embedding_lookup(embedding,inputs)
        return input_data

    #构建LSTM模型
    def create_model(self,inputs,batch_size,rnn_size,words_size,num_layers,is_training,keep_prob):
        lstm = rnn.BasicLSTMCell(num_units=rnn_size,state_is_tuple=True)
        input_data = self.embedding_variable(inputs,rnn_size,words_size)
        if is_training:
            lstm = rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
            input_data = tf.nn.dropout(input_data,keep_prob)
        cell = rnn.MultiRNNCell([lstm] * num_layers,state_is_tuple=True)
        initial_state = cell.zero_state(batch_size, tf.float32)
        outputs,last_state = tf.nn.dynamic_rnn(cell,input_data,initial_state=initial_state)
        outputs = tf.reshape(outputs,[-1, rnn_size])
        w,b = self.rnn_variable(rnn_size,words_size)
        logits = tf.matmul(outputs,w) + b
        probs = tf.nn.softmax(logits)
        return logits,probs,initial_state,last_state