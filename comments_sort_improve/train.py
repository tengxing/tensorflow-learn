import os
import random
import tensorflow as tf
import pickle
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

f = open('lexcion.pickle', 'rb')
lex = pickle.load(f)
f.close()


def get_random_line(file, point):
    file.seek(point)
    file.readline()
    return file.readline()


# 从文件中随机选择n条记录
def get_n_random_line(file_name, n=150):
    lines = []
    file = open(file_name, encoding='latin-1')
    total_bytes = os.stat(file_name).st_size
    for i in range(n):
        random_point = random.randint(0, total_bytes)
        lines.append(get_random_line(file, random_point))
    file.close()
    return lines


def get_test_dataset(test_file):
    with open(test_file, encoding='latin-1') as f:
        test_x = []
        test_y = []
        lemmatizer = WordNetLemmatizer()
        for line in f:
            label = line.split(':%:%:%:')[0]
            tweet = line.split(':%:%:%:')[1]
            words = word_tokenize(tweet.lower())
            words = [lemmatizer.lemmatize(word) for word in words]
            features = np.zeros(len(lex))
            for word in words:
                if word in lex:
                    features[lex.index(word)] = 1

            test_x.append(list(features))
            test_y.append(eval(label))
    return test_x, test_y


test_x, test_y = get_test_dataset('tesing.csv')

#######################################################################

n_input_layer = len(lex)  # 输入层

n_layer_1 = 2000  # hide layer
n_layer_2 = 2000  # hide layer(隐藏层)听着很神秘，其实就是除输入输出层外的中间层

n_output_layer = 3  # 输出层


def neural_network(data):
    # 定义第一层"神经元"的权重和biases
    layer_1_w_b = {'w_': tf.Variable(tf.random_normal([n_input_layer, n_layer_1])),
                   'b_': tf.Variable(tf.random_normal([n_layer_1]))}
    # 定义第二层"神经元"的权重和biases
    layer_2_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_1, n_layer_2])),
                   'b_': tf.Variable(tf.random_normal([n_layer_2]))}
    # 定义输出层"神经元"的权重和biases
    layer_output_w_b = {'w_': tf.Variable(tf.random_normal([n_layer_2, n_output_layer])),
                        'b_': tf.Variable(tf.random_normal([n_output_layer]))}

    # w·x+b
    layer_1 = tf.add(tf.matmul(data, layer_1_w_b['w_']), layer_1_w_b['b_'])
    layer_1 = tf.nn.relu(layer_1)  # 激活函数
    layer_2 = tf.add(tf.matmul(layer_1, layer_2_w_b['w_']), layer_2_w_b['b_'])
    layer_2 = tf.nn.relu(layer_2)  # 激活函数
    layer_output = tf.add(tf.matmul(layer_2, layer_output_w_b['w_']), layer_output_w_b['b_'])

    return layer_output


X = tf.placeholder('float')
Y = tf.placeholder('float')
batch_size = 90


def train_neural_network(X, Y):
    predict = neural_network(X)
    cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(predict, Y))
    optimizer = tf.train.AdamOptimizer().minimize(cost_func)

    with tf.Session() as session:
        session.run(tf.initialize_all_variables())

        lemmatizer = WordNetLemmatizer()
        saver = tf.train.Saver()
        i = 0
        pre_accuracy = 0
        while True:  # 一直训练
            batch_x = []
            batch_y = []

            # if model.ckpt文件已存在:
            #	saver.restore(session, 'model.ckpt')  恢复保存的session

            try:
                lines = get_n_random_line('training.csv', batch_size)
                for line in lines:
                    label = line.split(':%:%:%:')[0]
                    tweet = line.split(':%:%:%:')[1]
                    words = word_tokenize(tweet.lower())
                    words = [lemmatizer.lemmatize(word) for word in words]

                    features = np.zeros(len(lex))
                    for word in words:
                        if word in lex:
                            features[lex.index(word)] = 1  # 一个句子中某个词可能出现两次,可以用+=1，其实区别不大

                    batch_x.append(list(features))
                    batch_y.append(eval(label))

                session.run([optimizer, cost_func], feed_dict={X: batch_x, Y: batch_y})
            except Exception as e:
                print(e)

            # 准确率
            if i > 100:
                correct = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                accuracy = accuracy.eval({X: test_x, Y: test_y})
                if accuracy > pre_accuracy:  # 保存准确率最高的训练模型
                    print('准确率: ', accuracy)
                    pre_accuracy = accuracy
                    saver.save(session, 'model.ckpt')  # 保存session
                i = 0
            i += 1


train_neural_network(X, Y)