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
            img = img.resize((3000, 3000))
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




# 读取二进制数据
def read_and_decode(filename):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }
    )
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [1024, 1024, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(label, tf.int32)
    return img, label

if __name__ == '__main__':
    if 0:
        create_record()
    else:

        #create_record()
        #  g = tf.Graph()

        with tf.Graph().as_default() as g:
        # with tf.GraphDef() as g:

         #Placeholders for inputs and labels.
         with tf.name_scope('input'):
            images_ph = tf.placeholder(tf.float32, [None, 1024, 1024, 3], name="images_ph")
            print images_ph
            labels_ph = tf.placeholder(tf.int32, [None], name="labels_ph")

            # Flatten input from: [None, height, width, channels]
            # To: [None, height * width * channels] == [None, 3072]
            images_flat = tf.contrib.layers.flatten(images_ph)
            print images_flat
            # Fully connected layer.
            # Generates logits of size [None, 62]
            logits = tf.contrib.layers.fully_connected(images_flat, 2, tf.nn.relu)
            print logits
            # Convert logits to label indexes (int).
            # Shape [None], which is a 1D vector of length == batch_size.
         with tf.variable_scope('Accuracy'):
            predicted_labels = tf.argmax(logits, 1, name="predicted_labels")
            # Define the loss function.
            # Cross-entropy is a good choice for classification.
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels_ph))

            # Create training op.
            train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
           # merged_summary_op = tf.merge_all_summaries()
           # summary_writer = tf.train.SummaryWriter('/tmp/mnist_logs', sess.graph)

            img, label = read_and_decode("train.tfrecords")
            #print "tengxing",img,label
            #使用shuffle_batch可以随机打乱输入 next_batch挨着往下取
            # shuffle_batch才能实现[img,label]的同步,也即特征和label的同步,不然可能输入的特征和label不匹配
            # 比如只有这样使用,才能使img和label一一对应,每次提取一个image和对应的label
            # shuffle_batch返回的值就是RandomShuffleQueue.dequeue_many()的结果
            # Shuffle_batch构建了一个RandomShuffleQueue，并不断地把单个的[img,label],送入队列中
            img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                        batch_size=10, capacity=2000,
                                                        min_after_dequeue=1000)

            # 初始化所有的op
            init = tf.global_variables_initializer()

            #
            tf.summary.scalar("loss", loss)
            #tf.scalar_summary("accuracy", accuracy)
            # Merge all summaries to a single operator
            merged_summary_op = tf.summary.merge_all()

            with tf.Session() as sess:
                sess.run(init)

                # load the data that last saved
                if gfile.Exists("data/checkpoint"):
                    try:
                        saver = tf.train.Saver(tf.global_variables())  # 'Saver' misnomer! Better: Persister!
                    except:
                        pass
                    print("load data")
                    saver.restore(sess, "./data/data.ckpt")
                #tensorboard 可以观看
                summary_writer = tf.summary.FileWriter("logs/", sess.graph)
                # 1.0不提供支持
                # summary_writer = tf.train.SummaryWriter('logs/tensorflowlogs', graph_def=sess.graph)

                # 启动队列
                threads = tf.train.start_queue_runners(sess=sess)
                for i in range(100):
                    print "tengxing"
                    val, l = sess.run([img_batch, label_batch])
                    # l = to_categorical(l, 12)
                    #print(val.shape, l)
                    loss_value,_= sess.run(
                        [loss, train],
                        feed_dict={images_ph: val, labels_ph: l})
                    if i % 10 == 0:
                        print("Loss: ", loss_value)
                    #print "---------结果-----------"
                    #print "预测值：{4}".format(predicted)
                    #print "标签值：{4}".format(l)
                val, l = sess.run([img_batch, label_batch])
                #print(val.shape, l)
                # l = to_categorical(l, 12)
                # print(val.shape, l[4],l[1],l[2],l[3],l[4])
                sample_indexes = random.sample(range(len(val)), 5)
                sample_images = [val[i] for i in sample_indexes]
                sample_labels = [l[i] for i in sample_indexes]

                #test_img = np.array(sample_images)
                #test_labels =np.array(sample_labels, dtype=np.int32)
                #print test_img.shape,test_labels

                #loss_value, predicted = sess.run(
                 #   [loss, predicted_labels],
                  #  feed_dict={images_ph: test_img, labels_ph: test_labels})

                # change the type of int64 to int32
                predicted = tf.cast(predicted_labels, tf.int32)
                #print tf.cast(predicted_labels, tf.int32),labels_ph
                correct_prediction = tf.equal(predicted, labels_ph)#预测值与label进行比较
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print "训练图片识别：{0}".format(accuracy.eval({images_ph:sample_images,labels_ph:sample_labels}))
                        #a = accuracy.eval({images_ph: sample_images, labels_ph: sample_labels})
                #print '最终的测试正确率：{4}'.format(accuracy)
                #print '最终的测试正确率：{4}'.format(predicted)


                # save the variables on disk
                variables = tf.global_variables()
                saver = tf.train.Saver(variables)
                saver.save(sess, "data/data.ckpt")

                # save the model to file and wo can use it predict sth like images
                # tf.train.write_graph(sess.graph_def, 'graph', 'model.ph', False)

                '''
                node_seq = {}  # Keyed by node name.
                seq = 4
                for node in g.as_graph_def().node:

                    seq += 1
                print g.as_graph_def().node[22]
                print seq
                '''
                #a ="Accuracy/predicted_labels".split('.')

                # Write out the trained graph and labels with the weights stored as constants.
                output_graph_def = graph_util.convert_variables_to_constants(
                    sess, g.as_graph_def(), ['Accuracy/predicted_labels'])
                #graph is not be sess.graph must be sess.graph.def
                with gfile.FastGFile(os.path.join("graph", 'classify_image_graph_def.ph'), 'wb') as f:
                    f.write(output_graph_def.SerializeToString())


