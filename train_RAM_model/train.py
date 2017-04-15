# -*- coding: utf-8 -*-
import sys
import argparse
from datetime import datetime
from tfrecords_util import *
from change import *
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

from pre_data import *
from train_model import *
from suf_data import *



# 全局
BOTTLENECK_SIZE = 2048
TRAIN_SPEP_SIZE = 4000
EVAL_STEP_INTERVAR = 10
TRAIN_BATCH_SIZE = 50

def create_train_data():
    print


def main(_):
    print '初始化'
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    print '第一步:加载数据到内存'
    # Look at the folder structure, and create lists of all the images.
    image_lists = create_image_lists(FLAGS.image_dir, FLAGS.testing_percentage,
                                     FLAGS.validation_percentage)
    # class_count
    class_count = len(image_lists.keys())

    # load images to cache
    image_caches = create_image_caches(image_lists,FLAGS.image_dir)
    #print "all:",(image_caches['one'])
    print '第二步:开启全局sess'
    sess = tf.Session()

    print '第三步:训练(测试)模型'
    # Add the new layer that we'll be training.
    (train_step, cross_entropy, bottleneck_input, ground_truth_input,
     final_tensor) = add_final_training_ops(class_count,
                                            FLAGS.final_tensor_name,
                                            BOTTLENECK_SIZE)
    # print bottleneck_input,ground_truth_input,cross_entropy,train_step,final_tensor
    # Create the operations we need to evaluate the accuracy of our new layer.
    evaluation_step, prediction = add_evaluation_step(
        final_tensor, ground_truth_input)
    print prediction

    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                         sess.graph)
    # init all variables
    init = tf.global_variables_initializer()
    sess.run(init)
    # Run the training for as many cycles as requested on the command line.
    for i in range(TRAIN_SPEP_SIZE):
        # Get a batch of input bottleneck values adn input_label from the cache stored on ram not disk.
        input_img, input_label = get_train_image_batch(image_caches, TRAIN_BATCH_SIZE)
        #print input_img[0],input_label
        # Feed the bottlenecks and ground truth into the graph, and run a training
        # step. Capture training summaries for TensorBoard with the `merged` op.
        train_summary, _ = sess.run([merged, train_step],
                                    feed_dict={bottleneck_input: input_img,
                                               ground_truth_input: input_label})
        train_writer.add_summary(train_summary, i)

        # Every so often, print out how well the graph is training.
        is_last_step = (i + 1 == TRAIN_SPEP_SIZE)
        if (i % EVAL_STEP_INTERVAR) == 0 or is_last_step:
            train_accuracy, cross_entropy_value = sess.run(
                [evaluation_step, cross_entropy],
                feed_dict={bottleneck_input: input_img,
                           ground_truth_input: input_label})
            print('%s: Step %d: Train accuracy = %.1f%%' % (datetime.now(), i,
                                                            train_accuracy * 100))
            print('%s: Step %d: Cross entropy = %f' % (datetime.now(), i,
                                                       cross_entropy_value))

    print '第四步:保存模型'
    # Write out the trained graph and labels with the weights stored as constants.
    output_graph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph.as_graph_def(), [FLAGS.final_tensor_name])
    with gfile.FastGFile(FLAGS.output_graph, 'wb') as f:
        f.write(output_graph_def.SerializeToString())
    with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
        f.write('\n'.join(image_lists.keys()) + '\n')

    print '训练完毕'




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_dir',
        type=str,
        default='images',
        help='Path to folders of labeled images.'
    )
    parser.add_argument(
        '--output_graph',
        type=str,
        default='tmp/output_graph.pb',
        help='Where to save the trained graph.'
    )
    parser.add_argument(
        '--output_labels',
        type=str,
        default='tmp/output_labels.txt',
        help='Where to save the trained graph\'s labels.'
    )
    parser.add_argument(
      '--summaries_dir',
      type=str,
      default='tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.'
    )
    parser.add_argument(
        '--final_tensor_name',
        type=str,
        default='final_result',
        help="""\
          The name of the output classification layer in the retrained graph.\
          """
    )
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a test set.'
    )
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=10,
        help='What percentage of images to use as a validation set.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)