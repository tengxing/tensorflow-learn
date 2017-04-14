#i want to use coding=UTF-8
import tensorflow as tf
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
import numpy as np
import random
def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
  bottleneck_values = sess.run(
      bottleneck_tensor,
      {image_data_tensor: image_data})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values


def getpool3(sess,data_tensor,labels,step,bottleneck_tensor,image_data_tensor):
    print data_tensor,labels
    for i in range(step):
        image_data = sess.run(
            data_tensor,

        )
        print image_data
        #train_img = run_bottleneck_on_image(sess,image_data,image_data_tensor,bottleneck_tensor)
        #print train_img
    return None,None

#把值转化为集合(one,0)
def get_data_batch(sess, img_batch,label_batch):
    input_img_batch, input_label_batch = sess.run([img_batch, label_batch])
    input_label_batches = []
    for i in range(len(input_label_batch)):
        ground_truth = np.zeros(2, dtype=np.float32)
        ground_truth[input_label_batch[i]] = 1.0
        input_label_batches.append(ground_truth)
    return (input_img_batch, input_label_batches)
def get_data_batch1(data,class_count):
    bottlenecks = []
    ground_truths = []
    for i in range(50):
        label_index = random.randrange(class_count)
        ground_truth = np.zeros(class_count, dtype=np.float32)
        ground_truth[label_index] = 1.0
        img = random.sample(data[label_index], 1)
        img = np.squeeze(img) #qu []
        bottlenecks.append(img)
        ground_truths.append(ground_truth)
    return bottlenecks,ground_truths