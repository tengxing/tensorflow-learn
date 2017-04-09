import tensorflow as tf
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
def max_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='SAME')

def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                        strides=[1, stride[0], stride[1], 1], padding='SAME')


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def conv2d(x, W, stride=(1, 1), padding='SAME'):
  return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1],
                      padding=padding)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def fast_pool(image):
   # print image.shape
    image = image.tobytes()
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    x_image = tf.reshape(image, [-1, 512, 512, 3])
    #x_expanded = tf.expand_dims(x_image, 0)


    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    print h_conv1
    x = max_pool(h_conv1, ksize=(10, 10), stride=(8, 8))
    x1 = max_pool(x, ksize=(3, 3), stride=(2, 2))
    x2 = max_pool(x1, ksize=(3, 3), stride=(2, 2))
    x3 = max_pool(x2, ksize=(3, 3), stride=(2, 2))
    print x3
    img = tf.reshape(x3, [2048])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(img)
        return sess.run(img)

def fast_pool1(image):
   # print image.shape
    image = image.tobytes()
    image = tf.decode_raw(image, tf.uint8)
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    x_image = tf.reshape(image, [-1, 512, 512, 3])
    #x_expanded = tf.expand_dims(x_image, 0)


    W_conv1 = weight_variable([5, 5, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    print h_conv1
    x = max_pool(h_conv1, ksize=(10, 10), stride=(8, 8))
    x1 = max_pool(x, ksize=(3, 3), stride=(2, 2))
    x2 = max_pool(x1, ksize=(3, 3), stride=(2, 2))
    x3 = max_pool(x2, ksize=(3, 3), stride=(2, 2))
    print x3
    img = tf.reshape(x3, [2048])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        return sess.run(img)





