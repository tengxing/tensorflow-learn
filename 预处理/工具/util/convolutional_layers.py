import tensorflow as tf
from convolutional_model import *
from PIL import Image

WINDOW_SHAPE = (64, 128)


def a():
    """
    Get the convolutional layers of the model.

    """
    x = tf.placeholder(tf.float32, [None, None, None])
    image = Image.open("0.jpg")
    image = image.tobytes()
    image = tf.decode_raw(image,tf.uint8)
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    print
    sess = tf.InteractiveSession()
    x = sess.run(image)
    # First layer
    W_conv1 = weight_variable([5, 5, 1, 48])
    b_conv1 = bias_variable([48])
    x_expanded = tf.expand_dims(x, 0)
    x_image = tf.reshape(x_expanded, [-1, 1024, 768, 1])
    print x_image
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool(h_conv1, ksize=(4, 4), stride=(4, 4))
    print h_conv1,h_pool1
    # Second layer
    W_conv2 = weight_variable([5, 5, 48, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool(h_conv2, ksize=(2, 1), stride=(2, 2))
    print h_conv2, h_pool2
    # Third layer
    W_conv3 = weight_variable([5, 5, 64, 96])
    b_conv3 = bias_variable([96])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool(h_conv3, ksize=(4, 4), stride=(4, 4))
    print h_conv3, h_pool3

    # fourth layer
    W_conv4 = weight_variable([5, 5, 96, 128])
    b_conv4 = bias_variable([128])

    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool(h_conv4, ksize=(4, 2), stride=(4, 4))
    print h_conv4, h_pool4

    # densely connected layer
    w_fc4 = weight_variable([8 * 6 * 128, 2048])
    b_fc4 = bias_variable([2048])

    h_pool4_flat = tf.reshape(h_pool4, [-1, 8 * 6 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, w_fc4) + b_fc4)
    return h_fc1


h_fc1 = a()
# readout layer
w_fc2 = weight_variable([2048, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)
print y_conv
