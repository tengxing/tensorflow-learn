import numpy as np
import os.path
import urllib.request
import sys
import tarfile
from queue import Queue
from threading import Thread

from tensorflow.python.platform import gfile
import tensorflow as tf

from constants import *

def get_image_list(image_dir, category, partition):
    """ Compute list of images.
    """
    file_glob = os.path.join(image_dir, category, partition, '*.jpeg')
    file_list = gfile.Glob(file_glob)
    return file_list

image_queue = Queue()

def worker():
    while not image_queue.empty():
        (sess, category_bottleneck_dir, img, bottleneck_tensor, jpeg_data_tensor) = image_queue.get()
#         print(img)
        #create each bottleneck
        cache_image(sess, category_bottleneck_dir, img, bottleneck_tensor, jpeg_data_tensor)
        image_queue.task_done()

def cache_category(sess, image_dir, category, partition, bottleneck_dir, bottleneck_tensor, jpeg_data_tensor):
    #get list of image to calculate bottleneck
    image_list = get_image_list(image_dir, category, partition)
    print('{} images considered to calculate bottleneck in category: {} partition: {}'.format(len(image_list), category, partition))

    category_bottleneck_dir = '{}/{}/{}'.format(bottleneck_dir,category, partition)
    #create dir for bottlenecks if it doesn't exist
    os.makedirs(category_bottleneck_dir, exist_ok=True)

    #enqueue each image
    for img_path in image_list:
        image_queue.put((sess, category_bottleneck_dir, img_path, bottleneck_tensor, jpeg_data_tensor))

    for i in range(NUM_WORKER_THREAD):
        t = Thread(target=worker)
        t.daemon = True
        t.start()

    image_queue.join()    # block until all tasks are done
    print("Cache category: {} partition: {} completed".format(category, partition))


def cache_image(sess, category_bottleneck_dir, image_path, bottleneck_tensor, jpeg_data_tensor):
    """
        Calculate and save bottleneck of one image only if it hasn't been
        computed before.
    """
    image_name = os.path.basename(image_path)
    bottleneck_path = '{}/{}.txt'.format(category_bottleneck_dir,image_name)

    #if bottleneck hasn't been calculated before
    if not os.path.exists(bottleneck_path):
        print('Creating bottleneck at ' + bottleneck_path)

        #check that image exist
        if not gfile.Exists(image_path):
            tf.logging.fatal('File does not exist %s', image_path)

        image_data = gfile.FastGFile(image_path, 'rb').read()
        
        try:
            bottleneck_values = run_bottleneck_on_image(sess, image_data,
                                                jpeg_data_tensor,
                                                bottleneck_tensor)
            bottleneck_string = ','.join(str(x) for x in bottleneck_values)

            with open(bottleneck_path, 'w') as bottleneck_file:
                bottleneck_file.write(bottleneck_string)
        except:
            print('error procesing image {}'.format(image_path))

def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
  """Runs inference on an image to extract the 'bottleneck' summary layer.

  Args:
    sess: Current active TensorFlow Session.
    image_data: String of raw JPEG data.
    image_data_tensor: Input data layer in the graph.
    bottleneck_tensor: Layer before the final softmax.

  Returns:
    Numpy array of bottleneck values.
  """
  bottleneck_values = sess.run(
      bottleneck_tensor,
      {image_data_tensor: image_data})
  bottleneck_values = np.squeeze(bottleneck_values)
  return bottleneck_values

def create_inception_graph(model_dir):
  """"Creates a graph from saved GraphDef file and returns a Graph object.

  Returns:
    Graph holding the trained Inception network, and various tensors we'll be
    manipulating.
  """
  BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
  JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
  RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'

  with tf.Session() as sess:
    model_filename = os.path.join(
        model_dir, 'classify_image_graph_def.pb')

    with gfile.FastGFile(model_filename, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
          tf.import_graph_def(graph_def, name='', return_elements=[
              BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,
              RESIZED_INPUT_TENSOR_NAME]))
  return sess.graph, bottleneck_tensor, jpeg_data_tensor, resized_input_tensor

def maybe_download_and_extract(model_dir):
  DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
  """Download and extract model tar file.

  If the pretrained model we're using doesn't already exist, this function
  downloads it from the TensorFlow.org website and unpacks it into a directory.
  """
  dest_directory = model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):

    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' %
                       (filename,
                        float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(DATA_URL,
                                             filepath,
                                             _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def start_async_bottleneck_cache():
    """Calculate bottleneck of all cropped images.
    """
    maybe_download_and_extract(INCEPTION_MODEL_DIR)

    session = tf.Session()
    graph, bottleneck_tensor, jpeg_data_tensor, resized_image_tensor = create_inception_graph(INCEPTION_MODEL_DIR)

    for category in categories:
        for partition in partitions:
            cache_category(session, BASE_CROP_DIRECTORY, category, partition,BASE_BOTTLENECK_DIRECTORY, bottleneck_tensor, jpeg_data_tensor)


NUM_WORKER_THREAD = 4

if __name__ == '__main__':
    start_async_bottleneck_cache()
