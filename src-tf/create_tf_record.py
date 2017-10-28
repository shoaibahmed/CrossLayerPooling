# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert the Oxford pet dataset to TFRecord for object_detection.

See: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar
     Cats and Dogs
     IEEE Conference on Computer Vision and Pattern Recognition, 2012
     http://www.robots.ox.ac.uk/~vgg/data/pets/

Example usage:
    ./create_pet_tf_record --data_dir=/home/user/pet \
        --output_dir=/home/user/pet/output
"""

import hashlib
import io
import logging
import os
import random
import re

from lxml import etree
import PIL.Image
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/netscratch/siddiqui/CrossLayerPooling/data/MFW/', 'Root directory to raw dataset.')
flags.DEFINE_string('output_dir', '/netscratch/siddiqui/CrossLayerPooling/data/', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', '/netscratch/siddiqui/CrossLayerPooling/tf-clp/label_map.pbtxt', 'Path to label map proto')
FLAGS = flags.FLAGS

TRAIN_EXAMPLE = 1
VALIDATION_EXAMPLE = 2
TEST_EXAMPLE = 0

import dataset_util

def dict_to_tf_example(data):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (image and and corresponding label)

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  # print (os.path.join(image_subdirectory, data['filename'] + '.bmp'))
  img_path = data['filename']
  with tf.gfile.GFile(img_path) as fid:
    encoded_img = fid.read()
  encoded_img_io = io.BytesIO(encoded_img)
  image = PIL.Image.open(encoded_img_io)
  if image.format == 'PNG':
    img_format = 'png'
  elif image.format == 'JPEG':
    img_format = 'jpeg'
  elif image.format == 'BMP':
    img_format = 'bmp'
  else:
    raise ValueError('Image format not PNG/JPEG/BMP')
  key = hashlib.sha256(encoded_img).hexdigest()

  (width, height) = image.size

  class_id = data['class_id']
  class_text = data['class_text']

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/channels': dataset_util.int64_feature(3),
      'image/filename': dataset_util.bytes_feature(data['filename']),
      'image/source_id': dataset_util.bytes_feature(data['filename']),
      'image/key/sha256': dataset_util.bytes_feature(key),
      'image/encoded': dataset_util.bytes_feature(encoded_img),
      'image/format': dataset_util.bytes_feature(img_format),
      'image/class/text': dataset_util.bytes_feature(class_text),
      'image/class/label': dataset_util.int64_feature(class_id),
  }))
  return example


def create_tf_record(output_filename,
                     images,
                     labels,
                     classes_dict):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    images: Absolute path of images in the dataset.
    labels: Corresponding image label.
    classes_dict: Dictionary mapping class IDs to class names.
  """
  writer = tf.python_io.TFRecordWriter(output_filename)
  for idx, example in enumerate(images):
    if idx % 100 == 0:
      logging.info('On image %d of %d', idx, len(images))

    data = {}
    data['filename'] = images[idx]
    data['class_id'] = labels[idx]
    data['class_text'] = classes_dict[labels[idx]]
    tf_example = dict_to_tf_example(data)
    writer.write(tf_example.SerializeToString())

  writer.close()


def read_file(file_name):
  data_list = []
  with open(file_name, 'r') as file:
    for line in file:
      line_chunks = line.strip().split(' ')
      assert(len(line_chunks) == 2)
      data_list.append(line_chunks[1])

  return data_list

def read_classes_dict(file_name):
  classes_map = {}
  with open(file_name, 'r') as file:
    for line in file:
      line_chunks = line.strip().split(' ')
      assert(len(line_chunks) == 2)
      classes_map[int(line_chunks[0])] = line_chunks[1]

  return classes_map

# TODO: Add test for pet/PASCAL main files.
def main(_):
  data_dir = FLAGS.data_dir

  logging.info('Reading from specified dataset.')
  image_dir = os.path.join(data_dir, 'images')
  image_examples_path = os.path.join(data_dir, 'images.txt')
  label_examples_path = os.path.join(data_dir, 'image_class_labels.txt')
  split_examples_path = os.path.join(data_dir, 'train_test_split.txt')
  classes_path = os.path.join(data_dir, 'classes.txt')

  # Read in the data from the images
  image_data = read_file(image_examples_path)
  label_data = read_file(label_examples_path)
  split_data = read_file(split_examples_path)
  classes_dict = read_classes_dict(classes_path)

  # Convert label and split data to int
  image_data = [os.path.join(image_dir, val) for val in image_data]
  label_data = [int(val) for val in label_data]
  split_data = [int(val) for val in split_data]

  # Split the data into different sets according to the defined train test split
  train_images = []
  train_labels = []
  val_images = []
  val_labels = []
  test_images = []
  test_labels = []
  for idx in range(len(image_data)):
    if split_data[idx] == TRAIN_EXAMPLE:
      train_images.append(image_data[idx])
      train_labels.append(label_data[idx])
    elif split_data[idx] == VALIDATION_EXAMPLE:
      val_images.append(image_data[idx])
      val_labels.append(label_data[idx])
    elif split_data[idx] == TEST_EXAMPLE:
      test_images.append(image_data[idx])
      test_labels.append(label_data[idx])
    else:
      print ("Error: Undefined split")
      exit (-1)

  logging.info('%d training ,%d validation and %d test examples.',
               len(train_images), len(val_images), len(test_images))

  train_output_path = os.path.join(FLAGS.output_dir, 'train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'val.record')
  test_output_path = os.path.join(FLAGS.output_dir, 'test.record')

  create_tf_record(train_output_path, train_images, train_labels, classes_dict)
  create_tf_record(val_output_path, val_images, val_labels, classes_dict)
  create_tf_record(test_output_path, test_images, test_labels, classes_dict)

if __name__ == '__main__':
  tf.app.run()
