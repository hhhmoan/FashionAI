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
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import csv
import tensorflow as tf
import numpy as np
import resnet
import argparse
parser = argparse.ArgumentParser(description='Initial args.')
parser.add_argument('--attr_index', type=int, default=0, help='The index of the attribute [0,7].')
args = parser.parse_args()
attr_index = args.attr_index

AttrInfo = [{'AttrKey': 'skirt_length_labels', 'AttrValues': ['Invisible', 'Short Length', 'Knee Length', 'Midi Length', 'Ankle Length', 'Floor Length']}, 
 {'AttrKey': 'coat_length_labels', 'AttrValues': ['Invisible', 'High Waist Length', 'Regular Length', 'Long Length', 'Micro Length', 'Knee Length', 'Midi Length', 'Ankle&Floor Length']}, 
 {'AttrKey': 'collar_design_labels', 'AttrValues': ['Invisible', 'Shirt Collar', 'Peter Pan', 'Puritan Collar', 'Rib Collar']}, 
 {'AttrKey': 'lapel_design_labels', 'AttrValues': ['Invisible', 'Notched', 'Collarless', 'Shawl Collar', 'Plus Size Shawl']}, 
 {'AttrKey': 'neck_design_labels', 'AttrValues': ['Invisible', 'Turtle Neck', 'Ruffle Semi-High Collar', 'Low Turtle Neck', 'Draped Collar']}, 
 {'AttrKey': 'pant_length_labels', 'AttrValues': ['Invisible', 'Short Pant', 'Mid Length', '3/4 Length', 'Cropped Pant', 'Full Length']}, 
 {'AttrKey': 'sleeve_length_labels', 'AttrValues': ['Invisible', 'Sleeveless', 'Cup Sleeves', 'Short Sleeves', 'Elbow Sleeves', '3/4 Sleeves', 'Wrist Length', 'Long Sleeves', 'Extra Long Sleeves']}]

#AttrValueLens = []
for attr in AttrInfo:
    attr['AttrValueLens'] = len(attr['AttrValues'])
    #AttrValueLens.append(attr['AttrValueLens'])
    #print(attr['AttrKey'], attr['AttrValueLens'], 'Values')

attr = AttrInfo[attr_index]
print('Use', attr['AttrKey'], '|', attr['AttrValueLens'], 'Values')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_CLASSES = attr['AttrValueLens']

_NUM_IMAGES = {
    'train': 7378,
    'validation': 1845,
}

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 1500
_TRAIN_STYLE = attr['AttrKey']#'neck_design_labels'#'skirt_length_labels'
_TRAIN_CSV_PATH = './data/fashionAI_attributes_train_20180222/base/Annotations/label.csv'
_TRAIN_IMAGE_PATH = './data/fashionAI_attributes_train_20180222/base/'
_TEST_CSV_PATH = "./data/fashionAI_attributes_test_a_20180222/rank/Tests/question.csv"
_TEST_IMAGE_PATH = './data/fashionAI_attributes_test_a_20180222/rank/'

###############################################################################
# Data processing
###############################################################################

def load_from_csv(train_csv, train_style, root_path):
	with open(train_csv,'rb') as csvfile:
		reader = csv.reader(csvfile)
		train_rows = [[row[0],row[2]] for row in reader if row[1] == train_style]
	#print(train_rows)
	image_name_list = [root_path+row[0] for row in train_rows]
	labels = [row[1] for row in train_rows]
	labels = [nym2label(i) for i in labels]
	return image_name_list, labels

def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string, channels=_NUM_CHANNELS)
  image_resized = tf.image.resize_images(image_decoded, [256,256])
  image_resized = tf.image.random_flip_left_right(image_resized)
  #offset_height = np.random.randint(31)
  #offset_width = np.random.randint(31)
  #target_height = 224
  #target_width = 224
  image_crop = tf.random_crop(image_resized,[224,224,3])#tf.image.crop_to_bounding_box(image_resized, offset_height, offset_width, target_height, target_width)
  label = tf.one_hot(label, _NUM_CLASSES)
  return image_crop, label

def _parse_function_eval(filename, label):
	image_string = tf.read_file(filename)
	image_decoded = tf.image.decode_jpeg(image_string, channels=_NUM_CHANNELS)
	image_resized = tf.image.resize_images(image_decoded, [224,224])
	#offset_height = np.random.randint(31)
	#offset_width = np.random.randint(31)
	#target_height = 224
	#target_width = 224
	#image_crop = tf.image.crop_to_bounding_box(image_resized, offset_height, offset_width, target_height, target_width)
	label = tf.one_hot(label, _NUM_CLASSES)
	return image_resized, label


def nym2label(nym):
	return nym.find('y')
 
def input_fn_new(is_training,  batch_size=32, num_epochs=10):
  #filenames, labels = load_from_csv('/home/hhh/fashionAI/FashionAI/data/label.csv', 'skirt_length_labels', '/home/data/fashionAI/fashionAI_attributes_train_20180222/base/')
  if is_training=='train':
    filenames, labels = load_from_csv(_TRAIN_CSV_PATH, _TRAIN_STYLE, _TRAIN_IMAGE_PATH)
    filenames = filenames[0:int(len(filenames)*0.8)]
    labels = labels[0:int(len(labels)*0.8)]
  elif is_training=='eval':
    filenames, labels = load_from_csv(_TRAIN_CSV_PATH, _TRAIN_STYLE, _TRAIN_IMAGE_PATH)
    filenames = filenames[int(len(filenames)*0.8):-1]
    labels = labels[int(len(labels)*0.8):-1]
  else:
    filenames, labels = load_from_csv(_TEST_CSV_PATH, _TRAIN_STYLE, _TEST_IMAGE_PATH)

  dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

  #dataset = dataset.prefetch(buffer_size=cfg.batch_size)
  if is_training == 'train':
    dataset = dataset.map(_parse_function)
    dataset = dataset.shuffle(buffer_size=1000)
  else:
    dataset = dataset.map(_parse_function_eval) 
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat(num_epochs)
  #dataset = dataset.prefetch(1)
  iterator = dataset.make_one_shot_iterator()
  features, labels = iterator.get_next()

  return features, labels

###############################################################################
# Running the model
###############################################################################
class FashionAIModel(resnet.Model):

  def __init__(self, resnet_size, data_format=None, num_classes=_NUM_CLASSES,
    version=resnet.DEFAULT_VERSION):
    """These are the parameters that work for Imagenet data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      version: Integer representing which version of the ResNet network to use.
        See README for details. Valid values: [1, 2]
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 50:
      bottleneck = False
      final_size = 512
    else:
      bottleneck = True
      final_size = 2048

    super(FashionAIModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        second_pool_size=7,
        second_pool_stride=1,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        final_size=final_size,
        version=version,
        data_format=data_format)


def _get_block_sizes(resnet_size):
  """The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.
  """
  choices = {
      18: [2, 2, 2, 2],
      34: [3, 4, 6, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)


def fashionai_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""
  learning_rate_fn = resnet.learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=256,
      num_images=_NUM_IMAGES['train'], boundary_epochs=[30, 60, 80, 90],
      decay_rates=[1, 0.5, 0.25, 0.125, 1e-4])

  return resnet.resnet_model_fn(features, labels, mode, FashionAIModel,
                                resnet_size=params['resnet_size'],
                                weight_decay=1e-4,
                                learning_rate_fn=learning_rate_fn,
                                momentum=0.9,
                                data_format=params['data_format'],
                                version=params['version'],
                                loss_filter_fn=None,
                                multi_gpu=params['multi_gpu'])


def main(unused_argv):
  input_function = input_fn_new#get_synth_input_fn()
  resnet.resnet_main(FLAGS, fashionai_model_fn, input_function)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)

  parser = resnet.ResnetArgParser(
      resnet_size_choices=[18, 34, 50, 101, 152, 200])
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
