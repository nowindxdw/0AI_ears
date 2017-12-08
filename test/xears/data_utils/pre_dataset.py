# -*- coding: UTF-8 -*-
"""Utilities audio convert."""
from __future__ import absolute_import
from __future__ import print_function

from . import convert_utils
from . import split_audio
from . import shuffle_audio
import tensorflow as tf
import numpy as np

def dense_to_one_hot(labels_dense, num_classes=3):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  labels_one_hot = np.zeros((num_labels, num_classes))
  #print(labels_one_hot.shape)
  for i in range(num_labels):
    labels_index = int(labels_dense[i])
    labels_one_hot[i][labels_index] = 1
  #print(labels_one_hot)  
  return labels_one_hot

class DataSet(object):
  def __init__(self, images, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,labels.shape))
      self._num_examples = int(images.shape[0])
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
     
      #assert images.shape[3] == 1
      #images = images.reshape(images.shape[0],images.shape[1] * images.shape[2])
      # Convert from [0, 255] -> [0.0, 1.0].
      images = tf.to_float(images)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1.0 for _ in xrange(784)]
      fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    #若当前训练读取的index>总体的images数时，则读取读取开始的batch_size大小的数据
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def pre_data(mp3_a,mp3_b,step = 200000):
    wav_a = convert_utils.convert_mp3_to_wav(mp3_a)
    wav_b = convert_utils.convert_mp3_to_wav(mp3_b)
    train_ori_a = split_audio.split(wav_a,step)
    train_ori_b = split_audio.split(wav_b,step)
    train_x, train_y = shuffle_audio.shuffle_two_audio(train_ori_a,train_ori_b)
    train_x_flatten = train_x.reshape(train_x.shape[1], -1).T
    #gen test set
    train_len = train_x.shape[1]
    test_size = int(train_len*0.3)
    test_x = np.zeros((train_x_flatten.shape[0],test_size))
    test_y = np.zeros(test_size)
    print('test_size:'+str(test_size))
    for i in range(test_size):
       test_index = np.random.randint(train_x_flatten.shape[1])
       test_x[:,i] = train_x_flatten[:,test_index]
       test_y[i] = train_y[test_index]
       train_x_flatten= np.delete(train_x_flatten, test_index, axis = 1)
       train_y = np.delete(train_y, test_index)
    return train_x_flatten,train_y,test_x,test_y
    
def pre_wav_data(wav_a,wav_b,step = 200000,fake_data=False,):
    class DataSets(object):
        pass
    data_sets = DataSets()
    if fake_data:
      data_sets.train = DataSet([], [], fake_data=True)
      data_sets.validation = DataSet([], [], fake_data=True)
      data_sets.test = DataSet([], [], fake_data=True)
      return data_sets

    train_ori_a = split_audio.split(wav_a,step)
    train_ori_b = split_audio.split(wav_b,step)
    train_x, train_y = shuffle_audio.shuffle_two_audio(train_ori_a,train_ori_b)
    train_x_flatten = train_x.reshape(train_x.shape[1], -1).T
    #gen test set
    train_len = train_x.shape[1]
    test_size = int(train_len*0.3)
    test_x = np.zeros((train_x_flatten.shape[0],test_size))
    test_y = np.zeros(test_size)
    print('test_size:'+str(test_size))
    for i in range(test_size):
       test_index = np.random.randint(train_x_flatten.shape[1])
       test_x[:,i] = train_x_flatten[:,test_index]
       test_y[i] = train_y[test_index]
       train_x_flatten= np.delete(train_x_flatten, test_index, axis = 1)
       train_y = np.delete(train_y, test_index)
    train_x_flatten = tf.transpose(train_x_flatten)
    train_y = dense_to_one_hot(train_y)
    test_x = tf.transpose(test_x)
    test_y = dense_to_one_hot(test_y)
    data_sets.train = DataSet(train_x_flatten, train_y)
    data_sets.validation = DataSet(test_x, test_y)
    data_sets.test = DataSet(test_x, test_y)
    return data_sets
	