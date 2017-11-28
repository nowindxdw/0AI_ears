"""Utilities audio convert."""
from __future__ import absolute_import
from __future__ import print_function

from . import convert_utils
from . import split_audio
from . import shuffle_audio
import tensorflow as tf
import numpy as np

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
    print('test_size'+str(test_size))
    for i in range(test_size):
       test_index = np.random.randint(train_x_flatten.shape[1])
       test_x[:,i] = train_x_flatten[:,test_index]
       test_y[i] = train_y[test_index]
       train_x_flatten= np.delete(train_x_flatten, test_index, axis = 1)
       train_y = np.delete(train_y, test_index)
    return train_x_flatten,train_y,test_x,test_y