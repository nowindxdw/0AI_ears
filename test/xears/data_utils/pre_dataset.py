"""Utilities audio convert."""
from __future__ import absolute_import
from __future__ import print_function

from . import convert_utils
from . import split_audio
from . import shuffle_audio

def pre_data(mp3_a,mp3_b):
    wav_a = convert_utils.convert_mp3_to_wav(mp3_a)
    wav_b = convert_utils.convert_mp3_to_wav(mp3_b)
    train_ori_a = split_audio.split(wav_a)
    train_ori_b = split_audio.split(wav_b)
    train_x, train_y = shuffle_audio.shuffle_two_audio(train_ori_a,train_ori_b)
    return train_x,train_y