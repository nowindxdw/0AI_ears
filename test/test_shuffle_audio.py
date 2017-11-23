from __future__ import print_function
import pytest
import os

from xears.data_utils import shuffle_audio
from xears.data_source import train_set
#xears_test

def test_shuffle_audio():
    trainA = train_set.train_A_x()
    trainB = train_set.train_B_x()
    train_x, train_y = shuffle_audio.shuffle_two_audio(trainA,trainB)
    print(train_x.shape)
    print(train_y.shape)

if __name__ == '__main__':
    pytest.main([__file__])