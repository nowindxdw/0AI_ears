from __future__ import absolute_import
from __future__ import print_function
import pytest
import os

from xears.data_utils import pre_dataset
from xears.models.tf import dnn_train
#xears_test

def test_DNN():
    A_path = os.path.dirname(__file__)+'\\xears\\mp3source\\model.wav'
    B_path = os.path.dirname(__file__)+'\\xears\\mp3source\\sad.wav'
    #train_x, train_y, test_x, test_y= pre_dataset.pre_data(A_path,B_path)
    data_set = pre_dataset.pre_wav_data(A_path,B_path)
    dnn_train.train(data_set)

if __name__ == '__main__':
    pytest.main([__file__])
    raw_input('Press Enter to exit...')