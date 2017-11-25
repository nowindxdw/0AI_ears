from __future__ import print_function
import pytest
import os

from xears.data_utils import pre_dataset
#xears_test

def test_pre_data():
    A_path = os.path.dirname(__file__)+'/xears/mp3source/model.mp3'
    B_path = os.path.dirname(__file__)+'/xears/mp3source/sad.mp3'
    train_x, train_y = pre_dataset.pre_data(A_path,B_path)
    print(train_x.shape)
    print(train_y.shape)

if __name__ == '__main__':
    pytest.main([__file__])