from __future__ import print_function
from __future__ import absolute_import
from __future__ import print_function
import pytest
import os

from xears.data_utils import pre_dataset
from xears.models import DNN
#xears_test

def test_DNN():
    A_path = os.path.dirname(__file__)+'/xears/mp3source/model.mp3'
    B_path = os.path.dirname(__file__)+'/xears/mp3source/sad.mp3'
    train_x, train_y, test_x, test_y= pre_dataset.pre_data(A_path,B_path)
    print('train_x shape='+str(train_x.shape))
    print('test_x shape='+str(test_x.shape))
    DNN.build_model(train_x, train_y, test_x, test_y)

if __name__ == '__main__':
    pytest.main([__file__])