from __future__ import print_function
from __future__ import absolute_import
from __future__ import print_function
import pytest
import os

from xears.data_utils import pre_dataset
#xears_test

def test_pre_data():
    A_path = os.path.dirname(__file__)+os.path.sep+'xears'+os.path.sep+'mp3source'+os.path.sep+'model.mp3'
    B_path = os.path.dirname(__file__)+os.path.sep+'xears'+os.path.sep+'mp3source'+os.path.sep+'sad.mp3'
    train_x, train_y, test_x, test_y = pre_dataset.pre_data(A_path,B_path)
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)
if __name__ == '__main__':
    pytest.main([__file__])