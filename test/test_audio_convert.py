from __future__ import print_function
import pytest
import os

from xears.data_utils import convert_utils

#xears_test

def test_vector_classification():
    convert_utils.convert_mp3_to_wav(os.path.dirname(__file__)+'/xears/mp3source/model.mp3')

if __name__ == '__main__':
    pytest.main([__file__])