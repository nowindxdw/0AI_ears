from __future__ import print_function
import pytest
import os

from xears.data_utils import convert_utils
from xears.data_utils import show_pcm
#xears_test

def test_vector_classification():
    wav_name = convert_utils.convert_mp3_to_wav(os.path.dirname(__file__)+'/xears/mp3source/model.mp3')
    print(wav_name)
    show_pcm.show(wav_name,0,5000)

if __name__ == '__main__':
    pytest.main([__file__])