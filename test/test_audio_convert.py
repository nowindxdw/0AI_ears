from __future__ import print_function
import pytest
import os

from xears.data_utils import convert_utils
from xears.data_utils import show_pcm
from xears.data_utils import split_audio
#xears_test

def test_vector_classification():
    audio_path = os.path.dirname(__file__)+os.path.sep+'xears'+os.path.sep+'mp3source'+os.path.sep+'model.mp3'
    wav_name = convert_utils.convert_mp3_to_wav(audio_path)
    print(wav_name)
    #show_pcm.show(wav_name,0,5000)
    train_ori = split_audio.split(wav_name )
    print(train_ori.shape)

if __name__ == '__main__':
    pytest.main([__file__])