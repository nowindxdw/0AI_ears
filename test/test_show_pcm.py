from __future__ import print_function
import pytest
import os


from xears.data_utils import show_pcm

#xears_test

def test_show_pcm():
    wav_name = os.path.dirname(__file__)+os.path.sep+"xears"+os.path.sep+"mp3source"+os.path.sep+"model.wav"
    print(wav_name)
    show_pcm.show(wav_name,0,400000)

if __name__ == '__main__':
    pytest.main([__file__])