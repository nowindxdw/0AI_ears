# -*- coding: UTF-8 -*-
from __future__ import print_function
import pytest
import os


from xears.data_utils import split_audio

#xears_test

def test_split_wave():
    wav_name = os.path.dirname(__file__)+os.path.sep+"xears"+os.path.sep+"mp3source"+os.path.sep+"model.wav"
    print(wav_name)
    wav_splits = split_audio.split(wav_name,400000)
    #print(type(wav_splits))
    #print(wav_splits.shape)
    output_base_path = os.path.dirname(__file__)+os.path.sep+"xears"+os.path.sep+"data_source"+os.path.sep
    split_audio.store(wav_splits,output_base_path+"model.bin",store_type = "bin")
    #split_audio.store(wav_splits,output_base_path+"model.npy",store_type = "npy")
    #split_audio.store(wav_splits,output_base_path+"modelFile.npy",store_type = "file")

if __name__ == '__main__':
    pytest.main([__file__])