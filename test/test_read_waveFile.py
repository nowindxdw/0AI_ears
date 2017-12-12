from __future__ import print_function
import pytest
import os

import numpy as np
from xears.data_utils import split_audio
from xears.data_utils import show_pcm
#xears_test

def test_read_file(type="bin"):
    output_base_path = os.path.dirname(__file__)+os.path.sep+"xears"+os.path.sep+"data_source"+os.path.sep

    if type == 'bin':
        binfile = output_base_path+"model.bin"
        output = np.fromfile(binfile,dtype='float')
        output.shape = 33,400000,2
    if type == 'npy':
        npyfile = output_base_path+"model.npy"
        output = np.load(npyfile)
    if type == 'file':
        file_path = output_base_path+"modelFile.npy"
        f = file(file_path,"rb")
        output = np.load(f)

    output_wave_path = output_base_path+"test.wave"
    split_audio.recover(output[10],output_wave_path)

    #show_pcm.showWavArray(output[10],0,400000)

    #print(output.shape)
    #print(type(output))
    #npyfile = output_base_path+"model.npy"
    #txtfile = output_base_path+"model.txt"

if __name__ == '__main__':
    pytest.main([__file__])