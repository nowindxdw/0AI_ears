from __future__ import print_function
import pytest
import os

import numpy as np
from xears.data_utils import split_audio
from xears.data_utils import show_pcm
#xears_test

def test_read_file():
    output_base_path = os.path.dirname(__file__)+os.path.sep+"xears"+os.path.sep+"data_source"+os.path.sep
    binfile = output_base_path+"model.bin"
    output = np.fromfile(binfile,dtype='float')
    output.shape = 33,400000,2
    show_pcm.showWavArray(output[0],0,400000)
    #print(output.shape)
    #print(type(output))
    #npyfile = output_base_path+"model.npy"
    #txtfile = output_base_path+"model.txt"

if __name__ == '__main__':
    pytest.main([__file__])