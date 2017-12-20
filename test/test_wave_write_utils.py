from __future__ import absolute_import
from __future__ import print_function
import pytest
import os

#xears_test
from xears.data_utils import wave_utils


def test_wave_write_test():
    wave_path = os.path.dirname(__file__)+os.path.sep+'xears'+os.path.sep+'data_source'+os.path.sep+'test30.wav'
    #wave_utils.writeSampleWav(wave_path)
    A_path = os.path.dirname(__file__)+os.path.sep+'xears'+os.path.sep+'mp3source'+os.path.sep+'sad.wav'
    wave_data,time = wave_utils.readWav(A_path)
    #print(wave_data.shape)#(2,13507200)
    #print(type(wave_data))#nmupy.ndarray

    wave_data1 = wave_data[:,:1350720]
    wave_data2 = wave_data[:,1350720:1350720*2]
    print(wave_data1.shape)
    params ={
    'nframes' : 1350720,
    'nchannels':1,
    'sampwidth':2,
    'framerate':44100
    }
    wave_utils.writeWav(wave_data2[0],params,wave_path)
    #wave_utils.playWav(wave_path)


if __name__ == '__main__':
    pytest.main([__file__])
    raw_input('Press Enter to exit...')