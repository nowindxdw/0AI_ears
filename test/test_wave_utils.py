from __future__ import absolute_import
from __future__ import print_function
import pytest
import os

#xears_test
from xears.data_utils import wave_utils


def test_wave_test():
    #A_path = os.path.dirname(__file__)+os.path.sep+'xears'+os.path.sep+'mp3source'+os.path.sep+'model.wav'
    A_path = os.path.dirname(__file__)+os.path.sep+'xears'+os.path.sep+'data_source'+os.path.sep+'test.wav'
    wave_data,time = wave_utils.readWav(A_path)
    wave_utils.drawWave(wave_data,time)

if __name__ == '__main__':
    pytest.main([__file__])
    raw_input('Press Enter to exit...')