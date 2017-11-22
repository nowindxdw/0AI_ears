"""Utilities audio convert."""
from __future__ import absolute_import
from __future__ import print_function

import array
import os
import numpy as np

from matplotlib import pyplot

def split(filePath, step = 200000):
    file = open(filePath, 'rb')
    base = 1 / (1<<15)

    shortArray = array.array('h') # int16
    size = int(os.path.getsize(filePath) / shortArray.itemsize)
    count = int(size / 2)
    shortArray.fromfile(file, size) # faster than struct.unpack
    file.close()
    leftChannel = shortArray[::2]
    rightChannel = shortArray[1::2]
    print("leftChannel"+str(len(leftChannel)))
    print("rightChannel"+str(len(rightChannel)))
    slen = len(leftChannel)/step
    train_ori = np.zeros((2,slen,step))
    for i in range(slen):
       train_ori[0][i] = leftChannel[i*step:(i+1)*step]
       train_ori[1][i] =  rightChannel[i*step:(i+1)*step]
    return train_ori