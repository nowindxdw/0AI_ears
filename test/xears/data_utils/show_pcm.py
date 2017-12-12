"""Utilities audio convert."""
from __future__ import absolute_import
from __future__ import print_function

import array
import os
from matplotlib import pyplot

def show(filePath, start = 0, end = 5000):
    file = open(filePath, 'rb')
    base = 1 / (1<<15)

    shortArray = array.array('h') # int16
    size = int(os.path.getsize(filePath) / shortArray.itemsize)
    count = int(size / 2)
    shortArray.fromfile(file, size) # faster than struct.unpack
    file.close()
    leftChannel = shortArray[::2]
    rightChannel = shortArray[1::2]
    fig = pyplot.figure(1)

    pyplot.subplot(211)
    pyplot.title('pcm left channel [{0}-{1}] max[{2}]'.format(start, end, count))
    pyplot.plot(range(start, end), leftChannel[start:end])
    pyplot.subplot(212)
    pyplot.title('pcm right channel [{0}-{1}] max[{2}]'.format(start, end, count))
    pyplot.plot(range(start, end), rightChannel[start:end])
    pyplot.show()

def showWavArray(inputArray, start = 0, end = 5000):
    count = inputArray.shape[0]
    leftChannel = inputArray[:,0]
    #print(leftChannel.shape)
    rightChannel = inputArray[:,1]
    #print(rightChannel.shape)
    fig = pyplot.figure(1)
    pyplot.subplot(211)
    pyplot.title('pcm left channel [{0}-{1}] max[{2}]'.format(start, end, count))
    pyplot.plot(range(start, end), leftChannel[start:end])
    pyplot.subplot(212)
    pyplot.title('pcm right channel [{0}-{1}] max[{2}]'.format(start, end, count))
    pyplot.plot(range(start, end), rightChannel[start:end])
    pyplot.show()