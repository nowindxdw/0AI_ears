import array
import os
from matplotlib import pyplot

fileName = '/Users/dawei/Downloads/bandarimp3.wav' # 2 channel, 16 bit per sample
file = open(fileName, 'rb')
base = 1 / (1<<15)

shortArray = array.array('h') # int16
size = int(os.path.getsize(fileName) / shortArray.itemsize)
count = int(size / 2)
shortArray.fromfile(file, size) # faster than struct.unpack
file.close()
leftChannel = shortArray[::2]
rightChannel = shortArray[1::2]

def showPCM(leftChannel, rightChannel, start = 0, end = 5000):
    fig = pyplot.figure(1)

    pyplot.subplot(211)
    pyplot.title('pcm left channel [{0}-{1}] max[{2}]'.format(start, end, count))
    pyplot.plot(range(start, end), leftChannel[start:end])

    pyplot.subplot(212)
    pyplot.title('pcm right channel [{0}-{1}] max[{2}]'.format(start, end, count))
    pyplot.plot(range(start, end), rightChannel[start:end])

    pyplot.show()
    # fig.savefig('pcm.pdf') # save figure to pdf file

showPCM(leftChannel, rightChannel, 0, count)