# -*- coding: UTF-8 -*-
"""Utilities audio convert."""
from __future__ import absolute_import
from __future__ import print_function

import array
import os
import numpy as np
import wave
import contextlib

def split(filePath, step = 200000):
    file = open(filePath, 'rb')
    base = 1 / (1<<15)

    shortArray = array.array('h') # int16
    size = int(os.path.getsize(filePath) / shortArray.itemsize)
    count = int(size / 2)
    shortArray.fromfile(file, size) # faster than struct.unpack
    file.close()
    leftChannel = shortArray[::2] #从整列表中切出，分隔为“2”
    rightChannel = shortArray[1::2] #从整列表中切出，从1开始，分隔为“2”
    print("leftChannel"+str(len(leftChannel)))
    print("rightChannel"+str(len(rightChannel)))
    slen = int(len(leftChannel)/step)
    train_ori = np.zeros((2,slen,step))
    for i in range(slen):
       train_ori[0][i] = leftChannel[i*step:(i+1)*step]
       train_ori[1][i] =  rightChannel[i*step:(i+1)*step]
    return np.transpose(train_ori, (1, 2, 0))  #output (m, size, depth)


def store(input, store_path, store_type='npy'):
    if store_type == 'npy':
        np.save(store_path,input)  # np.load(store_path)
    if store_type == 'bin':
        input.tofile(store_path)
        # output = np.fromfile(store_path,dtype=**),output.shape = []
    if store_type == 'file':  #store npy file
        f = file(store_path,'wb')
        np.save(f,input)
        f.close()
        #读取 f = file(path,"rb")  np.load(f)

def recover(input, store_path):
    input = np.reshape(input[:,:],(input.shape[0]*2,1))
    with contextlib.closing(wave.open(store_path, 'wb')) as wavfile:
        wavfile.setparams((2, 2, 44100, 0, 'NONE', 'NONE'))
        wavfile.writeframes(input)



