# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import wave
import pylab as pl
import numpy as np
import scipy.signal as signal
import pyaudio
import struct

def readWav(wav_path):
   f = wave.open(wav_path,"rb")
   params = f.getparams()
   #读取格式信息
   nchannels, sampwidth, framerate, nframes = params[:4]
   print('Input wav: %i channels  %i sampwidth %i framerate %i nframes' %
                    (nchannels,sampwidth, framerate, nframes ))
   #读取波形数据
   str_data = f.readframes(nframes)
   f.close()
   #将波形数据转换为数组
   wave_data = np.fromstring(str_data, dtype=np.short)
   wave_data.shape = -1, nchannels
   wave_data = wave_data.T
   time = np.arange(0, nframes) * (1.0 / framerate)
   print('time:'+str(time))
   return wave_data,time

def drawWave(wave_data,time):
    pl.subplot(211)
    pl.plot(time, wave_data[0])
    if(wave_data.shape[0]==2):
        pl.subplot(212)
        pl.plot(time, wave_data[1], c="g")
    pl.xlabel("time (seconds)")
    pl.show()


def writeSampleWav(wave_path):
    framerate = 44100
    time = 10
    # 产生10秒44.1kHz的100Hz - 1kHz的频率扫描波
    t = np.arange(0, time, 1.0/framerate)
    wave_data = signal.chirp(t, 100, time, 1000, method='linear') * 10000
    wave_data = wave_data.astype(np.short)
    print(wave_data)
    print(type(wave_data))
    print(wave_data.shape)
    # 打开WAV文档
    f = wave.open(wave_path, "wb")

    # 配置声道数、量化位数和取样频率
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(framerate)
    # 将wav_data转换为二进制数据写入文件
    f.writeframes(wave_data.tostring())
    f.close()

def writeWav(wave_data, params, wave_path):
    nframes = params['nframes']      #采样点数
    nchannels = params['nchannels']  #通道数
    sampwidth = params['sampwidth']  #量化位数（byte）
    fs = params['framerate']         #采样频率
    outData = wave_data              #待写入wav的数据，

    outData = np.reshape(outData,[nframes*nchannels,1])
    outwave = wave.open(wave_path, 'wb')#定义存储路径以及文件名
    framerate = int(fs)
    comptype = "NONE"
    compname = "not compressed"
    outwave.setparams((nchannels, sampwidth, framerate, nframes,
        comptype, compname))

    for v in outData:
        outwave.writeframes(struct.pack('h', int(v))) #outData:16位，-32767~32767，注意不要溢出
    outwave.close()

def playWav(wav_path):
    chunk = 1024
    wf = wave.open(wav_path, 'rb')
    p = pyaudio.PyAudio()
    # 打开声音输出流
    stream = p.open(format = p.get_format_from_width(wf.getsampwidth()),
                    channels = wf.getnchannels(),
                    rate = wf.getframerate(),
                    output = True)
    # 写声音输出流进行播放
    while True:
        data = wf.readframes(chunk)
        if data == "": break
        stream.write(data)
    stream.close()
    p.terminate()

def preprocess_wave(wave_data):
    wave_data = wave_data.reshape((670,672,3))
    wave_data = np.expand_dims(wave_data, axis=0)
    return wave_data

def deprocess_wave(wave_data):
    wave_data = wave_data.reshape((1,wave_data.shape[1]*wave_data.shape[2]*wave_data.shape[3]))
    return wave_data

def gen_noise_wave(wave_data):
    #满足正态分布sigma * np.random.randn(...) + mu
    len = wave_data.shape[1]
    noise_data = np.zeros((1,len))

    for i in range(len):
       if(np.abs(wave_data[0][i]) >15000):
          noise_data[0][i] = wave_data[0][i] +np.random.randn()*40000
       if(noise_data[0][i]<-32767):
          noise_data[0][i] = 0
       if(noise_data[0][i]>32767):
          noise_data[0][i] = 0

    return noise_data
