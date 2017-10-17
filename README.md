# 0AI_ears
try to listen music and recognize the author


##prepare list


python 声音处理库选择


TimeSide（https://github.com/Parisson/TimeSide）
★★★
[第三方库] TimeSide是一个能够进行音频分析、成像、转码、流媒体和标签处理的Python框架，可以对任何音频或视频内容非常大的数据集进行复杂的处理


pydub(https://github.com/jiaaro/pydub)
[第三方库] pydub支持多种格式声音文件，可进行多种信号处理（例如压缩、均衡、归一化）、信号生成（例如正弦、方波、锯齿等）、音效注册、静音处理等
★★★


audiolazy (https://github.com/danilobellini/audiolazy)

audioread(https://github.com/beetbox/audioread)

beets(https://github.com/beetbox/audioread)

dejavu(https://github.com/worldveil/dejavu)

eyeD3(https://eyed3.readthedocs.io/en/latest/)

id3reader()

m3u8(https://github.com/globocom/m3u8)

mutagen(https://bitbucket.org/lazka/mutagen)

talkbox(https://scikits.appspot.com/talkbox)

tinytag(https://github.com/devsnd/tinytag)

mingus(https://bspaans.github.io/python-mingus/)


##timeslide（感觉偏重的一个工具，功能很强大，适合深度用户）
官方推荐docker安装：（http://files.parisson.com/timeside/doc/install.html）

-First, install Docker and docker-compose

-Then clone TimeSide:
```angular2html
git clone --recursive https://github.com/Parisson/TimeSide.git
cd TimeSide
docker-compose pull
```

##pydub
安装
```angular2html
pip install pydub

```

另需要安装依赖支持多文件格式(https://www.ffmpeg.org/general.html#File-Formats)，默认只支持wav
```angular2html
brew install libav --with-libvorbis --with-sdl --with-theora
```
例子代码：
```python
from pydub import AudioSegment
sound  = AudioSegment.from_file("/Users/dawei/Downloads/bandari.mp3")
print(sound.duration_seconds)
```

##audiolazy


##audioread
```python
pip install audioread
```

```python
import audioread
filename="/Users/dawei/Downloads/bandari.mp3"
with audioread.audio_open(filename) as f:
    print(f.channels, f.samplerate, f.duration)
```


