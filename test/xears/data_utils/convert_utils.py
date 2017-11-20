"""Utilities audio convert."""
from __future__ import absolute_import
from __future__ import print_function

import audioread
import sys
import os
import wave
import contextlib

def convert_mp3_to_wav(filename):
    if not os.path.exists(filename):
            print("File not found.")
            sys.exit(1)

    try:
       with audioread.audio_open(filename) as f:
           print('Input file: %i channels at %i Hz; %.1f seconds.' %
                 (f.channels, f.samplerate, f.duration))
           print('Backend:', str(type(f).__module__).split('.')[1])

           with contextlib.closing(wave.open(filename + '.wav', 'w')) as of:
                of.setnchannels(f.channels)
                of.setframerate(f.samplerate)
                of.setsampwidth(2)

                for buf in f:
                    of.writeframes(buf)
           return filename+".wav"

    except audioread.DecodeError:
       print("File could not be decoded.")
       sys.exit(1)