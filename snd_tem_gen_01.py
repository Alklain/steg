import pandas as pd
import math
import cmath
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
#import seaborn as sns
#import soundfile as sf
import playsound
import wavio
from glob import glob
from docx import Document
from docxtpl import DocxTemplate
import librosa.display
import librosa
#import IPython.display as ipd
#from playsound import playsound
from sndmrk import *
import copy
#filename='C:/py_test_temp/kladbische.mp3'
#filename='C:/py_test_temp/Kassa.wav'
#y = wavio.read(filename)

#filename='/content/drive/MyDrive/01-echo_a0987.wav'
#from PIL import Image
# читаем файл
#y, sr = librosa.load(filename,sr=None)
sr=48000
y1=get_sin_sig(3000, 0, 4, sr,2048)
y2=get_sin_sig(7000, 0, 4, sr,2048)
y3=get_sin_sig(13000, 0, 4, sr,2048)
y4=get_sin_sig(13000, 0, 4, sr,2048)
y=y1+y2+y3+н4

wavio.write('C:/py_test_temp/sin_3fi_180.wav', y/abs(y).max(), sr, sampwidth=2)

print('done')


plt.plot(y/abs(y).max())
#plt.plot(ys1/abs(ys1).max())
plt.show()

