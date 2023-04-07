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
filename='C:/py_test_temp/Kassa.wav'
#y = wavio.read(filename)

#filename='/content/drive/MyDrive/01-echo_a0987.wav'
#from PIL import Image
# читаем файл
y, sr = librosa.load(filename,sr=None)

mark_0 = get_lfm_sig(6000, 10000, 0.1, sr, 2048)
mark_1 = np.flipud(mark_0)

pos=1000

code=[1, 0, 0, 1, 1, 1, 0, 1]
pk=0
Ka=0.0025
ys1=copy.deepcopy(y)
while pos<len(y)-len(mark_0):
  if code[pk]==1:
      ys1[pos:pos + len(mark_0)] = ys1[pos:pos + len(mark_0)] + Ka * mark_1
  else:
      ys1[pos:pos + len(mark_0)] = ys1[pos:pos + len(mark_0)] + Ka * mark_0
  pos=pos+int(sr*0.11)
  pk+=1
  if pk==len(code):
      pk=0
      pos=pos+int(sr*0.21)
      #ys1[pos:pos + len(mark_0)] = ys1[pos:pos + len(mark_0)] + Ka * mark_1 + Ka * mark_0
      #pos = pos + int(sr * 0.11)








#wavio.write('C:/py_test_temp/orig_sig.wav', y/abs(y).max(), sr, sampwidth=2)
wavio.write('C:/py_test_temp/lfm5.wav', ys1/abs(ys1).max(), sr, sampwidth=2)
print('done')


plt.plot(y/abs(y).max())
plt.plot(ys1/abs(ys1).max())
plt.show()

