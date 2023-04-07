import pandas as pd
import math
import cmath
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
import seaborn as sns
import soundfile as sf
import playsound
import wavio
from glob import glob
from docx import Document
from docxtpl import DocxTemplate
import librosa.display
import librosa
import IPython.display as ipd
from playsound import playsound
from sndmrk import *
import copy
#filename='C:/py_test_temp/kladbische.mp3'
filename='C:/py_test_temp/Kassa.wav'
#filename='/content/drive/MyDrive/01-echo_a0987.wav'
from PIL import Image
# читаем файл
y, sr = librosa.load(filename,sr=None)

#y1=y[190000-100:210000]


Nfft = 2048
Step = 512
#CompSp, LogSp, Step = GetMatrixSpectrum(y1, Nfft, Step )
CompSp, LogSp, Step = GetMatrixSpectrumNoWin(y, Nfft, Step )
ys = GetSignalFromSpectrum(CompSp, Step)

NumBlk=len(CompSp[:][1])
max_freq=[]

#img = Image.fromarray(LogSp)
#img.show()

frq_list=[30, 75, 260]
NewMarkSp=copy.deepcopy(CompSp)
for i in range(NumBlk):
  sp=copy.deepcopy(CompSp[:,i])
  pos_max = get_max_soectrum_in_window(sp, sr, 800, 2000)  # нашли максимум в полосе частот
  max_freq.append(pos_max)
  frq_list[1]=pos_max
  if i%6==0:
    # меняем фазы
    sp3=copy.deepcopy(sp)
    for k in range(len(frq_list)):
        tfr=frq_list[k]
        Mod_Fi = SetComplexRotationFi(sp[tfr], 90)
        sp3=cange_dat_in_complex_spec(sp3,tfr,Mod_Fi)
    NewMarkSp[:, i] = sp3

ys1 = GetSignalFromSpectrum(NewMarkSp, Step)

#plt.plot(y1/abs(y1).max())
plt.plot(ys/abs(ys).max())
plt.plot(ys1/abs(ys1).max())
#plt.plot(max_freq)
plt.show()
wavio.write('C:/py_test_temp/Kassa_90.wav', ys1/abs(ys1).max(), sr, sampwidth=2)