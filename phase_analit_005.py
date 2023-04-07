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

y1=y[190000-100:210000]


Nfft = 2048
Step = 512
#CompSp, LogSp, Step = GetMatrixSpectrum(y1, Nfft, Step )
CompSp, LogSp, Step = GetMatrixSpectrumNoWin(y1, Nfft, Step )
ys = GetSignalFromSpectrum(CompSp, Step)

NumBlk=len(CompSp[:][1])
max_freq=[]

#img = Image.fromarray(LogSp)
#img.show()

frq_list1=[28, 128, 218]
frq_list0=[83, 178, 250]
NewMarkSp=copy.deepcopy(CompSp)

code=[1,0,0,1,0,1,1,1] # ПОСЫЛКА ДАННЫХ
pk=0
for i in range(NumBlk):
  sp=copy.deepcopy(CompSp[:,i])
  # ищем номера максимальных частот в полосах
  sp3=copy.deepcopy(sp)
  frq_list=frq_list1
  #for k in range(len(frq_list)):
  #  tfr=frq_list[k]
  #  Mod_Fi = SetComplexRotationFi(sp[tfr], 180)
  #  sp3 = cange_dat_in_complex_spec(sp3, tfr, Mod_Fi)
    
  if i%15==0:     
    for k in range(len(frq_list)):
        tfr=frq_list[k]
        Mod_Fi = SetComplexRotationFi(sp[tfr], 90)
        sp3 = cange_dat_in_complex_spec(sp3, tfr, Mod_Fi)

    NewMarkSp[:, i] = sp3

ys1 = GetSignalFromSpectrum(NewMarkSp, Step)

#plt.plot(y1/abs(y1).max())
#plt.plot(ys/abs(ys).max())
#plt.plot(ys1/abs(ys1).max())
#plt.plot(max_freq)
#plt.show()
#wavio.write('C:/py_test_temp/Kassa_90.wav', ys1/abs(ys1).max(), sr, sampwidth=2)

## детектировование сигнала
print('start detection')
CompSpD, LogSpD, Step = GetMatrixSpectrum(ys1, Nfft, Step )
NewMarkSpD=copy.deepcopy(CompSp)
max_freq2=[]
for i in range(NumBlk):
    sp = copy.deepcopy(CompSpD[:, i])
    frq_list = frq_list1 
    sp3 = copy.deepcopy(sp)
    for k in range(len(frq_list)):
        tfr = frq_list[k]
        Mod_Fi = SetComplexRotationFi(sp[tfr], 90)
        sp3 = cange_dat_in_complex_spec(sp3, tfr, Mod_Fi)
    NewMarkSpD[:, i] = sp3

ys2 = GetSignalFromSpectrum(NewMarkSpD, Step)


plt.plot(ys2-ys1)
plt.plot(ys-ys1)
#
#plt.plot(np.array(max_freq2))
#plt.plot(np.array(max_freq)+1)
plt.show()
