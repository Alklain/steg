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
filename='C:/py_test_temp/sin_3fi_180.wav'
#filename='C:/py_test_temp/Kassa.wav'
#filename='/content/drive/MyDrive/01-echo_a0987.wav'
from PIL import Image
# читаем файл
y, sr = librosa.load(filename,sr=None)
#plt.plot(y)
#plt.show()
#print(GetFerstPosMax([1,1,3,4,5,3,2,4,3],lw=2))

Nfft = 2048
Step = 512
CompSp, LogSp, Step = GetMatrixSpectrum(y, Nfft, Step )
ys = GetSignalFromSpectrum(CompSp, Step)

#temp_s=CompSp[:,150]


#pos_max=get_max_soectrum_in_window(temp_s,sr,800,4000) # нашли максимум в полосе частот
#Sp_new_fi=change_phase_in_spectrum(temp_s,pos_max,1,0)
#print(pos_max)

#plt.figure()
#plt.plot(20*np.log10(abs(temp_s)))
#plt.plot(20*np.log10(abs(Sp_new_fi)))


nrf = int((1000 / sr) * Nfft)

NewMarkSp=copy.deepcopy(CompSp)

nrf=299
#
NumBlk=len(CompSp[:][1])
max_freq=[]
for i in range(NumBlk):
  sp=copy.deepcopy(CompSp[:,i])
  #pos_max = get_max_soectrum_in_window(sp, sr, 800, 4000)  # нашли максимум в полосе частот
  #max_freq.append(pos_max)
  #Sp_new_fi = change_phase_in_spectrum(sp, pos_max, 3, 90)
  Sp_new_fi = change_phase_in_spectrum(sp, nrf, 10, 180)
  NewMarkSp[:, i] = Sp_new_fi
#  if i==100:
#    # поворот фазы
#    sp_f1=sp[nf-1]
#    sp_f1_f=SetComplexRotationFi(sp_f1, 0)
#    sp_f2=sp[nf]
#    sp_f2_f=SetComplexRotationFi(sp_f2, 0)
#    sp_f3=sp[nf+1]
#    sp_f3_f=SetComplexRotationFi(sp_f3, 0)
#    sp[nf-1]=sp_f1_f
#    sp[len(sp)-(nf-1)]=sp[nf-1]
#    sp[nf]=sp_f2_f
#    sp[len(sp)-(nf)]=sp[nf]
#    sp[nf+1]=sp_f3_f
#    sp[len(sp)-(nf+1)]=sp[nf+1]
  # сохраняем данные



ys1 = GetSignalFromSpectrum(NewMarkSp, Step)
print('done')
wavio.write('C:/py_test_temp/fi_4.wav', ys1/abs(ys1).max(), sr, sampwidth=2)

#plt.figure()
#plt.plot(max_freq)

plt.figure()
plt.plot(y/abs(y).max())
plt.plot(ys1/abs(ys1).max())
plt.show()
