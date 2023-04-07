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
filename='C:/py_test_temp/kladbische.mp3'
#filename='/content/drive/MyDrive/01-echo_a0987.wav'
from PIL import Image
# читаем файл
y, sr = librosa.load(filename,sr=None)

## генерим тест
#sr=22050
#Tm=1
#t = np.linspace(0, Tm, Tm*sr)
#y = (np.sin(t*2*math.pi*500) + np.sin(t*2*math.pi*1500)+
#     np.sin(t*2*math.pi*2500) + np.sin(t*2*math.pi*3500) + np.sin(t*2*math.pi*4000) +
#     np.sin(t*2*math.pi*4500) + np.sin(t*2*math.pi*6500)+
#     np.random.normal(0, 0.001, len(t))                       
#)


#y = 2*np.sin(t*2*math.pi*500) + np.sin(t*2*math.pi*500)+ np.random.normal(0, 0.001, len(t)) 

#plt.plot(t,y)
#plt.show()
#print(GetFerstPosMax([1,1,3,4,5,3,2,4,3],lw=2))

Nfft = 2048
Step = 512
CompSp, LogSp, Step = GetMatrixSpectrum(y, Nfft, Step )
ys = GetSignalFromSpectrum(CompSp, Step)

#plt.plot(y)
#plt.plot(ys)
#plt.show()


## обработка максимумов спектра


#nrf1=int((1000/sr)*Nfft)
#nrf2=int((5000/sr)*Nfft)
temp_s=CompSp[:,130]

f1=1000
f2=5000
df=5

print('start')
NumBlk=len(CompSp[:][1])

NewMarkSp=copy.deepcopy(CompSp)
band_shift=0
step_sift=0
Ku=1
gr=1
for i in range(NumBlk):
  sp=copy.deepcopy(CompSp[:,i])
  fl=0
  if i%16==0:
    NewSp=GetLocalMaxSpectrum(sp,f1,f2,df)
    #SiftSp=ShiftSpectrumDown(NewSp, f1, f2,2,5)
    SiftSp=ShiftSpectrumFullDwon(NewSp, f1, f2,band_shift,step_sift)
    #SiftSp=ShiftSpectrumFullUp(NewSp, f1, f2,500,3)
    sp=Ku*copy.deepcopy(SiftSp)
    #sp=MixSpectrum(Ku*copy.deepcopy(SiftSp), sp)
    fl=1
    if gr==1:
      plt.figure(1,clear=True)
      plt.plot(abs(NewSp))
      plt.plot(abs(Ku*copy.deepcopy(SiftSp)))
      plt.show()
      gr=0
  if (i+1)%16==0:
    NewSp=GetLocalMaxSpectrum(sp,f1,f2,df)
    #SiftSp=ShiftSpectrumUp(NewSp, f1, f2,2,5)
    SiftSp=ShiftSpectrumFullDwon(NewSp, f1, f2,band_shift,step_sift)
    #SiftSp=ShiftSpectrumFullUp(NewSp, f1, f2,500,3)
    sp=Ku*copy.deepcopy(SiftSp)
    #sp=MixSpectrum(Ku*copy.deepcopy(SiftSp), sp)
    fl=1
  if (i+10)%16==0:
    NewSp=GetLocalMaxSpectrum(sp,f1,f2,df)
    #SiftSp=ShiftSpectrumUp(NewSp, f1, f2,2,5)
    #SiftSp=ShiftSpectrumFullDwon(NewSp, f1, f2,500,3)
    SiftSp=ShiftSpectrumFullUp(NewSp, f1, f2,band_shift,step_sift)
    sp=Ku*copy.deepcopy(SiftSp)
    #sp=MixSpectrum(Ku*copy.deepcopy(SiftSp), sp)
    fl=1
  if (i+11)%16==0:
    NewSp=GetLocalMaxSpectrum(sp,f1,f2,df)
    #SiftSp=ShiftSpectrumUp(NewSp, f1, f2,2,5)
    SiftSp=ShiftSpectrumFullDwon(NewSp, f1, f2,band_shift,step_sift)
    #SiftSp=ShiftSpectrumFullUp(NewSp, f1, f2,500,3)
    sp=Ku*copy.deepcopy(SiftSp)
    #sp=MixSpectrum(Ku*copy.deepcopy(SiftSp), sp)
    fl=1    

  #if i%7==0:
  #  NewSp=GetLocalMaxSpectrum(sp,f1,f2,df)
  #  SiftSp=ShiftSpectrumUp(NewSp, f1, f2,220)
  #  sp=SiftSp
  #  fl=1
  #if (i+3)%7==0:
  #  NewSp=GetLocalMaxSpectrum(sp,f1,f2,df)
  #  SiftSp=ShiftSpectrumDown(NewSp, f1, f2,220)
  #  sp=SiftSp
  #  fl=1  

    
  #if fl==1:
  #  NewMarkSp[:,i]=sp
  #else:
  NewMarkSp[:,i]=sp
    
    

ys1 = GetSignalFromSpectrum(NewMarkSp, Step)    

wavio.write('C:/py_test_temp/orig_sig.wav', y/abs(y).max(), sr, sampwidth=2)
wavio.write('C:/py_test_temp/out_sig3.wav', ys1/abs(ys1).max(), sr, sampwidth=2)
print('done')


plt.plot(y/abs(y).max())
plt.plot(ys1/abs(ys1).max())
plt.show()

