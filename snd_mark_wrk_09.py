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
df=10
Af=5
K=21
nf1=int((830/sr)*Nfft)
nf3=int((2530/sr)*Nfft)

nf2=int((1840/sr)*Nfft)
nf4=int((4830/sr)*Nfft)
nf5=int((5930/sr)*Nfft)
# np.linspace(1000, 6000, 9)
#fv=[940, 1625, 2250, 2875, 3530, 4145, 4750, 5375, 6040]
fv = [6000, 6125, 6250, 6375, 6500, 6625, 6750, 6875, 7000]
FrMat=np.loadtxt(open("C:/pythonProject/matrix.txt","rb"),delimiter=",",skiprows=0)
print(FrMat)

n_time=0
bit_code=40
freq_code=FrMat[:,bit_code] # код байта
pos_sp=0
while pos_sp < NumBlk:
  sp=copy.deepcopy(CompSp[:,pos_sp])
  # считываем данные
  ft=int(freq_code[n_time])
  nf = int((fv[ft] / sr) * Nfft) # номре частоты в спектре
  sp[nf] = sp[nf] + Af
  sp[len(sp) - nf] = sp[nf]
  n_time+=1
  if n_time==20:
    n_time=0
  NewMarkSp[:,pos_sp]=sp
  NewMarkSp[:, pos_sp+1] = copy.deepcopy(CompSp[:,pos_sp+1])
  pos_sp+=2

ys1 = GetSignalFromSpectrum(NewMarkSp, Step)    

wavio.write('C:/py_test_temp/orig_sig.wav', y/abs(y).max(), sr, sampwidth=2)
wavio.write('C:/py_test_temp/p2.wav', ys1/abs(ys1).max(), sr, sampwidth=2)
print('done')


plt.plot(y/abs(y).max())
plt.plot(ys1/abs(ys1).max())
plt.show()

