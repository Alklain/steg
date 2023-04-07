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
#plt.plot(y)
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


nrf1=int((1000/sr)*Nfft)
nrf2=int((5000/sr)*Nfft)
temp_s=CompSp[:,150]
#plt.plot(20*np.log10(abs(sp)))


PosMaxSp=GetFerstPosMax(abs(temp_s[nrf1:nrf2]),lw=3)
for i in range(len(PosMaxSp)):
  PosMaxSp[i]+=nrf1

print(PosMaxSp)

#temp_s[int(len(temp_s)/2)]=0
plt.Figure()
plt.plot(abs(temp_s))

sp1=np.zeros(len(temp_s),dtype=complex)
## сборка

sp1=copy.deepcopy(temp_s)
sp1[min(PosMaxSp):max(PosMaxSp)]=0
sp1[len(sp1)-max(PosMaxSp):len(sp1)-min(PosMaxSp)]=0

#plt.Figure()
#plt.plot(1+abs(sp1))
#plt.plot(abs(CompSp[:,240]))


#sp1[np.array(PosMaxSp)]=sp[np.array(PosMaxSp)]
#sp1[len(sp1)-np.array(PosMaxSp)]=sp[len(sp1)-np.array(PosMaxSp)]

fi=complex(math.cos(math.pi/4),math.sin(math.pi/4)) # множитель поворота фазы 

num_f=0
listfi=[]
fid=0
for i in range(len(PosMaxSp)):
  if int(PosMaxSp[i])>int((3000/sr)*Nfft): # меняем фазу
    s_old_fi=temp_s[int(PosMaxSp[i])]
    fi=get_fi(PosMaxSp[i])
#    sp1[int(PosMaxSp[i])]=temp_s[int(PosMaxSp[i])]
#    sp1[len(sp1)-int(PosMaxSp[i])]=temp_s[len(temp_s)-int(PosMaxSp[i])]
    s_new_fi=ComplexRotationFi(listfi[num_f], s_old_fi, fid)
    print(get_fi(s_old_fi),get_fi(s_new_fi),listfi[num_f])
#    sp1[int(PosMaxSp[i])]=s_old_fi
#    sp1[len(sp1)-int(PosMaxSp[i])]=s_old_fi
    sp1[int(PosMaxSp[i])]=s_new_fi
    sp1[len(sp1)-int(PosMaxSp[i])]=s_new_fi
    
    num_f+=1
  else:  
    sp1[int(PosMaxSp[i])]=temp_s[int(PosMaxSp[i])]
    sp1[len(sp1)-int(PosMaxSp[i])]=temp_s[len(temp_s)-int(PosMaxSp[i])]
    listfi.append(get_fi(sp1[int(PosMaxSp[i])]))
    print(int(PosMaxSp[i]))
'''
    fi=get_fi(PosMaxSp[i])
    if num_f<len(listfi):
      s_new_fi=ComplexRotationFi(listfi[num_f], sp1[int(PosMaxSp[i])], fid)
      sp1[int(PosMaxSp[i])]=s_new_fi
      sp1[len(sp1)-int(PosMaxSp[i])]=s_new_fi
      num_f+=1
    else:
      sp1[int(PosMaxSp[i])]=temp_s[int(PosMaxSp[i])]
      sp1[len(sp1)-int(PosMaxSp[i])]=temp_s[len(temp_s)-int(PosMaxSp[i])]    
    
    #print(int(PosMaxSp[i]))
'''
#  else: ## сохраняем фазу 
#    sp1[int(PosMaxSp[i])]=temp_s[int(PosMaxSp[i])]
#    sp1[len(sp1)-int(PosMaxSp[i])]=temp_s[len(temp_s)-int(PosMaxSp[i])]
#    listfi.append(get_fi(sp1[int(PosMaxSp[i])]))
#    print(int(PosMaxSp[i]))
  


#sp1[PosMaxSp]=sp[PosMaxSp]
#sp1=sp

plt.plot(abs(sp1))

ysw=np.fft.ifft(temp_s)
ysw1=np.fft.ifft(sp1)
plt.figure()
plt.plot(ysw.real)
plt.plot(ysw1.real)

## детектирование
Ham_win = np.hamming ( Nfft )
y1=ysw1.real*Ham_win
#y1=ysw.real*Ham_win
sp22=np.fft.fft(y1)

plt.figure()
plt.plot(abs(sp22))

PosMaxSp22=GetFerstPosMax(abs(sp22[nrf1:nrf2]),lw=3)
for i in range(len(PosMaxSp22)):
  PosMaxSp22[i]+=nrf1


for i in range(len(PosMaxSp22)):
  if int(PosMaxSp22[i])>int((3000/sr)*Nfft): # вычистаем меняем фазу
    
    fi=get_fi(sp22[int(PosMaxSp22[i])])
    print(int(fi),int(listfi[num_f]),abs(int(fi)-int(listfi[num_f]))) 
    num_f+=1
  else:      
    listfi.append(get_fi(sp22[int(PosMaxSp22[i])]))
    print(int(PosMaxSp22[i]))




print('done')
plt.show()




