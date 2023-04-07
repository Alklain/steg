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
## принятый файл
filename='C:/py_test_temp/p1.wav'
y, sr = librosa.load(filename,sr=None)
Nfft = 2048
Step = 512
CompSp, LogSp, Step = GetMatrixSpectrum(y, Nfft, Step )
NumBlk=len(CompSp[:][1])

# строим спектр
fmin=600
fmax=6000
nfmax=int((fmax / sr) * Nfft)
nfmin=int((fmin / sr) * Nfft)

sp=copy.deepcopy(CompSp[:,180])


wrksp=sp[nfmin:nfmax]

# ищем окрестности первых трех максимумов спектра
listsp=list(abs(wrksp))
posm1=listsp.index(max(listsp))
print(posm1)

df1=400 # ширинп полосы частот от главного максимума
ndf1=int((df1 / sr) * Nfft)
ps1=posm1-ndf1
if ps1<0:
    ps1=0

ps2=posm1+ndf1
print(ps1,ps2)

# теперь в зоне от максимума до границы ищем минимум

listsp2=listsp[ps2:]
posm2=listsp2.index(max(listsp2))
print(posm2+ps2)
ps3=posm2+ps2+ndf1

# теперь ищем в зоне после 2 максимума
listsp3=listsp[ps3:]
posm3=listsp3.index(max(listsp3))
print(posm3+ps3)

# теперь есть три частоты
spt=int(((posm2-posm1)+(posm3-posm2))/2)
nf1=posm1+spt+nfmin
nf2=posm2+spt+nfmin
nf3=posm3+spt+nfmin
sp1=sp*0
sp1[nf1]=abs(sp[nf1])+5
sp1[nf2]=abs(sp[nf2])+5
sp1[nf3]=abs(sp[nf3])+5

plt.figure(1,clear=True)
#plt.plot(20*np.log10(abs(sp[nfmin:nfmax])))
plt.plot((abs(sp[nfmin:nfmax])))
plt.plot((abs(sp1[nfmin:nfmax])))
plt.show()

#NewMarkSp=copy.deepcopy(CompSp)
#fv=[940, 1625, 2250, 2875, 3530, 4145, 4750, 5375, 6040]
#fv = [6000, 6125, 6250, 6375, 6500, 6625, 6750, 6875, 7000]
#FrMat=np.loadtxt(open("C:/pythonProject/matrix.txt","rb"),delimiter=",",skiprows=0)
#print(FrMat)

#n_time=0
#bit_code=40
#freq_code=FrMat[:,bit_code] # код байта
#resalt=[]
#for i in range(NumBlk):
#  sp=copy.deepcopy(CompSp[:,i])
#  resalt.append(PulseDetect(sp, fv, Nfft,sr))
#print(resalt)