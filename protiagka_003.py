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

y1=y[190000:210000]

Nsmp=len(y1)
#fs=1000 # частота синусоиды
fs=343 # частота синусоиды
Ts=1/fs # период синусоиды
tv = np.linspace(0, Nsmp/sr, Nsmp)
y1 = np.sin(2 * math.pi * tv * fs)

Nfft = 2048
Step = 512
#CompSp, LogSp, Step = GetMatrixSpectrum(y1, Nfft, Step )
CompSp, LogSp, Step = GetMatrixSpectrumNoWin(y1, Nfft, Step )
#ys = GetSignalFromSpectrum(CompSp, Step)
ys = GetSignalFromSpectrum_hanning(CompSp, Step)
NewMarkSp=copy.deepcopy(CompSp)
NumBlk=len(CompSp[:][1])
i=0
Ham_win = np.hanning ( Nfft )


# СМОРИМ СПЕКТРАЛЬНЫЕ СРЕЗЫ
sp=copy.deepcopy(CompSp[:,5])


tls = list(abs(sp[:round(Nfft / 2)]))
Num = tls.index(max(tls))
NewFi = 50
sp1=copy.deepcopy(sp)

sp = copy.deepcopy(CompSp[:, 6])
sp2 = copy.deepcopy(sp)

sp = copy.deepcopy(CompSp[:, 7])
sp3 = copy.deepcopy(sp)

sp=copy.deepcopy(CompSp[:,15])
sp4=copy.deepcopy(sp)

sp = copy.deepcopy(CompSp[:, 16])
sp5 = copy.deepcopy(sp)

sp = copy.deepcopy(CompSp[:, 17])
sp6 = copy.deepcopy(sp)

sp1f, sp2f = change_phase_in_2_wind(sp1,sp2,Num,Nfft,NewFi,Step,sr)
sp2f, sp3f = change_phase_in_2_wind(sp2,sp3,Num,Nfft,NewFi,Step,sr)

sp4f, sp5f = change_phase_in_2_wind(sp4,sp5,Num,Nfft,NewFi,Step,sr)
sp5f, sp6f = change_phase_in_2_wind(sp5,sp6,Num,Nfft,NewFi,Step,sr)
# меняем фазу гармоники
#tls = list(abs(sp2[:round(Nfft / 2)]))
#Nmax = tls.index(max(tls))
#Mod_Fi = SetComplexRotationFi(sp2[Nmax], 50)
#sp2f = cange_dat_in_complex_spec(sp2, Nmax, Mod_Fi)
#Oldfi=get_fi(sp2f[Nmax])
#dop_fi=get_new_fi(Oldfi,Nmax,Step,Nfft,sr)
# меняем фазу гармоники
#tls = list(abs(sp3[:round(Nfft / 2)]))
#Nmax = tls.index(max(tls))
#Mod_Fi = SetComplexRotationFi(sp3[Nmax], dop_fi)
#sp3f = cange_dat_in_complex_spec(sp3, Nmax, Mod_Fi)



# меняем фазу гармоники
#tls = list(abs(sp4[:round(Nfft / 2)]))
#Nmax = tls.index(max(tls))
#Mod_Fi = SetComplexRotationFi(sp4[Nmax], 50)
#sp4f = cange_dat_in_complex_spec(sp4, Nmax, Mod_Fi)
#Oldfi=get_fi(sp4f[Nmax])
#dop_fi=get_new_fi(Oldfi,Nmax,Step,Nfft,sr)
# меняем фазу гармоники
#tls = list(abs(sp5[:round(Nfft / 2)]))
#Nmax = tls.index(max(tls))
#Mod_Fi = SetComplexRotationFi(sp5[Nmax], dop_fi)
#sp5f = cange_dat_in_complex_spec(sp5, Nmax, Mod_Fi)



#tls = list(abs(sp3f[:round(Nfft / 2)]))
#Nmax = tls.index(max(tls))

#ysw2 = np.fft.ifft(sp2f)
#wdat2 = ysw2.real# * Ham_win

#ysw3 = np.fft.ifft(sp3f)
#wdat3 = ysw3.real# * Ham_win

#ytmp = np.zeros(Nfft + Step)
#ytmp[len(ytmp) - Nfft:] = wdat3
#plt.figure(i)
#plt.plot(wdat2)
#plt.plot(ytmp)
#plt.show()

NewMarkSp[:,5]=sp1f
NewMarkSp[:,6]=sp2f
NewMarkSp[:,7]=sp3f

NewMarkSp[:,15]=sp4f
NewMarkSp[:,16]=sp5f
NewMarkSp[:,17]=sp6f

# get_fi(sp[Nmax])
print(get_fi(sp2f[Num]), get_fi(sp5f[Num]))
# оценка набега фазы

#print(dop_fi, dop_fi1)
print('ok')



#    NewMarkSp[:, i] = sp3
#    i+=1

ys1 = GetSignalFromSpectrum(NewMarkSp, Step)

## детектировование сигнала
print('start detection')
CompSpD, LogSpD, Step = GetMatrixSpectrum(ys1, Nfft, Step )
#NewMarkSpD=copy.deepcopy(CompSp)
det_fi=[]
for i in range(NumBlk):
    sp = copy.deepcopy(CompSpD[:, i])
    spd = copy.deepcopy(sp)
    # читаем фазу гармоники
    det_fi.append(get_fi(spd[Num]))
#    NewMarkSpD[:, i] = sp3

pos=0
delt=[]
while pos<len(det_fi)-11:
    delt.append(det_fi[pos]-det_fi[pos+10])
    pos+=1


#ys2 = GetSignalFromSpectrum(NewMarkSpD, Step)

plt.plot(det_fi)
plt.plot(delt)
plt.plot(abs(np.array(delt)))
#plt.plot(ys1)
#plt.plot(ys-ys1)
#plt.plot(ys2-ys1)
#
#plt.plot(np.array(max_freq2))
#plt.plot(np.array(max_freq)+1)
#plt.figure()
#plt.plot(ys1)
plt.show()
