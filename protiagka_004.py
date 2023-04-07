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

y1=y[190000:280000]

Nsmp=len(y1)
#fs=1000 # частота синусоиды
fs=343 # частота синусоиды
Ts=1/fs # период синусоиды
tv = np.linspace(0, Nsmp/sr, Nsmp)
#y1 = np.sin(2 * math.pi * tv * fs)

Nfft = 2048
Step = 512
#CompSp, LogSp, Step = GetMatrixSpectrum(y1, Nfft, Step )
CompSp, LogSp, Step = GetMatrixSpectrumNoWin(y1, Nfft, Step )
#ys = GetSignalFromSpectrum(CompSp, Step)
ys = GetSignalFromSpectrum_hanning(CompSp, Step)
NewMarkSp=copy.deepcopy(CompSp)
NumBlk=len(CompSp[:][1])
i=2
Ham_win = np.hanning ( Nfft )
NTbit=9
nfmin=30
nfmax=40

nfmin1=130
nfmax1=140
#Num = 30
while i<NumBlk-10:
    i+=1
    if i%NTbit==0:
        sp = copy.deepcopy(CompSp[:, i])
        #tls = list(abs(sp[:round(Nfft / 2)]))
        tls = list(abs(sp[nfmin:nfmax]))
        Num = nfmin+tls.index(max(tls))

        tls = list(abs(sp[nfmin1:nfmax1]))
        Num1 = nfmin1+tls.index(max(tls))
        #Num = 50
        NewFi = 0
        sp = copy.deepcopy(CompSp[:, i-2]) ## важно - ps меньше нуля, меняем фазу в прошлых срезах
        sp1 = copy.deepcopy(sp)
        sp = copy.deepcopy(CompSp[:, i-1])
        sp2 = copy.deepcopy(sp)
        sp = copy.deepcopy(CompSp[:, i])
        sp3 = copy.deepcopy(sp)
        sp1f, sp2f = change_phase_in_2_wind(sp1, sp2, Num, Nfft, NewFi, Step, sr)
        sp1f, sp2f = change_phase_in_2_wind(sp1, sp2, Num1, Nfft, NewFi, Step, sr)
        #sp1f, sp2f = change_phase_in_2_wind(sp1, sp2, Num-1, Nfft, NewFi, Step, sr)
        #sp1f, sp2f = change_phase_in_2_wind(sp1, sp2, Num+1, Nfft, NewFi, Step, sr)
        sp2f, sp3f = change_phase_in_2_wind(sp2, sp3, Num, Nfft, NewFi, Step, sr)
        sp2f, sp3f = change_phase_in_2_wind(sp2, sp3, Num1, Nfft, NewFi, Step, sr)
        #sp2f, sp3f = change_phase_in_2_wind(sp2, sp3, Num-1, Nfft, NewFi, Step, sr)
        #sp2f, sp3f = change_phase_in_2_wind(sp2, sp3, Num+1, Nfft, NewFi, Step, sr)
        NewMarkSp[:, i-2] = sp1f
        NewMarkSp[:, i-1] = sp2f
        NewMarkSp[:, i] = sp3f





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


# get_fi(sp[Nmax])
#print(get_fi(sp2f[Num]), get_fi(sp5f[Num]))
# оценка набега фазы

#print(dop_fi, dop_fi1)
print('ok')



#    NewMarkSp[:, i] = sp3
#    i+=1

ys1_orig = GetSignalFromSpectrum(NewMarkSp, Step)
## моделирование сдвига фазы при приеме сигнала
TimeShift=512
ys1=ys1_orig[TimeShift:]
#ys1=ys1_orig
## детектировование сигнала
print('start detection')
CompSpD, LogSpD, Step = GetMatrixSpectrum(ys1, Nfft, Step )
#NewMarkSpD=copy.deepcopy(CompSp)
det_fi=[]
det_fi1=[]
#Num = 30
NumBlk=len(CompSpD[:][1])
for i in range(NumBlk):
    sp = copy.deepcopy(CompSpD[:, i])
    spd = copy.deepcopy(sp)
    tls = list(abs(spd[nfmin:nfmax]))
    Num = nfmin+tls.index(max(tls))
    tls = list(abs(sp[nfmin1:nfmax1]))
    Num1 = nfmin1+tls.index(max(tls))
    #Num = 50
    # читаем фазу гармоники
    det_fi.append(get_fi(spd[Num]))
    det_fi1.append(get_fi(spd[Num1]))
#    NewMarkSpD[:, i] = sp3

pos=0
delt=[]
delt1=[]
Zero_cross_delt=[]
while pos<len(det_fi)-NTbit:
    delt.append(det_fi[pos]-det_fi[pos+NTbit])
    delt1.append(det_fi1[pos]-det_fi1[pos+NTbit])
    Zero_cross_delt.append(0)
    if pos>1:
        if (delt[pos-1]*delt[pos])<0:
            Zero_cross_delt[pos]=1
    pos+=1


#ys2 = GetSignalFromSpectrum(NewMarkSpD, Step)

erfi = abs(abs(np.array(delt))-180)
spt=np.fft.fft(erfi)
spt[0]=0
plt.figure(1); plt.plot(abs(spt)) #plt.show()

erfi = delt
spt=np.fft.fft(erfi)
spt[0]=0
plt.figure(2); plt.plot(abs(spt)) #plt.show()

#plt.figure(3); plt.plot(delt)

#plt.plot(det_fi)
#plt.plot(delt)
plt.figure(3); plt.plot(abs(abs(np.array(delt))-180)) #plt.show()
plt.plot(abs(abs(np.array(delt1))-180))

#plt.figure(3); plt.plot(Zero_cross_delt)
#plt.plot(ys1); plt.show()
plt.figure(4); plt.plot(ys[TimeShift:]-ys1) # plt.show()
#plt.plot(ys2-ys1)
#
#plt.plot(np.array(max_freq2))
#plt.plot(np.array(max_freq)+1)
#plt.figure()
#plt.plot(ys1)
plt.show()
