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

y1=y[190000:195000]

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
sp=copy.deepcopy(CompSp[:,2])
sp2=copy.deepcopy(sp)

sp = copy.deepcopy(CompSp[:, 3])
sp3 = copy.deepcopy(sp)

# меняем фазу гармоники
tls = list(abs(sp2[:round(Nfft / 2)]))
Nmax = tls.index(max(tls))
Mod_Fi = SetComplexRotationFi(sp2[Nmax], 50)
sp2f=np.zeros(Nfft,dtype=complex)
sp2f = cange_dat_in_complex_spec(sp2f, Nmax, Mod_Fi)
#sp2f = cange_dat_in_complex_spec(sp2f, Nmax-1, 0)
#sp2f = cange_dat_in_complex_spec(sp2f, Nmax+1, 0)

#fidd=math.pi*2*(Step/((1/((Nmax/Nfft)*sr))*sr)-math.floor(Step/((1/((Nmax/Nfft)*sr))*sr)))
Oldfi=get_fi(sp2f[Nmax])
dop_fi=get_new_fi(Oldfi,Nmax,Step,Nfft,sr)

#dop_fi = ((((Nmax) * Step)/ Nfft) % (math.pi * 2)) * (180 / math.pi)
#dop_fi = (180 / math.pi)*fidd
print(dop_fi)
#dop_fi1 = (fs * (Step / sr) % (math.pi * 2)) * (180 / math.pi)

# меняем фазу гармоники
tls = list(abs(sp3[:round(Nfft / 2)]))
Nmax = tls.index(max(tls))
Mod_Fi = SetComplexRotationFi(sp3[Nmax], dop_fi)
sp3f=np.zeros(Nfft,dtype=complex)
sp3f = cange_dat_in_complex_spec(sp3f, Nmax, Mod_Fi)
#sp3f = cange_dat_in_complex_spec(sp3f, Nmax-1, 0)
#sp3f = cange_dat_in_complex_spec(sp3f, Nmax+1, 0)


tls = list(abs(sp3f[:round(Nfft / 2)]))
Nmax = tls.index(max(tls))

ysw2 = np.fft.ifft(sp2f)
wdat2 = ysw2.real# * Ham_win

ysw3 = np.fft.ifft(sp3f)
wdat3 = ysw3.real# * Ham_win

ytmp = np.zeros(Nfft + Step)
ytmp[len(ytmp) - Nfft:] = wdat3
plt.figure(i)
plt.plot(wdat2)
plt.plot(ytmp)
plt.show()

# get_fi(sp[Nmax])
print(get_fi(sp2f[Nmax]), get_fi(sp3f[Nmax]))
# оценка набега фазы

#print(dop_fi, dop_fi1)
print('ok')



#    NewMarkSp[:, i] = sp3
#    i+=1

#ys1 = GetSignalFromSpectrum(NewMarkSp, Step)

## детектировование сигнала
#print('start detection')
#CompSpD, LogSpD, Step = GetMatrixSpectrum(ys1, Nfft, Step )
#NewMarkSpD=copy.deepcopy(CompSp)
#for i in range(NumBlk):
#    sp = copy.deepcopy(CompSpD[:, i])
#    sp3 = copy.deepcopy(sp)
#    NewMarkSpD[:, i] = sp3

#ys2 = GetSignalFromSpectrum(NewMarkSpD, Step)

#plt.plot(ys-ys1)
#plt.plot(ys2-ys1)
#
#plt.plot(np.array(max_freq2))
#plt.plot(np.array(max_freq)+1)
#plt.figure()
#plt.plot(ys1)
#plt.show()
