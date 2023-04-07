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
from mpl_toolkits.mplot3d import Axes3D
#filename='C:/py_test_temp/kladbische.mp3'
filename='C:/py_test_temp/ttt1.m4a'
#filename='/content/drive/MyDrive/01-echo_a0987.wav'
from PIL import Image
# читаем файл
y, sr = librosa.load(filename,sr=None)

y1=y[40000:]
#plt.figure; plt.plot(y)
# думодуляция
Nsmp=len(y1)
tv = np.linspace(0, Nsmp/sr, Nsmp)
yop = np.sin(2 * math.pi * tv * 440)

nTbit=round(0.05*sr)
nTb2=round(0.02*sr)

pls=int(0.1*sr)
while pls<Nsmp-3 * nTbit:
    s0=y1[pls:pls+nTb2]
    s1 = y1[pls:pls + nTb2]
    s2=y1[pls+nTbit:pls+nTb2+nTbit]
    s3 = y1[pls + 2*nTbit:pls + nTb2 + 2*nTbit]
    s4 = y1[pls + 3 * nTbit:pls + nTb2 + 3 * nTbit]
    npm=np.mean(s0*s1)
    print(pls/sr,np.mean(s0*s2)/npm,np.mean(s0*s3)/npm,np.mean(s0*s4)/npm)
    pls+=nTbit


#ymlt = y1 * yop
#plt.figure; plt.plot(ymlt)
#y1=y

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
nfmin=10
nfmax=40
#plt.figure(1); plt.matshow(LogSp[2000:2048,:])

lgsp, mat_fi = get_mat_for_pik(CompSp[2000:2048,:])
#plt.figure(0); plt.matshow(lgsp)
#plt.figure(1); plt.matshow(mat_fi)
#plt.matshow(lgsp)
#plt.matshow(mat_fi)
NewFi=5
sp = copy.deepcopy(CompSp[:, 7])
tls = list(abs(sp[nfmin:nfmax]))
Num = nfmin+tls.index(max(tls))
print('num f max=',Num)
plt.figure; plt.plot(abs(sp))

lgsp0, mat_fi0 = get_mat_for_pik(CompSp[Nfft-nfmax:Nfft,:])
plt.matshow(lgsp0)
NewSpMat=change_phase_in_matrix(CompSp, Num-1, Num+1, 6, 8, NewFi,Step, sr)

lgsp1, mat_fi1 = get_mat_for_pik(NewSpMat[Nfft-nfmax:Nfft,:])
#plt.matshow(lgsp1)
plt.matshow(mat_fi1)


#ys2 = GetSignalFromSpectrum(NewSpMat, Step)
ys2 = GetSignalFromSpectrum_hanning(NewSpMat, Step)
CompSpD, LogSpD, StepD = GetMatrixSpectrumNoWin(ys2, Nfft, Step )
lgsp2, mat_fi2 = get_mat_for_pik(CompSpD[Nfft-nfmax:Nfft,:])
plt.matshow(mat_fi2)
plt.matshow(lgsp2)
#plt.matshow(mat_fi1)

#sp=CompSp[:,10]
#yst1 = my_test_ifft(sp)
#yst2 = np.fft.ifft(sp)

#plt.figure(1)
#plt.plot(yst1)
#plt.plot(yst2.real)


#Num = 30
num_t=5
sp_mat=copy.deepcopy(CompSp[:, num_t-4:num_t+4])
sp = copy.deepcopy(CompSp[:, num_t])
tls = list(abs(sp[nfmin:nfmax]))
Num = nfmin+tls.index(max(tls))
NewFi=9
#NewSpMat=change_phase_in_matrix(sp_mat, Num, NewFi, Step, sr)

#sp_mat=change_phase_in_matrix(sp_mat, Num, 144, Step, sr)

#fim=np.linspace(0, 360, 36)
#for i in range(len(fim)):
#    NewFi=fim[i]
#    NewSpMat=change_phase_in_matrix(sp_mat, Num, NewFi, Step, sr)

#fim=np.linspace(0, 360, 36)
#for i in range(len(fim)):
#    NewFi=fim[i]
#    NewSpMat=change_phase_in_matrix(sp_mat, Num, NewFi, Step, sr)


#print(dop_fi, dop_fi1)
print('ok')



#    NewMarkSp[:, i] = sp3
#    i+=1


#plt.figure(3); plt.plot(delt)

#plt.plot(det_fi)
#plt.plot(delt)
#plt.figure(3); plt.plot(abs(abs(np.array(delt))-180)) #plt.show()
#plt.plot(abs(abs(np.array(delt1))-180))

#plt.figure(3); plt.plot(Zero_cross_delt)
#plt.plot(ys1); plt.show()
#plt.figure(4); plt.plot(ys[TimeShift:]-ys1) # plt.show()
#plt.plot(ys2-ys1)
#
#plt.plot(np.array(max_freq2))
#plt.plot(np.array(max_freq)+1)
#plt.figure()
#plt.plot(ys1)
plt.show()
