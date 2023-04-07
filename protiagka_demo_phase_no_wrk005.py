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
from PIL import Image



# open file
pi=math.pi
filename='C:/py_test_temp/Kassa.wav'
# читаем файл
y, sr = librosa.load(filename,sr=None)
# фрагмент сигнала ( чтобы удобней было)
y1=y[190000:215000]
Nsmp=len(y1)


tv = np.linspace(0, Nsmp/sr, Nsmp)
#y1 = np.cos(2 * math.pi * tv * 440)

# параметры спектрального анализа
Nfft = 2048
Step = 512
# считаем матрицу спектрограммы
CompSp, LogSp, Step = GetMatrixSpectrumNoWin(y1, Nfft, Step )
# восстановили сигнал для проверки
#ys = GetSignalFromSpectrum_hanning(CompSp, Step)
# сейчас будем менять фазу пытаться
# скопирум спектр в временую матрицу
#NewMarkSp=copy.deepcopy(CompSp)
# число столбцов ( окон аналиа )
#NumBlk=len(CompSp[:][1])
# коридор частот, ГЦ
#fmin = 200
#fmax = 1000
# номера гармоник в спектре
#nfmin=int((fmin/sr)*Nfft)
#nfmax=int((fmax/sr)*Nfft)

# построим картинки
#lgsp, mat_fi = get_mat_for_pik(CompSp[Nfft-nfmax:Nfft,:])
#plt.matshow(mat_fi)
# новая фаза
#NewFi=90
# хотим поменять в Ntm  окне анализа фазу
Ntm = 33
#wavio.write('C:/py_test_temp/or_sig.wav', ys/abs(ys).max(), sr, sampwidth=2)
#wavio.write('C:/py_test_temp/mod_sig.wav', ysmod/abs(ysmod).max(), sr, sampwidth=2)

# вносим метки
f1 = 200
#f2 = 200
f3 = 300
ModSp, MLogSp = InsertBitToSpekMat(Ntm,CompSp,sr,Step,f1,f3,0)
#Ntm1 = 17
#wrk_sp = copy.deepcopy(ModSp)
#ModSp1, MLogSp1 = InsertBitToSpekMat(Ntm1,wrk_sp,sr,Step,f1,f3,0)
lgspM, mat_fiM = get_mat_for_pik(ModSp)
# формируем сигнал

ysmod = GetSignalFromSpectrum_hanning(ModSp, Step, 0)

# прием прием!!!
# считываем метки

CompSpR, LogSpR, Step = GetMatrixSpectrumNoWin(ysmod, Nfft, Step )
Ntpm = Ntm
#sp0 = copy.deepcopy(CompSp[:, Ntpm])
sp1 = copy.deepcopy(CompSpR[:, Ntpm])
#get_plot_spectrum(sp0,sr)
#get_plot_spectrum(sp1,sr)
#plt.figure()
#plt.plot(20*np.log10(abs(sp0[:round(Nfft/2)])))
#plt.plot(20*np.log10(abs(sp1[:round(Nfft/2)])))
#paths = 'C:/py_test_temp/mds1.wav'
#wavio.write(paths, ysmod/abs(ysmod).max(), sr, sampwidth=2)

fNum=get_max_soectrum_in_window(sp1, sr, f1, f3)
#fNum=19
df = 2
dt = 5
sp_frame = select_from_matr_sp(CompSpR, fNum, Ntpm, df, dt)
ysf = GetSignalFromSpectrum_hanning(sp_frame, Step, 0)
Nt=len(ysf)
tv1 = np.linspace(0,Nt/sr, Nt)
fs = get_freq_from_num(fNum, Nfft, sr)
fi = get_fi(CompSpR[fNum,Ntpm-dt])
yop = np.cos(2 * math.pi * tv1 * fs+fi*pi/180)

plt.figure()
plt.plot(tv1,ysf)
plt.plot(tv1,yop*max(ysf))


# смотрим мгновенныей спектр
#sp = copy.deepcopy(CompSp[:, Ntm])
#tls = list(abs(sp[nfmin:nfmax]))
#fNumM = nfmin+tls.index(max(tls))
# смотрим сигнал в окретности 
#sp_frameM=select_from_matr_sp(ModSp,fNum,Ntm,df,dt)
#ysfM = GetSignalFromSpectrum_hanning(sp_frameM, Step)

#plt.matshow(mat_fiM[0:40,:])
#plt.matshow(abs(sp_frame[0:40,:]))
#plt.matshow(abs(FragSp[0:40,:]))

plt.matshow(LogSp[0:40,:])
plt.matshow(MLogSp[0:40,:])
#plt.figure()
#plt.plot(ys)

#Tm=len(ytmp)
#tv1=np.linspace(0, Tm/sr, Tm)
#fs = (fNumM/Nfft)*sr
#yopr = np.cos(2 * pi * tv1 * fs)
#plt.figure()
#plt.plot(tv1,ytmp)
#plt.plot(tv1,yopr*max(ytmp))
#plt.figure()
#plt.plot(tv1,ysf)
#plt.plot(tv1,yopr*max(ysf))
#plt.figure()
#plt.plot(tv1,ysfM)
#plt.plot(tv1,yopr*max(ysfM))
#get_plot_spectrum(ysf,sr)
'''
#plt.figure; plt.plot(20*np.log10(abs(sp)))

# меняем фазу в матрице спектра
Fi_r2 = 20
Fi_r3 = 0
out_fi=[]
print('start')
er=1000
for fi1 in range(1,360,10):
    print('fi1=',fi1)
    for fi2 in range(1,360,10):
        print('fi2=', fi2)
        for fi3 in range(1,360,10):
            Fi_r1 = fi1
            Fi_r2 = fi2
            Fi_r3 = fi3
            TNewSpMat, NewSfi=insert_new_fi_into_sp_matr(Fi_r3, Fi_r2, Fi_r1, CompSp[:,Ntm-3:Ntm+4],Num, sr, Step)
            er=abs(NewFi-NewSfi)
            #print(er)
            out_fi.append([Fi_r1, Fi_r2, Fi_r3, NewSfi])
            if er<2:
                print(Fi_r1, Fi_r2, Fi_r3, NewSfi)
                break
        if er < 2:
            break
    if er < 2:
        break

rezmat=np.array(out_fi)
#plt.figure()
#plt.plot(rezmat[:,1])



CompSpD=copy.deepcopy(CompSp)
CompSpD[:,Ntm-3:Ntm+4]=TNewSpMat
print('check1=',get_fi(TNewSpMat[Num, 3]))
print('check=',get_fi(CompSp[Num, Ntm]))
print('check2=',get_fi(CompSpD[Num, Ntm]))
#

#NewSpMat=change_phase_in_matrix(CompSp, Num-7, Num+7, Ntm-3, Ntm+3, NewFi,Step, sr)

# строим графики и видим что матрица мзменена корректно
#lgsp1, mat_fi1 = get_mat_for_pik(CompSpD[Nfft-nfmax:Nfft,:])
#plt.matshow(lgsp1)
#plt.matshow(mat_fi1)
# синтезируем сигнал по измененной матрице
ys2 = GetSignalFromSpectrum_hanning(CompSpD, Step)
# снова смотрим матрицу спектра
CompSpD1, LogSpD, StepD = GetMatrixSpectrumNoWin(ys2, Nfft, Step )
print('check3=',get_fi(CompSpD1[Num, Ntm]))
# строим графики и ничига не видим
lgsp2, mat_fi2 = get_mat_for_pik(CompSpD1[Nfft-nfmax:Nfft,:])
#plt.matshow(mat_fi2)
#plt.matshow(lgsp2)
#plt.matshow(mat_fi1)

#sp=CompSp[:,10]
#yst1 = my_test_ifft(sp)
#yst2 = np.fft.ifft(sp)

#plt.figure(1)

## а теперь попробуем демодулировать и выделить фазу по двум синусам.
ps = 0
fi_det = []
fi_det1 = []
fi_det2 = []
t_det = []

f_gen = (Num/Nfft)*sr # частота из сетки фурье
y_gen = np.cos(2*pi*f_gen*tv)
y_gens = np.sin(2*pi*f_gen*tv)

NTper = int((1 / f_gen) * sr)

while ps < (len(ys2)/sr):
    nps = int(sr * ps)
    x1= y1[nps:nps + NTper]
    x = ys2[nps:nps + NTper]
    y = y_gen[nps:nps + NTper]
    y_S = y_gens[nps:nps + NTper]
    fi_det.append(get_fi_from_2_sig_acos(y, x))
    fi_det1.append(get_fi_from_2_sig_atan2(y, y_S, x))
    fi_det2.append(get_fi_from_2_sig_atan2(y, y_S, x1))
    t_det.append(ps)
    ps += 0.01

#Out_fi_new = np.mean(fi_det[27:45])
#Out_fi_new1 = np.mean(fi_det1[27:45])
plt.figure()
plt.plot(tv,y1)
plt.plot(tv,ys2[:len(tv)])
plt.figure()
plt.plot(t_det,fi_det2)
plt.plot(t_det,fi_det1)

#plt.plot(yst2.real)
'''
print('done')
plt.show()
