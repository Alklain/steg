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
y1=y[190000:255000]
Nsmp=len(y1)


tv = np.linspace(0, Nsmp/sr, Nsmp)
y1 = np.cos(2 * math.pi * tv * 1340)

# параметры спектрального анализа
Nfft = 2048
Step = 512
# считаем матрицу спектрограммы
CompSp, LogSp, Step = GetMatrixSpectrumNoWin(y1, Nfft, Step )
Phase_sp=get_phase_sep(CompSp)
# номер максимальной гармоники
Ntm=10
# коридор частот, ГЦ
fmin = 200
fmax = 2000
# номера гармоник в спектре
nfmin=int((fmin/sr)*Nfft)
nfmax=int((fmax/sr)*Nfft)
sp = copy.deepcopy(CompSp[:, Ntm])
tls = list(abs(sp[nfmin:nfmax]))
Num = nfmin+tls.index(max(tls))



# принудительо сделали потоянную фазу
CompSp1=set_const_phase_on_nf(CompSp,Num,Step,sr,-1)
#CompSp1=set_const_phase_on_nf(CompSp1,25,Step,sr,90)

# восстановили сигнал для проверки
ys = GetSignalFromSpectrum_hanning(CompSp, Step)
ys_cp = GetSignalFromSpectrum_hanning(CompSp1, Step)
CompSpNew, LogSpNew, Step = GetMatrixSpectrumNoWin(ys_cp, Nfft, Step )
Phase_spNew=get_phase_sep(CompSpNew)

lgsp, mat_fi = get_mat_for_pik(CompSpNew)

plt.matshow(LogSp[0:120,:])

plt.matshow(LogSpNew[0:120,:])
plt.matshow(mat_fi[0:120,:])
plt.matshow(Phase_spNew[0:120,:])
plt.figure()
plt.plot(mat_fi[Num,:]);
get_plot_spectrum(mat_fi[Num,:],sr/Step)
plt.figure()
plt.plot(y1)
plt.plot(ys_cp)
plt.show()
''''
# сейчас будем менять фазу пытаться
# скопирум спектр в временую матрицу
NewMarkSp=copy.deepcopy(CompSp)
# число столбцов ( окон аналиа )
NumBlk=len(CompSp[:][1])
# коридор частот, ГЦ
fmin = 200
fmax = 1000
# номера гармоник в спектре
nfmin=int((fmin/sr)*Nfft)
nfmax=int((fmax/sr)*Nfft)

# построим картинки
lgsp, mat_fi = get_mat_for_pik(CompSp[Nfft-nfmax:Nfft,:])
plt.matshow(mat_fi)
# новая фаза
NewFi=3
CompSpD=copy.deepcopy(CompSp)

# хотим поменять в Ntm  окне анализа фазу
Ntm1 = 9
# смотрим мгновенныей спектр
sp = copy.deepcopy(CompSp[:, Ntm1])
tls = list(abs(sp[nfmin:nfmax]))
Num1 = nfmin+tls.index(max(tls))
TNewSpMat=insert_phase_to_spectr_mat(CompSp[:,Ntm1-3:Ntm1+4],Num1,sr,Step,NewFi,0)
CompSpD[:,Ntm1-3:Ntm1+4]=TNewSpMat

# хотим поменять в Ntm  окне анализа фазу
Ntm2 = 16
# смотрим мгновенныей спектр
sp = copy.deepcopy(CompSp[:, Ntm2])
tls = list(abs(sp[nfmin:nfmax]))
Num2 = nfmin+tls.index(max(tls))
TNewSpMat=insert_phase_to_spectr_mat(CompSp[:,Ntm2-3:Ntm2+4],Num2,sr,Step,NewFi,0)
CompSpD[:,Ntm2-3:Ntm2+4]=TNewSpMat

# хотим поменять в Ntm  окне анализа фазу
Ntm3 = 23
# смотрим мгновенныей спектр
sp = copy.deepcopy(CompSp[:, Ntm3])
tls = list(abs(sp[nfmin:nfmax]))
Num3 = nfmin+tls.index(max(tls))
TNewSpMat=insert_phase_to_spectr_mat(CompSp[:,Ntm3-3:Ntm3+4],Num3,sr,Step,NewFi,0)
CompSpD[:,Ntm3-3:Ntm3+4]=TNewSpMat



#print('check1=',get_fi(TNewSpMat[Num, 3]))
print('check2=',get_fi(CompSpD[Num1, Ntm1]),get_fi(CompSpD[Num2, Ntm2]),
      get_fi(CompSpD[Num3, Ntm3]))
#
#print('num f max=',Num,' f = ', int((Num/Nfft)*sr),' Hz')
#plt.figure; plt.plot(20*np.log10(abs(sp)))
#NewSpMat=change_phase_in_matrix(CompSp, Num-7, Num+7, Ntm-3, Ntm+3, NewFi,Step, sr)

# строим графики и видим что матрица мзменена корректно
#lgsp1, mat_fi1 = get_mat_for_pik(CompSpD[Nfft-nfmax:Nfft,:])
#plt.matshow(lgsp1)
#plt.matshow(mat_fi1)
# синтезируем сигнал по измененной матрице
ys2 = GetSignalFromSpectrum_hanning(CompSpD, Step)
# снова смотрим матрицу спектра
CompSpD1, LogSpD, StepD = GetMatrixSpectrumNoWin(ys2, Nfft, Step )
print('check3=',get_fi(CompSpD1[Num1, Ntm1]),get_fi(CompSpD1[Num2, Ntm2]),
      get_fi(CompSpD1[Num3, Ntm3]))
# строим графики и ничига не видим
lgsp2, mat_fi2 = get_mat_for_pik(CompSpD1[Nfft-nfmax:Nfft,:])
plt.matshow(mat_fi2)
plt.matshow(lgsp2)
#plt.matshow(mat_fi)

#sp=CompSp[:,10]
#yst1 = my_test_ifft(sp)
#yst2 = np.fft.ifft(sp)

#plt.figure(1)

## а теперь попробуем демодулировать и выделить фазу по двум синусам.
#ps = 0
#fi_det = []
#fi_det1 = []
#fi_det2 = []
#t_det = []

#f_gen = (Num/Nfft)*sr # частота из сетки фурье
#y_gen = np.cos(2*pi*f_gen*tv)
#y_gens = np.sin(2*pi*f_gen*tv)

#NTper = int((1 / f_gen) * sr)

#while ps < (len(ys2)/sr):
#    nps = int(sr * ps)
#    x1= y1[nps:nps + NTper]
#    x = ys2[nps:nps + NTper]
#    y = y_gen[nps:nps + NTper]
#    y_S = y_gens[nps:nps + NTper]
#    fi_det.append(get_fi_from_2_sig_acos(y, x))
#    fi_det1.append(get_fi_from_2_sig_atan2(y, y_S, x))
#    fi_det2.append(get_fi_from_2_sig_atan2(y, y_S, x1))
#    t_det.append(ps)
#    ps += 0.01

#Out_fi_new = np.mean(fi_det[27:45])
#Out_fi_new1 = np.mean(fi_det1[27:45])
#plt.figure()
#plt.plot(tv,y1)
#plt.plot(tv,ys2[:len(tv)])
#plt.figure()
#plt.plot(t_det,fi_det2)
#plt.plot(t_det,fi_det1)

#plt.plot(yst2.real)
print('done')
plt.show()
'''
