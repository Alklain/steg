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
filename='C:/py_test_temp/Kassa.wav'
# читаем файл
y, sr = librosa.load(filename,sr=None)
# фрагмент сигнала ( чтобы удобней было)
y1=y[190000:205000]

Nsmp=len(y1)
# параметры спектрального анализа
Nfft = 2048
Step = 512
# считаем матрицу спектрограммы
CompSp, LogSp, Step = GetMatrixSpectrumNoWin(y1, Nfft, Step )
# восстановили сигнал для проверки
ys = GetSignalFromSpectrum_hanning(CompSp, Step)
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
NewFi=5
# хотим поменять в Ntm = 7 окне анализа фазу
Ntm = 7
# смотрим мгновенныей спектр
sp = copy.deepcopy(CompSp[:, Ntm])
tls = list(abs(sp[nfmin:nfmax]))
Num = nfmin+tls.index(max(tls))
print('num f max=',Num,' f = ', int((Num/Nfft)*sr),' Hz')
#plt.figure; plt.plot(20*np.log10(abs(sp)))

# меняем фазу в матрице спектра
NewSpMat=change_phase_in_matrix(CompSp, Num-7, Num+7, Ntm-3, Ntm+3, NewFi,Step, sr)

# строим графики и видим что матрица мзменена корректно
lgsp1, mat_fi1 = get_mat_for_pik(NewSpMat[Nfft-nfmax:Nfft,:])
#plt.matshow(lgsp1)
plt.matshow(mat_fi1)
# синтезируем сигнал по измененной матрице
ys2 = GetSignalFromSpectrum_hanning(NewSpMat, Step)
# снова смотрим матрицу спектра
CompSpD, LogSpD, StepD = GetMatrixSpectrumNoWin(ys2, Nfft, Step )
# строим графики и ничига не видим
lgsp2, mat_fi2 = get_mat_for_pik(CompSpD[Nfft-nfmax:Nfft,:])
plt.matshow(mat_fi2)
#plt.matshow(lgsp2)
#plt.matshow(mat_fi1)

#sp=CompSp[:,10]
#yst1 = my_test_ifft(sp)
#yst2 = np.fft.ifft(sp)

#plt.figure(1)
#plt.plot(yst1)
#plt.plot(yst2.real)
print('done')
plt.show()
