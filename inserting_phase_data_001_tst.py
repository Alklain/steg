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
y1=y[180000:225000]
# параметры спектрального анализа
Nfft = 2048
Step = 512

Nsmp=len(y1)
tv = np.linspace(0, Nsmp/sr, Nsmp)
fs = 18.8*(sr/Nfft)
fs1 = 73.5*(sr/Nfft)
#y1 = 1*np.cos(2 * math.pi * tv * fs)+1*np.cos(2 * math.pi * tv * fs1)
# добавили шум
s = np.random.normal(0, 0.01, len(y1))
#y1=y1+s



# считаем матрицу спектрограммы
CompSp, LogSp, Step = GetMatrixSpectrumNoWin(y1, Nfft, Step )
# восстановили сигнал для проверки
ys = GetSignalFromSpectrum_hanning(CompSp, Step)
# хотим поменять в Ntm  окне анализа фазу. Шаг окна - Nstp
Ntm = 25
Nstp = 6
# вносим метки в диапазоне частот
print(fs, fs1)
f1 = 200
f2 = 600
f3 = 1200
f4 = 2600
ModSp = copy.deepcopy(CompSp)
ModSp = InsertBitToSpekMat(Ntm,ModSp,sr,Step,f1,f2,1)
ModSp = InsertBitToSpekMat(Ntm,ModSp,sr,Step,f3,f4,0)
#ModSp = InsertBitToSpekMat(Ntm+2*Nstp,ModSp,sr,Step,f1,f2,0)
#ModSp = InsertBitToSpekMat(Ntm+3*Nstp,ModSp,sr,Step,f1,f2,0)

lgspM, mat_fiM = get_mat_for_pik(ModSp)
# формируем сигнал - модифицированый звук
ysmod = GetSignalFromSpectrum_hanning(ModSp, Step, 0)
# моделируем сдвиг во времени при приеме

ysmod = ysmod[0:]

# прием прием!!!

# вычисляем матрицу спектрограммы
CompSpR, LogSpR, Step = GetMatrixSpectrumNoWin(ysmod, Nfft, Step )

# детектирум биты
Ntpm = Ntm
lis_t1, lis_fi1, phz1 = DetectBit(CompSpR,Ntpm,sr,Step,f1, f2,0)
lis_t2, lis_fi2, phz2 = DetectBit(CompSpR,Ntpm,sr,Step,f3, f4,0)
#lis_t3, lis_fi3 = DetectBit(CompSpR,Ntpm+2*Nstp,sr,Step,f1, f2)
#lis_t4, lis_fi4 = DetectBit(CompSpR,Ntpm+3*Nstp,sr,Step,f1, f2)

# строим графики
plt.figure()
plt.plot(lis_t1,lis_fi1)
plt.plot(lis_t2,lis_fi2)
plt.figure()
plt.plot(phz1)
plt.figure()
plt.plot(phz2)
#plt.plot(ysmod)

#plt.plot(lis_t3,lis_fi3)
#plt.plot(lis_t4,lis_fi4)


plt.matshow(LogSp[0:40,:])
plt.matshow(LogSpR[0:40,:])
# сохраняем файлы
#wavio.write('C:/py_test_temp/sin_tst_2.wav', y1/abs(y1).max(), sr, sampwidth=2)
#wavio.write('C:/py_test_temp/mod_sin_tst_2.wav', ysmod/abs(ysmod).max(), sr, sampwidth=2)


print('done')
plt.show()
