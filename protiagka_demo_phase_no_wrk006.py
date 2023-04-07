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
# считаем матрицу спектрограммы
CompSp, LogSp, Step = GetMatrixSpectrumNoWin(y1, Nfft, Step )
# восстановили сигнал для проверки
ys = GetSignalFromSpectrum_hanning(CompSp, Step)
# хотим поменять в Ntm  окне анализа фазу. Шаг окна - Nstp
Ntm = 15
Nstp = 6
# вносим метки в диапазоне частот
f1 = 300
f2 = 600
f3 = 900
f4 = 1200
ModSp = copy.deepcopy(CompSp)
ModSp = InsertBitToSpekMat(Ntm,ModSp,sr,Step,f1,f2,0)
ModSp = InsertBitToSpekMat(Ntm+Nstp,ModSp,sr,Step,f1,f2,1)
ModSp = InsertBitToSpekMat(Ntm+2*Nstp,ModSp,sr,Step,f1,f2,0)
ModSp = InsertBitToSpekMat(Ntm+3*Nstp,ModSp,sr,Step,f1,f2,0)

lgspM, mat_fiM = get_mat_for_pik(ModSp)
# формируем сигнал - модифицированый звук
ysmod = GetSignalFromSpectrum_hanning(ModSp, Step, 0)
# моделируем сдвиг во времени при приеме

ysmod = ysmod[0:]

# прием прием!!!

# вычисляем матрицу спектрограммы
CompSpR, LogSpR, Step = GetMatrixSpectrumNoWin(ysmod, Nfft, Step )

# детектирум биты
Ntpm = 15
lis_t1, lis_fi1 = DetectBit(CompSpR,Ntpm,sr,Step,f1, f2)
lis_t2, lis_fi2 = DetectBit(CompSpR,Ntpm+Nstp,sr,Step,f1, f2,1)
lis_t3, lis_fi3 = DetectBit(CompSpR,Ntpm+2*Nstp,sr,Step,f1, f2)
lis_t4, lis_fi4 = DetectBit(CompSpR,Ntpm+3*Nstp,sr,Step,f1, f2)

# строим графики
plt.figure()
plt.plot(lis_t1,lis_fi1)
plt.plot(lis_t2,lis_fi2)
plt.plot(lis_t3,lis_fi3)
plt.plot(lis_t4,lis_fi4)


plt.matshow(LogSp[0:40,:])
plt.matshow(LogSpR[0:40,:])
# сохраняем файлы
#wavio.write('C:/py_test_temp/or_sig.wav', ys/abs(ys).max(), sr, sampwidth=2)
#wavio.write('C:/py_test_temp/mod_sig_1.wav', ysmod/abs(ysmod).max(), sr, sampwidth=2)


print('done')
plt.show()
