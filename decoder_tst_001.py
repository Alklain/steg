import os
import sys
import numpy as np
import math
import cmath
import copy
import wavio
from sndmrk_site import *
import copy

# open file
#filename_in=sys.argv[1]
#filename_out=sys.argv[3]
#filename_mark=sys.argv[2]
# читаем файл


# open file
filename_in='C:/py_test_temp/Kassa.wav'
filename_out='output.txt'
# читаем файл





wav_dat = wavio.read(filename_in)
y1 = wav_dat.data[:,0]
sampwidth = wav_dat.sampwidth
sr = wav_dat.rate
# фрагмент сигнала ( чтобы удобней было)
#y1=y[180000:225000]

# параметры спектрального анализа
Nfft = 2048
Step = 512
# считаем матрицу спектрограммы
CompSp, LogSp, Step = GetMatrixSpectrumNoWin(y1, Nfft, Step )
# восстановили сигнал для проверки
ys = GetSignalFromSpectrum_hanning(CompSp, Step)

# вносим метки в диапазоне частот
f1 = 300
f2 = 600
# прием прием!!!
ysmod = y1
# вычисляем матрицу спектрограммы
CompSpR, LogSpR, Step = GetMatrixSpectrumNoWin(ysmod, Nfft, Step )
# хотим поменять в Ntm  окне анализа фазу. Шаг окна - Nstp
# детектирум биты
Ntpm = 9
Nstp = 6
lis_t1, lis_fi1 = DetectBit(CompSpR,Ntpm,sr,Step,f1, f2)
lis_t2, lis_fi2 = DetectBit(CompSpR,Ntpm+Nstp,sr,Step,f1, f2,1)
lis_t3, lis_fi3 = DetectBit(CompSpR,Ntpm+2*Nstp,sr,Step,f1, f2)
lis_t4, lis_fi4 = DetectBit(CompSpR,Ntpm+3*Nstp,sr,Step,f1, f2)

Out_dat = [1, 4, 5, 6]
print(Out_dat)

f = open(filename_out, 'w')
for i in range(len(Out_dat)):
    f.write(str(Out_dat[i]) + '\n')
f.close()


print('done')
