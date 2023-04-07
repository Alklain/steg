import os
import sys
import numpy as np
import math
import cmath
import copy
import wavio
#from sndmrk_site import *
from sndmrk import *
import copy
import matplotlib.pylab as plt
# open file
#filename_in=sys.argv[1]
#filename_out=sys.argv[3]
#filename_mark=sys.argv[2]
# читаем файл


# open file
filename_in='C:/py_test_temp/kassa_mod_2f_lf.wav'
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
#CompSp, LogSp, Step = GetMatrixSpectrumNoWin(y1, Nfft, Step )
# восстановили сигнал для проверки
#ys = GetSignalFromSpectrum_hanning(CompSp, Step)

# вносим метки в диапазоне частот
f1 = 200
f2 = 250
f3 = 300
f4 = 350
f5 = 400
f6 = 450
f7 = 500
f8 = 600
# прием прием!!!
ysmod = y1
# вычисляем матрицу спектрограммы
CompSpR, LogSpR, Step = GetMatrixSpectrumNoWin(ysmod, Nfft, Step )
# хотим поменять в Ntm  окне анализа фазу. Шаг окна - Nstp
# детектирум биты
NumBlk=len(CompSpR[:][1])
tm=10

#while tm < NumBlk-10:
while tm < 40:    
    tm=tm+1
    if tm%8 == 0:
        #lis_t1, lis_fi1, phz1 = DetectBit(CompSpR,tm,sr,Step,f1, f2,0)
        #lis_t1, lis_fi1, phz1 = DetectBit(CompSpR,tm,sr,Step,f2, f3,0)
        #lis_t1, lis_fi1, phz1 = DetectBit(CompSpR,tm,sr,Step,f3, f4,0)
        #lis_t1, lis_fi1, phz1 = DetectBit(CompSpR,tm,sr,Step,f4, f5,0)
        lis_t1, lis_fi1, phz1 = DetectBit(CompSpR,tm,sr,Step,f5, f6,1)
        #plt.figure()
        #plt.plot(lis_t1, lis_fi1)
        #lis_t1, lis_fi1, phz1 = DetectBit(CompSpR,tm,sr,Step,f6, f7,0)
        #lis_t1, lis_fi1, phz1 = DetectBit(CompSpR,tm,sr,Step,f7, f8,0)

#Out_dat = [1, 4, 5, 6]
#print(Out_dat)
plt.show()
#f = open(filename_out, 'w')
#for i in range(len(Out_dat)):
#    f.write(str(Out_dat[i]) + '\n')
#f.close()


#print('done')
