
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
filename_in='C:/py_test_temp/Kassa_m1.wav'
filename_out='C:/py_test_temp/Kassa_test_m2.wav'
filename_mark='input.txt'
# читаем файл

f = open(filename_mark)
mark=[]
for line in f:
    if len(line)>1:
        mark.append(int(line))
f.close()
f1 = mark[0]
f2 = mark[1]
f3 = mark[2]
f4 = mark[3]
pwr = mark[4]

wav_dat = wavio.read(filename_in)
y = wav_dat.data[:,0]
sampwidth = wav_dat.sampwidth
sr = wav_dat.rate

# фрагмент сигнала ( чтобы удобней было)
y1=y
# параметры спектрального анализа
Nfft = 2048
Step = 512


# считаем матрицу спектрограммы
CompSp, LogSp, Step = GetMatrixSpectrumNoWin(y1, Nfft, Step )
# восстановили сигнал для проверки

# хотим поменять в Ntm  окне анализа фазу. Шаг окна - Nstp

Ntm = 15
Nstp = 8
# вносим метки в диапазоне частот
#f1 = 200
#f2 = 600
#f3 = 1200
#f4 = 2600

ModSp = copy.deepcopy(CompSp)
# число столбцов ( окон аналиа )
NumBlk=len(CompSp[:][1])
tm=Ntm
# данные из ФАЙЛА
data =  mark[5:]
Ldat = len(data)
p_dat = 0

while tm < NumBlk-10:
    tm=tm+1
    if tm % Nstp == 0:
        if data[p_dat] == 1:
            ModSp = InsertBitToSpekMat(tm,ModSp,sr,Step,f1,f2,0,pwr)
        else:
            ModSp = InsertBitToSpekMat(tm,ModSp,sr,Step,f3,f4,0,pwr)
        p_dat+=1
        if p_dat==Ldat:
            p_dat=0

# формируем сигнал - модифицированый звук
ysmod = GetSignalFromSpectrum_hanning(ModSp, Step, 0)
ysmod = ysmod[0:len(y1)]
# моделируем сдвиг во времени при приеме

# сохраняем файлы
wavio.write(filename_out, ysmod/abs(ysmod).max(), sr, sampwidth=2)
print('done')



