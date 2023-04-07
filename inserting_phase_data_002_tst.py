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
from scipy import signal
# open file
filename='C:/py_test_temp/Kassa.wav'
# читаем файл
y, sr = librosa.load(filename,sr=None)
# фрагмент сигнала ( чтобы удобней было)
y1=y[180000:255000]

f = open('input.txt')
mark=[]
for line in f:
    if len(line)>1:
        mark.append(int(line))
f.close()
print(mark)


# параметры спектрального анализа
Nfft = 2048
Step = 512
#y1=y
Nsmp=len(y1)
tv = np.linspace(0, Nsmp/sr, Nsmp)
fs = 18.8*(sr/Nfft)
fs1 = 73.5*(sr/Nfft)
#y1 = 1*np.cos(2 * math.pi * tv * fs)+1*np.cos(2 * math.pi * tv * fs1)
# добавили шум
s = np.random.normal(0, 0.01, len(y1))




# считаем матрицу спектрограммы
CompSp, LogSp, Step = GetMatrixSpectrumNoWin(y1, Nfft, Step )
# восстановили сигнал для проверки
ys = GetSignalFromSpectrum_hanning(CompSp, Step)
# хотим поменять в Ntm  окне анализа фазу. Шаг окна - Nstp
Ntm = 25
Nstp = 6
# вносим метки в диапазоне частот
print(fs, fs1)
f1 = mark[0]
f2 = mark[1]
f3 = mark[2]
f4 = mark[3]


ModSp = copy.deepcopy(CompSp)
# число столбцов ( окон аналиа )
NumBlk=len(CompSp[:][1])
tm=10

data =  mark[4:]
Ldat = len(data)
p_dat = 0

while tm < NumBlk-10:
    tm=tm+1
    if tm%8 == 0:
        if data[p_dat] == 1:
            ModSp = InsertBitToSpekMat(tm,ModSp,sr,Step,f1,f2,0)
        else:
            ModSp = InsertBitToSpekMat(tm,ModSp,sr,Step,f3,f4,0)
        p_dat+=1
        if p_dat==Ldat:
            p_dat=0


        #ModSp = InsertBitToSpekMat(tm,ModSp,sr,Step,f1,f2,0)
        #    ModSp = InsertBitToSpekMat(tm,ModSp,sr,Step,900,2000,0)
        #ModSp = InsertBitToSpekMat(tm,ModSp,sr,Step,f3,f4,0)
        #ModSp = InsertBitToSpekMat(tm,ModSp,sr,Step,f4,f5,0)
        #ModSp = InsertBitToSpekMat(tm,ModSp,sr,Step,1100,2000,0)
        #ModSp = InsertBitToSpekMat(tm,ModSp,sr,Step,f6,f7,0)
        #ModSp = InsertBitToSpekMat(tm,ModSp,sr,Step,f7,f8,0)




        
#ModSp = InsertBitToSpekMat(Ntm,ModSp,sr,Step,f3,f4,0)
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

tm=10

while tm < NumBlk-10:
    tm=tm+1
    if tm%8 == 0:
        #lis_t1, lis_fi1, phz1 = DetectBit(CompSpR,tm,sr,Step,f1, f2,0)
        lis_t1, lis_fi1, phz1 = DetectBit(CompSpR,tm,sr,Step,f1,f2,0)
        lis_t2, lis_fi2, phz2 = DetectBit(CompSpR,tm,sr,Step,f3,f4,0)
        #lis_t1, lis_fi1, phz1 = DetectBit(CompSpR,tm,sr,Step,900,2000,0)
        #lis_t1, lis_fi1, phz1 = DetectBit(CompSpR,tm,sr,Step,f3, f4,0)
        #lis_t1, lis_fi1, phz1 = DetectBit(CompSpR,tm,sr,Step,f4, f5,0)
        #lis_t1, lis_fi1, phz1 = DetectBit(CompSpR,tm,sr,Step,f5, f6,0)
        plt.figure()
        #plt.plot(lis_t1,lis_fi1)
        #plt.plot(phz1)
        #plt.plot(phz2)
        plt.plot(lis_t1,lis_fi1)
        plt.plot(lis_t2,lis_fi2)
        #lis_t1, lis_fi1, phz1 = DetectBit(CompSpR,tm,sr,Step,f6, f7,0)
        #lis_t1, lis_fi1, phz1 = DetectBit(CompSpR,tm,sr,Step,f7, f8,0)


#
#Ntpm = Ntm
#lis_t1, lis_fi1, phz1 = DetectBit(CompSpR,Ntpm,sr,Step,f1, f2,0)
#lis_t2, lis_fi2, phz2 = DetectBit(CompSpR,Ntpm,sr,Step,f3, f4,0)
#lis_t3, lis_fi3 = DetectBit(CompSpR,Ntpm+2*Nstp,sr,Step,f1, f2)
#lis_t4, lis_fi4 = DetectBit(CompSpR,Ntpm+3*Nstp,sr,Step,f1, f2)

# строим графики
#plt.figure()
#plt.plot(lis_t1,lis_fi1)
#plt.plot(lis_t2,lis_fi2)
#plt.figure()
#plt.plot(phz1)
#plt.figure()
#plt.plot(phz2)
#plt.plot(ysmod)

#plt.plot(lis_t3,lis_fi3)
#plt.plot(lis_t4,lis_fi4)


#plt.matshow(LogSp[0:40,:])
#plt.matshow(LogSpR[0:40,:])
# сохраняем файлы
#wavio.write('C:/py_test_temp/sin_tst_2.wav', y1/abs(y1).max(), sr, sampwidth=2)
wavio.write('C:/py_test_temp/kasmlf5.wav', ysmod/abs(ysmod).max(), sr, sampwidth=2)


print('done')
plt.show()
