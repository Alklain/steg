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
# константы
pi=math.pi
eps = np.finfo(float).eps

# генерим сигнал

filename='C:/py_test_temp/Kassa.wav'
# читаем файл
y, sr = librosa.load(filename,sr=None)
# фрагмент сигнала ( чтобы удобней было)
y1=y[190000:255000]
Nsmp=len(y1)


#tv = np.linspace(0, Nsmp/sr, Nsmp)
#y1 = np.cos(2 * math.pi * tv * 1340)

# параметры спектрального анализа
Nfft = 2048
Step = 512



# номер максимальной гармоники
Ntm=10

fs=736
Nst=2
Nst2=10
tv1 = np.linspace(0, Nst*Step/sr, Nst*Step)
tv2 = np.linspace(0, Nst2*Step/sr, Nst2*Step)
#ymark = np.cos(2 * math.pi * tv * fs)

ymark=np.hstack([np.cos(2 * pi * tv2 * fs), np.cos(2 * pi * tv1 * fs + (len(tv2)/sr)*2*pi*fs), np.cos(2 * pi * tv2 * fs + ((len(tv2)+len(tv1))/sr)*2*pi*fs)])


# считаем матрицу спектрограммы
CompSp, LogSp, Step = GetMatrixSpectrumNoWin(ymark, Nfft, Step )
Phase_sp=get_phase_sep(CompSp)    #
# номер максимальной гармоники
Ntm=5
# коридор частот, ГЦ
fmin = 200
fmax = 2000
# номера гармоник в спектре
nfmin=int((fmin/sr)*Nfft)
nfmax=int((fmax/sr)*Nfft)
sp = copy.deepcopy(CompSp[:, Ntm])
tls = list(abs(sp[nfmin:nfmax]))
Num = nfmin+tls.index(max(tls))

filtsp = get_part_of_spec_mat(CompSp,Num-1,Num+2,4,16)

ys_cp = GetSignalFromSpectrum_hanning(filtsp, Step)
CompSpNew, LogSpNew, Step = GetMatrixSpectrumNoWin(ys_cp, Nfft, Step )



wavio.write('C:/py_test_temp/fi_rv_180.wav', ys_cp/abs(ys_cp).max(), sr, sampwidth=2)



plt.matshow(LogSpNew[Num-10:Num+11,2:])
plt.matshow(LogSp[Num-10:Num+11,:])
Nstp=len(ymark)
tvp = np.linspace(0, Nstp/sr, Nstp)
plt.figure()
plt.plot(tvp,ymark)
plt.figure()
Nstp=len(ys_cp)
tvp = np.linspace(0, Nstp/sr, Nstp)
plt.plot(tvp,ys_cp)
## детектируем метку


y_c = np.cos(2 * math.pi * tvp * fs)
y_s = np.sin(2 * math.pi * tvp * fs)

det = (ys_cp*y_c)

plt.figure()
plt.plot(tvp,y_c)
plt.plot(tvp,ys_cp)
plt.figure()
plt.plot(tvp,det)
get_plot_spectrum(ymark,sr)
get_plot_spectrum(ymark,sr)
get_plot_spectrum(ys_cp,sr)
plt.show()
