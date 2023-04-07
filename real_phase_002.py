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
#filename='C:/py_test_temp/kladbische.mp3'
#filename='C:/py_test_temp/ttt1.m4a'

filename='C:/py_test_temp/ft2.m4a'
#filename='/content/drive/MyDrive/01-echo_a0987.wav'
from PIL import Image
# читаем файл
y, sr = librosa.load(filename,sr=None)

y1=y[150000:250000]
#plt.figure; plt.plot(y)
# думодуляция
Nsmp=len(y1)
tv = np.linspace(0, Nsmp/sr, Nsmp)
yop = np.sin(2 * math.pi * tv * (736))
Nfft=2048
Step = 512
CompSp, LogSp, Step = GetMatrixSpectrumNoWin(y1, Nfft, Step )
nTbit=round(0.03*sr)
nTb2=round(0.03*sr)

pls=int(0.1*sr)
det=[]
tv1=[]
while pls<Nsmp-3 * nTbit:
    s0=y1[pls:pls+nTb2]
    s1 = yop[pls:pls + nTb2]
    npm=np.mean(s0*s1)
    det.append(get_fi_from_2_sig_acos(s0,s1))
    tv1.append(pls/sr)
    pls+=nTbit

plt.matshow(LogSp[20:40,:])    
plt.figure()
plt.plot(tv,y1/max(y1));
plt.plot(tv,yop);
plt.figure()
plt.plot(tv1,det);

plt.show()

