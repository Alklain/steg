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
## принятый файл
filename='C:/py_test_temp/p1.wav'
y, sr = librosa.load(filename,sr=None)
Nfft = 2048
Step = 512
CompSp, LogSp, Step = GetMatrixSpectrum(y, Nfft, Step )
NumBlk=len(CompSp[:][1])

NewMarkSp=copy.deepcopy(CompSp)
#fv=[940, 1625, 2250, 2875, 3530, 4145, 4750, 5375, 6040]
fv = [6000, 6125, 6250, 6375, 6500, 6625, 6750, 6875, 7000]
FrMat=np.loadtxt(open("C:/pythonProject/matrix.txt","rb"),delimiter=",",skiprows=0)
print(FrMat)

n_time=0
bit_code=40
freq_code=FrMat[:,bit_code] # код байта
resalt=[]
for i in range(NumBlk):
  sp=copy.deepcopy(CompSp[:,i])
  resalt.append(PulseDetect(sp, fv, Nfft,sr))

print(resalt)