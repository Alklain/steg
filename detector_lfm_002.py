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
from scipy import signal
import copy
## принятый файл
#filename='C:/py_test_temp/lfm5.wav'
filename='C:/py_test_temp/lfm5_3m_RMX3393.wav'

y, sr = librosa.load(filename,sr=None)

mark_0 = get_lfm_sig(6000, 10000, 0.1, sr, 2048)
mark_1 = np.flipud(mark_0)
#y1=y[0:100000]
print('start')
yf1=signal.lfilter(mark_0,1,y)
yf2=signal.lfilter(mark_1,1,y)

plt.plot(yf1)
plt.plot(yf2)
#plt.plot(yf2-yf1)
plt.show()
