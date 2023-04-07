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
#filename='C:/py_test_temp/kladbische.mp3'
#filename='C:/py_test_temp/Kassa.wav'
#filename1='C:/py_test_temp/fi_90.wav'
#filename2='C:/py_test_temp/fi_0.wav'

#filename1='C:/py_test_temp/Kassa_180.wav'
filename1='C:/py_test_temp/Kassa_90.wav'
filename2='C:/py_test_temp/Kassa.wav'
#filename='/content/drive/MyDrive/01-echo_a0987.wav'
from PIL import Image
# читаем файл
y1, sr = librosa.load(filename1,sr=None)
y2, sr = librosa.load(filename2,sr=None)

plt.plot(y1)
plt.plot(y1-y2)
plt.show()
wavio.write('C:/py_test_temp/comp_Kassa.wav', (y1-y2)/abs(y1-y2).max(), sr, sampwidth=2)