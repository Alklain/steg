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
y, sr = librosa.load(filename,sr=None)
wavio.write('C:/py_test_temp/mod_sig_1.wav', y/abs(y).max(), sr, sampwidth=2)