import numpy as np
import math
import cmath
import copy
import matplotlib as mpl
import matplotlib.pylab as plt
from scipy import signal
import wavio
from sndmrk import *


y=get_lfm_sig(6000, 10000, 0.1, 48000, 2048)
y1=np.zeros_like(y)
y2=np.hstack([y1, y, y1, np.flipud(y), y1  ])

# filtering

#yf = signal.filtfilt(y, 1, y2)
#yf1=signal.lfilter(np.flipud(y),1,y2)
#yf2=signal.lfilter(y,1,y2)
wavio.write('C:/py_test_temp/lfm.wav', y2/abs(y2).max(), 48000, sampwidth=2)
print('done')

#plt.plot(yf1)
#plt.plot(yf2)
#plt.show()

