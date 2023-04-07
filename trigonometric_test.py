from sndmrk import *
import copy
import pandas as pd
import math
import cmath
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
sr=48000
fs=443 # частота синусоиды
Ts=1/fs # период синусоиды
Nsmp = int(sr * Ts)
tv = np.linspace(0, Nsmp/sr, Nsmp)
pi = math.pi
y = np.cos(2*pi*fs*tv)


trig_fun=[]
trig_acos=[]
trig_atan2=[]
det_out=[]
grad=[]
for fi in range(1,360):
    x1 = np.cos(2 * pi * fs * tv + (fi * pi / 180))
    x2 = np.sin(2 * pi * fs * tv + (fi * pi / 180))
    trig_atan2.append(get_fi_from_2_sig_atan2(x1, x2, y))
    trig_acos.append(get_fi_from_2_sig_acos(x1, y))
    grad.append(fi)

plt.plot(grad,trig_atan2)
#plt.plot(trig_fun,trig_acos)
plt.plot(grad,trig_acos)
plt.show()