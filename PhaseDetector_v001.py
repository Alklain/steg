import matplotlib.pylab as plt
import wavio
import librosa
from sndmrk import *
import copy
#filename='C:/py_test_temp/sin_3fi_180.wav'
#filename='C:/py_test_temp/Kassa.wav'
#filename='C:/py_test_temp/Kassa_fi_1.wav'
filename='C:/py_test_temp/Kassa_fi_3.wav'
y, sr = librosa.load(filename,sr=None)
Nfft = 2048
Step = 512
CompSp, LogSp, Step = GetMatrixSpectrum(y, Nfft, Step )
ys = GetSignalFromSpectrum(CompSp, Step)

NumBlk = len(CompSp[:][1])
phase_list=[]
max_freq=[]
nrf = int((1000 / sr) * Nfft)
for i in range(NumBlk):
    sp = copy.deepcopy(CompSp[:, i])
    #pos_max = get_max_soectrum_in_window(sp, sr, 800, 4000)  # нашли максимум в полосе частот
    #phase_list.append(get_fi(sp[pos_max]))
    phase_list.append(get_fi(sp[nrf]))
    #max_freq.append(pos_max)

plt.plot(phase_list)
#plt.figure()
#plt.plot(max_freq)
plt.show()
