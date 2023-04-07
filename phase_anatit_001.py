import matplotlib.pylab as plt
import wavio
import librosa
from sndmrk import *
import copy

# для начала - один период синусоиды

#fs=1000 # частота синусоиды
fs=1343 # частота синусоиды
Ts=1/fs # период синусоиды
sr=48000 # частота дискретизации

tv = np.linspace(0, Ts, int(Ts*sr))
y = np.sin(2 * math.pi * tv * fs)

#plt.plot(tv,y); plt.show()


# теперь строим синусоиду на окне анализа
Nfft = 2048

T_Nfft = Nfft/sr

tv1 = np.linspace(0, T_Nfft, Nfft)
y1 = np.sin(2 * math.pi * tv1 * fs)

#plt.plot(tv1,y1); plt.show()

## теперь строим спектр
Ham_win = np.hamming(Nfft)
#wdat=y1*Ham_win
wdat=y1
sp=np.fft.fft(wdat)
lg_sp=20*np.log10(abs(sp))
fv=np.linspace(0, sr, Nfft)

#plt.plot(fv,lg_sp); plt.show()

# ищем номер максимума
Nfmax=round((fs/sr)*Nfft)

print(sp[Nfmax],sp[Nfft-Nfmax])
print(get_fi(sp[Nfmax]))

# синтезируем спектр по максимуму

sp_3=np.zeros(Nfft,dtype=complex)
# вращаем фазу
Mod_Fi = SetComplexRotationFi(sp[Nfmax], 90)
sp_3=cange_dat_in_complex_spec(sp_3,Nfmax,Mod_Fi)

#sp_3=cange_dat_in_complex_spec(sp_3,Nfmax-1,sp[Nfmax-1])
#sp_3=cange_dat_in_complex_spec(sp_3,Nfmax+1,sp[Nfmax+1])
ysw = np.fft.ifft(sp_3)
#wdat1 = ysw.real
wdat1 = ysw.real * Ham_win
marker_sig = np.zeros(Nfft)
marker_sig[round(Nfft/2)]=max(wdat1)
#plt.plot(tv1,wdat)
plt.plot(tv1,marker_sig)
plt.plot(tv1,abs(wdat1)); plt.show()
