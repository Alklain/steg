import matplotlib.pylab as plt
import wavio
import librosa
from sndmrk import *
import copy

# для начала - один период синусоиды

#fs=1000 # частота синусоиды
fs1=343 # частота синусоиды
fs2=1343 # частота синусоиды
fs3=2543 # частота синусоиды
Ts=1/fs1 # период синусоиды
sr=48000 # частота дискретизации

tv = np.linspace(0, Ts, int(Ts*sr))
y = (np.sin(2 * math.pi * tv * fs1)+np.sin(2 * math.pi * tv * fs2+math.pi/3)+
    np.sin(2 * math.pi * tv * fs3+math.pi))

#plt.plot(tv,y); plt.show()
#plt.plot(tv,y)

# теперь строим синусоиду на окне анализа
Nfft = 2048

T_Nfft = Nfft/sr

tv1 = np.linspace(0, T_Nfft, Nfft)
y1 = (np.sin(2 * math.pi * tv1 * fs1)+np.sin(2 * math.pi * tv1 * fs2+math.pi/3)+
    np.sin(2 * math.pi * tv1 * fs3+math.pi))

#plt.plot(tv1,y1); plt.show()

## теперь строим спектр
Ham_win = np.hamming(Nfft)
#wdat=y1*Ham_win
wdat=y1
sp=np.fft.fft(wdat)
lg_sp=20*np.log10(abs(sp))
fv=np.linspace(0, sr, Nfft)

#plt.plot(fv,lg_sp)

# ищем номер максимума
Nfmax1=round((fs1/sr)*Nfft)
Nfmax2=round((fs2/sr)*Nfft)
Nfmax3=round((fs3/sr)*Nfft)

#print(sp[Nfmax],sp[Nfft-Nfmax])
#print(get_fi(sp[Nfmax]))

# синтезируем спектр по максимуму

sp_3=np.zeros(Nfft,dtype=complex)
# вращаем фазу
Mod_Fi1 = SetComplexRotationFi(sp[Nfmax1], 90)
Mod_Fi2 = SetComplexRotationFi(sp[Nfmax2], 90)
Mod_Fi3 = SetComplexRotationFi(sp[Nfmax3], 90)

#Mod_Fi1 = SetComplexRotationFi(sp[Nfmax1], 180)
#Mod_Fi2 = SetComplexRotationFi(sp[Nfmax2], 180)
#Mod_Fi3 = SetComplexRotationFi(sp[Nfmax3], 180)

sp_3=cange_dat_in_complex_spec(sp_3,Nfmax1,Mod_Fi1)
sp_3=cange_dat_in_complex_spec(sp_3,Nfmax2,Mod_Fi2)
sp_3=cange_dat_in_complex_spec(sp_3,Nfmax3,Mod_Fi3)


#sp_3=cange_dat_in_complex_spec(sp_3,Nfmax-1,sp[Nfmax-1])
#sp_3=cange_dat_in_complex_spec(sp_3,Nfmax+1,sp[Nfmax+1])
ysw = np.fft.ifft(sp_3)
wdat1 = ysw.real
#wdat1 = ysw.real * Ham_win
marker_sig = np.zeros(Nfft)
marker_sig[round(Nfft/2)]=max(wdat1)
#plt.plot(tv1,wdat)
plt.figure
plt.plot(tv1,marker_sig)
plt.plot(tv1,abs(wdat1)); plt.show()
