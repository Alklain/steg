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
Tm=1
Nsmp = int(sr * Tm)
tv = np.linspace(0, Nsmp/sr, Nsmp)
ys = np.zeros_like(tv)
ys_mark = np.zeros_like(tv)
Nfft = 2048
Step = 512
Numf = round((fs/sr)*Nfft)
# синтезируем сигнал на первые 5 шагов
garmonica = 1
fi0=90 # начальная фаза сигнала
dop_fi = get_new_fi(0, Numf, Step, Nfft, sr)
garmonica = SetComplexRotationFi(garmonica, -dop_fi+fi0)
pi=math.pi
f_gen = (Numf/Nfft)*sr # частота из сетки фурье
y_gen = np.cos(2*pi*f_gen*tv+fi0*pi/180)
sp = np.zeros(Nfft, dtype=complex)
sp = cange_dat_in_complex_spec(sp, Numf, garmonica) # виртуальный спектр, не участвует в рассчете нужен для иниализации цикла
Ham_win = np.hanning(Nfft)
pos=0
nTstep=0
for i in range(21):
    Oldfi = get_fi(sp[Numf])
    dop_fi = get_new_fi(Oldfi, Numf, Step, Nfft, sr)
    # меняем фазу гармоники
    Mod_Fi = SetComplexRotationFi(sp[Numf], dop_fi)
    sp = cange_dat_in_complex_spec(sp, Numf, Mod_Fi)
    #sp = cange_dat_in_complex_spec(sp, Numf, garmonica) # нет протяжки
    # синтезируем сигнал
    ysw = np.fft.ifft(sp)
    wdat = ysw.real * Ham_win
    ys[pos:pos + Nfft] = ys[pos:pos + Nfft] + wdat
    pos += Step
    nTstep+=1
    ys_mark[pos]=1

# сморим сигнал
CompSp, LogSp, Step = GetMatrixSpectrumNoWin(ys, Nfft, Step )

# коридор частот, ГЦ
fmin = 100
fmax = 1000
# номера гармоник в спектре
nfmin=int((fmin/sr)*Nfft)
nfmax=int((fmax/sr)*Nfft)

# построим картинки
#lgsp, mat_fi = get_mat_for_pik(CompSp[Nfft-nfmax:Nfft,:])
lgsp, mat_fi = get_mat_for_pik(CompSp[:nfmax,:])
#plt.figure(1); plt.plot(tv,ys)
#plt.matshow(lgsp)

# теперь самое интересно - мы ходим помнять фазу.
# текущий срез сигнала - смотрим фазу

# хотим поменять в Ntm окне анализа фазу
Ntm = nTstep
# смотрим мгновенныей спектр
sp = copy.deepcopy(CompSp[:, Ntm])
tls = list(abs(sp[nfmin:nfmax]))
Num = nfmin+tls.index(max(tls))
print('num f max=',Num,' f = ', int((Num/Nfft)*sr),' Hz')

# смотрим мгновенную фазу на нанало работы
Oldfi = get_fi(CompSp[Num,Ntm])
print('old fi',Oldfi,' окно номер',nTstep )

# пошли поварачивать фазу цель = за три такта на 4 повернуть как надо

Fi_new=180 # я хочу эту новую фазу чтоы в окне номер nTstep+3 и дальше она была. а в иделе начиная с nTstep + 2
Oldfi = get_fi(CompSp[Num,Ntm])
# зададим частоты 
Fi_r1=0
Fi_r2=0
Fi_r3=0

# первый раунд
Mod_Fi = SetComplexRotationFi(garmonica, Fi_r1)
sp1 = cange_dat_in_complex_spec(CompSp[:,Ntm+1], Num, Mod_Fi)
# добавляем к сигналу, пока в ручную
ysw = np.fft.ifft(sp1)
wdat = ysw.real * Ham_win
ys[pos:pos + Nfft] = ys[pos:pos + Nfft] + wdat
pos += Step
nTstep+=1
ys_mark[pos]=1.25

# второй раунд
Mod_Fi = SetComplexRotationFi(garmonica, Fi_r2)
sp2 = cange_dat_in_complex_spec(CompSp[:,Ntm+2], Num, Mod_Fi)

# добавляем к сигналу, пока в ручную
ysw = np.fft.ifft(sp2)
wdat = ysw.real * Ham_win
ys[pos:pos + Nfft] = ys[pos:pos + Nfft] + wdat
pos += Step
nTstep+=1
ys_mark[pos]=1.5
# третий раунд
Mod_Fi = SetComplexRotationFi(garmonica, Fi_r3)
sp3 = cange_dat_in_complex_spec(CompSp[:,Ntm+3], Num, Mod_Fi)

# добавляем к сигналу, пока в ручную
ysw = np.fft.ifft(sp3)
wdat = ysw.real * Ham_win
ys[pos:pos + Nfft] = ys[pos:pos + Nfft] + wdat
pos += Step
nTstep+=1
ys_mark[pos]=2
# а теперь снова цикл
sp = sp3
for i in range(21):
    Oldfi = get_fi(sp[Num])
    dop_fi = get_new_fi(Oldfi, Num, Step, Nfft, sr)
    # меняем фазу гармоники
    Mod_Fi = SetComplexRotationFi(sp[Num], dop_fi)
    sp = cange_dat_in_complex_spec(sp, Num, Mod_Fi)
    #sp = cange_dat_in_complex_spec(sp, Numf, garmonica) # нет протяжки
    # синтезируем сигнал
    ysw = np.fft.ifft(sp)
    wdat = ysw.real * Ham_win
    ys[pos:pos + Nfft] = ys[pos:pos + Nfft] + wdat
    pos += Step
    nTstep+=1
    ys_mark[pos] = 1


## а теперь смотрим сигнал снова
CompSpD, LogSpD, Step = GetMatrixSpectrumNoWin(ys, Nfft, Step )
lgsp1, mat_fi1 = get_mat_for_pik(CompSpD[:nfmax,:])

# смотрим фазу сигнала в нужный момент

plt.plot(tv,ys)
plt.plot(tv,ys_mark/512)
plt.plot(tv,y_gen/512)
## а теперь попробуем демодулировать и выделить фазу по двум синусам.
ps=0
fi_det=[]
t_det=[]

NTper=int((1/f_gen)*sr)

while ps<Tm:
    nps=int(sr * ps)
    x=ys[nps:nps+NTper]
    y = y_gen[nps:nps + NTper]
    fi_det.append((180/pi)*math.acos(get_my_corr(x,y)))
    t_det.append(ps)
    ps+=0.01
plt.figure()
plt.plot(t_det,fi_det)


#plt.figure; plt.plot(20*np.log10(abs(sp)))
#plt.matshow(mat_fi)
#plt.matshow(mat_fi1)
#plt.figure()
#plt.plot(mat_fi1[Num,:])
plt.show()
