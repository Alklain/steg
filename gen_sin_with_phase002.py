from sndmrk import *
import copy
import pandas as pd
import math
import cmath
import random
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt

def temp_test_form_new_fi(fi_3, fi_2, fi_1,Tm,garm,Comp_Sp,Num,Ntm,ps,ys_dat,y_gen,y_gens ,ys_mark,nTstp, Nfft, sr, Step,gr_on):
    Fi_r1 = fi_1
    Fi_r2 = fi_2
    Fi_r3 = fi_3
    sp=[]
    sp1=[]
    sp2=[]
    sp3=[]
    CompSp=copy.deepcopy(Comp_Sp)
    ys=copy.deepcopy(ys_dat)
    garmonica=copy.deepcopy(garm)
    # первый раунд
    Mod_Fi = SetComplexRotationFi(garmonica, Fi_r1)
    sp1 = cange_dat_in_complex_spec(CompSp[:, Ntm + 1], Num, Mod_Fi)
    # добавляем к сигналу, пока в ручную
    ysw = np.fft.ifft(sp1)
    Ham_win = np.hanning(Nfft)
    wdat = ysw.real * Ham_win
    pos=ps
    nTstep=nTstp
    ys[pos:pos + Nfft] = ys[pos:pos + Nfft] + wdat
    pos += Step
    nTstep += 1
    ys_mark[pos] = 1.25

    # второй раунд
    Mod_Fi = SetComplexRotationFi(garmonica, Fi_r2)
    sp2 = cange_dat_in_complex_spec(CompSp[:, Ntm + 2], Num, Mod_Fi)

    # добавляем к сигналу, пока в ручную
    ysw = np.fft.ifft(sp2)
    wdat = ysw.real * Ham_win
    ys[pos:pos + Nfft] = ys[pos:pos + Nfft] + wdat
    pos += Step
    nTstep += 1
    ys_mark[pos] = 1.5
    # третий раунд
    Mod_Fi = SetComplexRotationFi(garmonica, Fi_r3)
    sp3 = cange_dat_in_complex_spec(CompSp[:, Ntm + 3], Num, Mod_Fi)

    # добавляем к сигналу, пока в ручную
    ysw = np.fft.ifft(sp3)
    wdat = ysw.real * Ham_win
    ys[pos:pos + Nfft] = ys[pos:pos + Nfft] + wdat
    pos += Step
    nTstep += 1
    ys_mark[pos] = 2
    # а теперь снова цикл
    sp = sp3
    for i in range(21):
        Oldfi = get_fi(sp[Num])
        dop_fi = get_new_fi(Oldfi, Num, Step, Nfft, sr)
        # меняем фазу гармоники
        Mod_Fi = SetComplexRotationFi(sp[Num], dop_fi)
        sp = cange_dat_in_complex_spec(sp, Num, Mod_Fi)
        # sp = cange_dat_in_complex_spec(sp, Numf, garmonica) # нет протяжки
        # синтезируем сигнал
        ysw = np.fft.ifft(sp)
        wdat = ysw.real * Ham_win
        ys[pos:pos + Nfft] = ys[pos:pos + Nfft] + wdat
        pos += Step
        nTstep += 1
        ys_mark[pos] = 1

    ## а теперь смотрим сигнал снова
    CompSpD, LogSpD, Step = GetMatrixSpectrumNoWin(ys, Nfft, Step)
    #lgsp1, mat_fi1 = get_mat_for_pik(CompSpD[:nfmax, :])

    # смотрим фазу сигнала в нужный момент
    if gr_on == 1:
        plt.figure()
        plt.plot(tv, ys)
        plt.plot(tv, ys_mark / 512)
        plt.plot(tv, y_gen / 512)
    ## а теперь попробуем демодулировать и выделить фазу по двум синусам.
    ps = 0
    fi_det = []
    fi_det1=[]
    t_det = []

    NTper = int((1 / f_gen) * sr)

    while ps < Tm:
        nps = int(sr * ps)
        x = ys[nps:nps + NTper]
        y = y_gen[nps:nps + NTper]
        y_S = y_gens[nps:nps + NTper]

        fi_det.append(get_fi_from_2_sig_acos(y,x))
        fi_det1.append(get_fi_from_2_sig_atan2(y,y_S,x))
        t_det.append(ps)
        ps += 0.01

    Out_fi_new = np.mean(fi_det[27:45])
    Out_fi_new1 = np.mean(fi_det1[27:45])
    if gr_on==1:
        print('частота = ', round(f_gen))
        print('начальная фаза=', fi0)
        print('фаза -3 = ', Fi_r1)
        print('фаза -2 = ', Fi_r2)
        print('фаза -1 = ', Fi_r3)
        print('Итоговая фаза=', round(Out_fi_new))
    return round(Out_fi_new), round(Out_fi_new1)


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
y_gens = np.sin(2*pi*f_gen*tv+fi0*pi/180)
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
#lgsp, mat_fi = get_mat_for_pik(CompSp[:nfmax,:])
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
#print('num f max=',Num,' f = ', int((Num/Nfft)*sr),' Hz')

# смотрим мгновенную фазу на нанало работы
#Oldfi = get_fi(CompSp[Num,Ntm])
#print('old fi',Oldfi,' окно номер',nTstep )

# пошли поварачивать фазу цель = за три такта на 4 повернуть как надо

#Fi_new=180 # я хочу эту новую фазу чтоы в окне номер nTstep+3 и дальше она была. а в иделе начиная с nTstep + 2
#Oldfi = get_fi(CompSp[Num,Ntm])
# зададим частоты 
GR_on = 0

temp_v1=copy.deepcopy(garmonica)
temp_v2=copy.deepcopy(CompSp)
temp_v3=copy.deepcopy(pos)
temp_v4=copy.deepcopy(ys)
temp_v5=copy.deepcopy(Num)
temp_v6=copy.deepcopy(Ntm)


#Fi_r1 = 0
#Fi_r2 = 0
#Fi_r3 = 0
#fi_f1f2f3, fi1_f1f2f3=temp_test_form_new_fi(Fi_r3, Fi_r2, Fi_r1,Tm,garmonica,CompSp,Num,Ntm,pos,temp_v4,y_gen,y_gens ,ys_mark,nTstep, Nfft, sr, Step,GR_on)
#print(fi_f1f2f3, fi1_f1f2f3)

#Fi_r1 = 0
#Fi_r2 = 0
#Fi_r3 = 90
#fi_f1f2f3, fi1_f1f2f3=temp_test_form_new_fi(Fi_r3, Fi_r2, Fi_r1,Tm,garmonica,CompSp,Num,Ntm,pos,temp_v4,y_gen,y_gens ,ys_mark,nTstep, Nfft, sr, Step,GR_on)
#print(fi_f1f2f3, fi1_f1f2f3)
#print('start')
##убедились что все работает - начинам перебор
#MainTable=[]
#for f_1 in range(30,41,90):
#    print(f_1)
#    for f_2 in range(45, 46, 90):
#        print(f_2)
#        for f_3 in range(0, 361, 2):
#            Fi_r1 = f_1
#            Fi_r2 = f_2
#            Fi_r3 = f_3
#            fi_f1f2f3, fi1_f1f2f3 = temp_test_form_new_fi(Fi_r3, Fi_r2, Fi_r1, Tm, garmonica, CompSp, Num, Ntm, pos,
#                                                          temp_v4, y_gen, y_gens, ys_mark, nTstep, Nfft, sr, Step,
#                                                          GR_on)
#            res=[Fi_r1, Fi_r2, Fi_r3, fi1_f1f2f3]
#            MainTable.append(res)
#
#print('done')

# раз так все хорошо то можно просто подбирать фазу гармоники, которая нужна
Fi_target=145
er=1000
Fi_r1 = 0
Fi_r2 = 0
df2=30
while (er>2)and(Fi_r2<359):
    Fi_r2 += df2
    Fi_r3 = 0
    df=1
    while (er>2)and(Fi_r3<359):
        Fi_r3 +=df
        fi_f1f2f3, fi1_f1f2f3 = temp_test_form_new_fi(Fi_r3, Fi_r2, Fi_r1, Tm, garmonica, CompSp, Num, Ntm, pos,
                                                      temp_v4, y_gen, y_gens, ys_mark, nTstep, Nfft, sr, Step,
                                                      GR_on)
        er=abs(fi1_f1f2f3-Fi_target)

print(Fi_r1, Fi_r2, Fi_r3)
GR_on=1
fi_f1f2f3, fi1_f1f2f3 = temp_test_form_new_fi(Fi_r3, Fi_r2, Fi_r1, Tm, garmonica, temp_v2, Num, Ntm, pos,
                                                  temp_v4, y_gen, y_gens, ys_mark, nTstep, Nfft, sr, Step,
                                                  GR_on)


#MainTable=np.array(MainTable)
#np.save('temp_dat2', MainTable)
#plt.plot(MainTable[:,3]); plt.show()
#print(temp_v1==garmonica)
#print(temp_v2==CompSp)
#print(temp_v3==pos)
#print(temp_v4 is ys)
#print(temp_v5==Num)
#print(temp_v6==Ntm)

#Fi_r1 = 0
#Fi_r2 = 0
#Fi_r3 = 90
#GR_on = 0
#fi_f1f2f3, fi1_f1f2f3=temp_test_form_new_fi(Fi_r3, Fi_r2, Fi_r1,Tm,garmonica,CompSp,Num,Ntm,pos,ys,y_gen,y_gens ,ys_mark,nTstep, Nfft, sr, Step,GR_on)
#print(fi_f1f2f3, fi1_f1f2f3)
#plt.figure()
#plt.plot(t_det,fi_det)


#plt.figure; plt.plot(20*np.log10(abs(sp)))
#plt.matshow(mat_fi)
#plt.matshow(mat_fi1)
#plt.figure()
#plt.plot(mat_fi1[Num,:])
plt.show()
