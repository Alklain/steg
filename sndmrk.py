## модуль вспомогательных функций
import matplotlib.pyplot as plt
import numpy as np
import math
import cmath
import copy
from scipy import signal
eps = np.finfo(float).eps
pi = math.pi
## функция выдает список максимумов массива
def GetFerstPosMax(a,lw=1):
  max_pos=[]
  a=np.array(a)
  for i in range(lw,len(a)-lw-1):
    m1=max(a[i-lw:i])
    m2=max(a[i+1:i+lw+1])
    if (m1<a[i]) and (a[i]>m2):
      max_pos.append(i)
  return max_pos


def GetMatrixSpectrum(y, Nfft=2048, Step=512 ):
  # анализ и синтез
  #Nfft=2048 # база преобразования Фурье
  #Step=512  # шаг анализа
  num_blk=(len(y)-Nfft)/Step # число спектров, которые мы вычислим
  Lnew=math.ceil(num_blk)*Step+Nfft # коррекция длины массива
  a = np.zeros(Lnew)
  a[0:len(y)]=y
  ystart=a # отсчеты исходного сигнала 
  CeilNumBlk=math.ceil(num_blk)
  b = np.zeros((Nfft,CeilNumBlk)) # инициализация массивов
  Comp_b = np.zeros((Nfft,CeilNumBlk),dtype=complex)
  ### анализ
  pos=0
  Ham_win = np.hamming ( Nfft )
  for i in range(CeilNumBlk):
    dat=a[pos:pos+Nfft] 
    wdat=dat*Ham_win
    sp=np.fft.fft(wdat)
    b[:,i]=20*np.log10(abs(sp))
    Comp_b[:,i]=sp
    pos+=Step
  return Comp_b, b, Step


def GetMatrixSpectrumNoWin(y, Nfft=2048, Step=512 ):
  # анализ и синтез
  #Nfft=2048 # база преобразования Фурье
  #Step=512  # шаг анализа
  num_blk=(len(y)-Nfft)/Step # число спектров, которые мы вычислим
  Lnew=math.ceil(num_blk)*Step+Nfft # коррекция длины массива
  a = np.zeros(Lnew)
  a[0:len(y)]=y
  ystart=a # отсчеты исходного сигнала
  CeilNumBlk=math.ceil(num_blk)
  b = np.zeros((Nfft,CeilNumBlk)) # инициализация массивов
  Comp_b = np.zeros((Nfft,CeilNumBlk),dtype=complex)
  ### анализ
  pos=0
  eps = np.finfo(float).eps
  for i in range(CeilNumBlk):
    dat=a[pos:pos+Nfft]
    sp=np.fft.fft(dat)
    b[:,i]=20*np.log10(abs(sp)+eps)
    Comp_b[:,i]=sp
    pos+=Step
  return Comp_b, b, Step


def GetSignalFromSpectrum(Comp_b,Step,norm=1):
  Nfft=len(Comp_b)
  num_blk=len(Comp_b[0,:])
  Lnew=math.ceil(num_blk)*Step+Nfft
  ys=np.zeros(Lnew)
  pos=0
  Ham_win = np.hamming ( Nfft )
  for i in range(num_blk):
    sp=Comp_b[:,i]
    ysw=np.fft.ifft(sp)
    wdat=ysw.real*Ham_win
    ys[pos:pos+Nfft]=ys[pos:pos+Nfft]+wdat
    pos+=Step
  if norm == 1:
      return ys / abs(ys).max()
  else:
      return ys / 2

def GetSignalFromSpectrum_hanning(Comp_b,Step,norm=1):
  Nfft=len(Comp_b)
  num_blk=len(Comp_b[0,:])
  Lnew=math.ceil(num_blk)*Step+Nfft
  ys=np.zeros(Lnew)
  pos=0
  Ham_win = np.hanning ( Nfft )
  for i in range(num_blk):
    sp=Comp_b[:,i]
    ysw=np.fft.ifft(sp)
    wdat=ysw.real*Ham_win
    ys[pos:pos+Nfft]=ys[pos:pos+Nfft]+wdat
    pos+=Step
  if norm==1:
      return ys/abs(ys).max()
  else:
      return ys/2

def get_fi(CompA):
  fi=0
  if CompA.imag<0:
    fi=360+math.atan2(CompA.imag,CompA.real)*180/math.pi
  else:
    fi=math.atan2(CompA.imag,CompA.real)*180/math.pi
  return fi

def get_compl_rot(fi):
  return complex(math.cos(fi*math.pi/180),math.sin(fi*math.pi/180))

def rotA(CompA,fi):
  return get_compl_rot(fi)*CompA


def ComplexRotation(CompA, CompB, fi):
  # комплексное число B поворачиваем по фазе так, чтобы
  # новая фаза была равна фазе числа А+fi, fi - градусы
  fia=get_fi(CompA)
  fib=get_fi(CompB)
  dfi=(fia+fi-fib)%360
  cw=get_compl_rot(dfi)
  
  return CompB*cw
def ComplexRotationFi(fia, CompB, fi):
  # комплексное число B поворачиваем по фазе так, чтобы
  # новая фаза была равна фазе числа А+fi, fi - градусы
  
  fib=get_fi(CompB)
  dfi=(fia+fi-fib)%360
  cw=get_compl_rot(dfi)
  #print(dfi)
  return CompB*cw

def SetComplexRotationFi(CompB, fi):
  # комплексное число B поворачиваем по фазе так, чтобы
  #новая фаза была равна fi - градусы  
  B=complex(abs(CompB),0)
  cw=get_compl_rot(fi)  
  return B*cw


def GetLocalMaxSpectrum(Comp_sp, f1, f2,df, Nfft=2048, sr=22050 ):
  # алгоритм
  # находим максимумы спектра. после заданной частоты сдвигаем на df.
  nrf1=int((f1/sr)*Nfft)
  nrf2=int((f2/sr)*Nfft)
  nnf3=int((nrf1+nrf1)/2)
  temp_s=Comp_sp
  PosMaxSp=GetFerstPosMax(abs(temp_s[nrf1:nrf2]),df)
  for i in range(len(PosMaxSp)):
    PosMaxSp[i]+=nrf1
  sp1=np.zeros(len(temp_s),dtype=complex)
  ## сборка

  sp1=copy.deepcopy(temp_s)
  sp1[min(PosMaxSp):max(PosMaxSp)]=0
  sp1[len(sp1)-max(PosMaxSp):len(sp1)-min(PosMaxSp)]=0
  for i in range(len(PosMaxSp)):
    sp1[int(PosMaxSp[i])]=temp_s[int(PosMaxSp[i])]
    sp1[len(sp1)-int(PosMaxSp[i])]=temp_s[len(temp_s)-int(PosMaxSp[i])]
  return sp1

def ShiftSpectrumDown(Comp_sp, f1, f2,df,vl=1, Nfft=2048, sr=22050):
  
    nrf1=int((f1/sr)*Nfft)
    nrf2=int((f2/sr)*Nfft)
    nnf3=int((nrf1+nrf2)/2)
    pos=nrf1+1
    pop_on=1 # флаг  можно удалять
    fshif=0 # сколько удалили 
    while pos<nnf3:
      if (Comp_sp[pos]>0) and (pop_on==1):
        # убрали элемент
        for i in range(vl):
          Comp_sp=np.delete(Comp_sp, pos+1)
        #np.insert(Comp_sp, nnf3,0)
        fshif+=vl
        if fshif==df:
          pop_on=0
          pos=nnf3
      pos+=1
    
    pop_on=1 # флаг - можно добавлять
    fshif=0 # сколько добавили
    while pos<nrf2:
      if (Comp_sp[pos]>0) and (pop_on==1):
        # убрали элемент
        for i in range(vl):
          Comp_sp=np.insert(Comp_sp, pos+1,0)
        #np.delete(Comp_sp, pos+1)
        fshif+=vl
        if fshif==df:
          pop_on=0
          pos=nrf2
      pos+=1
    Comp_sp=np.delete(Comp_sp, nnf3)  

    # разворот
    HalfSp=Comp_sp[1:int(Nfft/2)]
    #Проверка длины 
    if len(Comp_sp)!=Nfft:
      if len(Comp_sp)<Nfft:
        while len(Comp_sp)<Nfft:
          Comp_sp=np.insert(Comp_sp, int(Nfft/2),0)
      else:
        while len(Comp_sp)>Nfft:
          Comp_sp=np.delete(Comp_sp, int(Nfft/2))
          
    Comp_sp[Nfft-len(HalfSp) :]=np.flipud(HalfSp).conjugate()

    return Comp_sp


def ShiftSpectrumUp(Comp_sp, f1, f2,df,vl=1, Nfft=2048, sr=22050):
  
    nrf1=int((f1/sr)*Nfft)
    nrf2=int((f2/sr)*Nfft)
    nnf3=int((nrf1+nrf2)/2)
    pos=nrf1+1
    pop_on=1 # флаг  можно удалять
    fshif=0 # сколько удалили 
    while pos<nnf3:
      if (Comp_sp[pos]>0) and (pop_on==1):
        # убрали элемент
        for i in range(vl):
          Comp_sp=np.insert(Comp_sp, pos+1,0)
        
        #np.insert(Comp_sp, nnf3,0)
        fshif+=vl
        if fshif==df:
          pop_on=0
          pos=nnf3
      pos+=1
    
    pop_on=1 # флаг - можно добавлять
    fshif=0 # сколько добавили
    while pos<nrf2:
      if (Comp_sp[pos]>0) and (pop_on==1):
        # убрали элемент
        for i in range(vl):
          Comp_sp=np.delete(Comp_sp, pos+1)
        #np.delete(Comp_sp, pos+1)
        fshif+=vl
        if fshif==df:
          pop_on=0
          pos=nrf2
      pos+=1
    #Comp_sp=np.delete(Comp_sp, nnf3)  

    # разворот
    HalfSp=Comp_sp[1:int(Nfft/2)]
    #Проверка длины 
    if len(Comp_sp)!=Nfft:
      if len(Comp_sp)<Nfft:
        while len(Comp_sp)<Nfft:
          Comp_sp=np.insert(Comp_sp, int(Nfft/2),0)
      else:
        while len(Comp_sp)>Nfft:
          Comp_sp=np.delete(Comp_sp, int(Nfft/2))
          
    Comp_sp[Nfft-len(HalfSp) :]=np.flipud(HalfSp).conjugate()

    return Comp_sp

   
def ShiftSpectrumFullDwon(Comp_sp, f1, f2,df,vl=1, Nfft=2048, sr=22050):
  
    nrf1=int((f1/sr)*Nfft)
    nrf2=int((f2/sr)*Nfft)
    nnf3=int((nrf1+nrf2)/2)
    pos=nrf1+1
    pop_on=1 # флаг  можно удалять
    fshif=0 # сколько удалили 
    while pos<nrf2:
      if (Comp_sp[pos]>0) and (pop_on==1):
        # убрали элемент
        for i in range(vl):          
          Comp_sp=np.delete(Comp_sp, pos+1)
          Comp_sp=np.insert(Comp_sp, nrf2,0)
        
        #np.insert(Comp_sp, nnf3,0)
        fshif+=vl
        if fshif==df:
          pop_on=0
          pos=nrf2
      pos+=1
    
    # разворот
    HalfSp=Comp_sp[1:int(Nfft/2)]
    #Проверка длины 
    if len(Comp_sp)!=Nfft:
      if len(Comp_sp)<Nfft:
        while len(Comp_sp)<Nfft:
          Comp_sp=np.insert(Comp_sp, int(Nfft/2),0)
      else:
        while len(Comp_sp)>Nfft:
          Comp_sp=np.delete(Comp_sp, int(Nfft/2))
          
    Comp_sp[Nfft-len(HalfSp) :]=np.flipud(HalfSp).conjugate()
    

    
    return Comp_sp

def ShiftSpectrumFullUp(Comp_sp, f1, f2,df,vl=1, Nfft=2048, sr=22050):
  
    nrf1=int((f1/sr)*Nfft)
    nrf2=int((f2/sr)*Nfft)
    nnf3=int((nrf1+nrf2)/2)
    pos=nrf2
    pop_on=1 # флаг  можно удалять
    fshif=0 # сколько удалили 
    while pos>nrf1:
      if (Comp_sp[pos]>0) and (pop_on==1):
        # убрали элемент
        for i in range(vl):          
          Comp_sp=np.delete(Comp_sp, pos-1)
          Comp_sp=np.insert(Comp_sp, nrf1,0)
        
        #np.insert(Comp_sp, nnf3,0)
        fshif+=vl
        if fshif==df:
          pop_on=0
          pos=nrf1
      pos-=1
    
    # разворот
    HalfSp=Comp_sp[1:int(Nfft/2)]
    #Проверка длины 
    if len(Comp_sp)!=Nfft:
      if len(Comp_sp)<Nfft:
        while len(Comp_sp)<Nfft:
          Comp_sp=np.insert(Comp_sp, int(Nfft/2),0)
      else:
        while len(Comp_sp)>Nfft:
          Comp_sp=np.delete(Comp_sp, int(Nfft/2))
          
    Comp_sp[Nfft-len(HalfSp) :]=np.flipud(HalfSp)    

    return Comp_sp

   
def MixSpectrum(Comp_sp1, Comp_sp2,Lev=0):
  sp1=np.zeros(len(Comp_sp1),dtype=complex)
  if Lev==0:
    for i in range(len(Comp_sp1)):
      if abs(Comp_sp1[i])>abs(Comp_sp2[i]):
        sp1[i]=Comp_sp1[i]
      else:
        sp1[i]=Comp_sp2[i]
  else:
    for i in range(len(Comp_sp1)):
      if (abs(Comp_sp1[i])>abs(Comp_sp2[i]))and(abs(Comp_sp2[i])>Lev):
        sp1[i]=Comp_sp1[i]
      else:
        sp1[i]=Comp_sp2[i]    
  return sp1

def PulseDetect(Comp_sp1, fv, Nfft=2048, sr=22050):
    # переведем частоты в номера отчетов спектра
    nfv=[]
    for i in range(len(fv)):
        nfv.append(int((fv[i] / sr) * Nfft) )
    en=[]
    for i in range(len(nfv)):
        #e1=abs(Comp_sp1[nfv[i]])+abs(Comp_sp1[nfv[i]-1])+abs(Comp_sp1[nfv[i]+1])
        e1 = abs(Comp_sp1[nfv[i]]) - (abs(Comp_sp1[nfv[i] - 1]) + abs(Comp_sp1[nfv[i] + 1]))/2
        en.append(e1)
    return en.index(max(en))

def get_lfm_sig(f1, f2, tm, sr,wind):
    dF = f2-f1;
    tv = np.linspace(0, tm, int(tm*sr))
    y = np.sin(2 * math.pi * (tv * f1 + 2*(tv**2)*dF))
    if wind!=0:
        #Ham_win = np.hamming(wind)
        Ham_win = np.blackman(wind)
        if len(y)>wind:
            nm=len(y)-wind
            new_w=np.zeros_like(y)
            new_w[0:int(wind/2)]=Ham_win[0:int(wind/2)]
            new_w[len(new_w)-int(wind / 2):] = Ham_win[int(wind / 2):]
            new_w[int(wind/2):len(new_w)-int(wind / 2)]=new_w[int(wind/2):len(new_w)-int(wind / 2)]+1
            y=y*new_w
    return y


def get_sin_sig(f1, fi, tm, sr,wind):
    tv = np.linspace(0, tm, int(tm*sr))
    y = np.sin(2 * math.pi * tv * f1 + (fi*math.pi/180))
    if wind!=0:
        #Ham_win = np.hamming(wind)
        Ham_win = np.blackman(wind)
        if len(y)>wind:
            nm=len(y)-wind
            new_w=np.zeros_like(y)
            new_w[0:int(wind/2)]=Ham_win[0:int(wind/2)]
            new_w[len(new_w)-int(wind / 2):] = Ham_win[int(wind / 2):]
            new_w[int(wind/2):len(new_w)-int(wind / 2)]=new_w[int(wind/2):len(new_w)-int(wind / 2)]+1
            y=y*new_w
    return y

def get_max_soectrum_in_window(sp,sr,f1,f2):
    Nfft=len(sp)
    nrf1 = int((f1 / sr) * Nfft)
    nrf2 = int((f2 / sr) * Nfft)
    N=nrf2-nrf1
    w=np.kaiser(N, 4)
    tst=list(abs(sp[nrf1:nrf2])*w)
    MaxP = tst.index(max(tst))
    #print(abs(sp[MaxP-1:MaxP+1]))
    return nrf1+MaxP

def change_phase_in_spectrum(sp,nf,df,fi):
    sp1 = copy.deepcopy(sp)
    for i in range(2 * df+1):
        pos=nf-df+i
        cfreq=sp[pos]
        sp1[pos]=SetComplexRotationFi(cfreq, fi)
        sp1[len(sp1)-pos]=sp1[pos].conjugate()
    return  sp1


def cange_dat_in_complex_spec(sp,num,new_dat):
    sp1 = copy.deepcopy(sp)
    sp1[num]=new_dat
    sp1[len(sp1)-num]=new_dat.conjugate()
    return sp1


def get_new_fi(Oldfi,Nmax,Step,Nfft,sr):
    new_fi=Oldfi+(math.pi*2*(Step/((1/((Nmax/Nfft)*sr))*sr)-math.floor(Step/((1/((Nmax/Nfft)*sr))*sr))))*(180/math.pi)
    return new_fi

def change_phase_in_2_wind(sp,sp_new,Num,Nfft,NewFi,Step,sr):
    Mod_Fi = SetComplexRotationFi(sp[Num], NewFi)
    spf = cange_dat_in_complex_spec(sp, Num, Mod_Fi)
    Oldfi = get_fi(spf[Num])
    dop_fi = get_new_fi(Oldfi, Num, Step, Nfft, sr)
    # меняем фазу гармоники
    Mod_Fi = SetComplexRotationFi(sp_new[Num], dop_fi)
    spnewf = cange_dat_in_complex_spec(sp_new, Num, Mod_Fi)
    return spf, spnewf

def get_mat_for_pik(sp_mat):
    eps=np.finfo(float).eps
    NumSp = len(sp_mat[1, :])
    Nfft = len(sp_mat[:, 1])
    logsp=np.zeros((Nfft,NumSp))
    fisp = np.zeros((Nfft,NumSp))
    for TimeSlot in range(NumSp):
        for Freq in range( Nfft):
            logsp[Freq,TimeSlot]=20*math.log10(abs(sp_mat[Freq,TimeSlot])+eps)
            fisp[Freq,TimeSlot] = get_fi(sp_mat[Freq,TimeSlot])
    return logsp, fisp


def change_phase_in_matrix(sp_mat, Nf1, Nf2, Nt1, Nt2, NewFi,Step,sr):
    NewSpMat = copy.deepcopy(sp_mat)
    NumSp=len(sp_mat[1, :])
    Nfft=len(sp_mat[:, 1])
    for TimeSlot in range(Nt1, Nt2):
        for Freq in range( Nf1, Nf2):
            if TimeSlot>Nt1:
                Oldfi = get_fi(sp_mat[Freq,TimeSlot-1])
                NewFi1=get_new_fi(Oldfi, Freq, Step, Nfft, sr)
            else:
                NewFi1=NewFi
            Mod_Fi = SetComplexRotationFi(sp_mat[Freq,TimeSlot], NewFi)
            NewSpMat[Freq,TimeSlot] = Mod_Fi
            NewSpMat[Nfft-Freq,TimeSlot] = Mod_Fi.conjugate()
    return NewSpMat

def my_test_ifft(sp):
    pi = math.pi
    tv = np.linspace(0, 2 * pi - (2 * pi / len(sp)), len(sp))
    y = np.zeros_like(tv)
    for f in range(round(len(sp) / 2)):
        yf = abs(sp[f]) * np.cos(tv * f + math.atan2(sp[f].imag, sp[f].real))
        y = y + yf
    y = y / round(len(sp) / 2)
    return y
# вычисляем нормированную корреляцию массивов
def get_my_corr(x,y):
    eps = np.finfo(float).eps
    x1 = x / (max(abs(x)) + eps)
    y1 = y / (max(abs(y)) + eps)
    K=2*np.mean(x1*y1)/np.sqrt(np.mean(abs(x1)**2)+np.mean(abs(y1)**2))
    if K>1:
        K=1
    if K<-1:
        K=-1
    return K
# вычисляем фазу между входным сигналом и двумя опорными, в квадратурах ( 0 - 360)
def get_fi_from_2_sig_atan2(Xcos,Xsin,y):
    pi = math.pi
    eps = np.finfo(float).eps
    Xc = Xcos / (max(abs(Xcos)) + eps)
    Xs = Xsin / (max(abs(Xsin)) + eps)
    y1 = y / (max(abs(y)) + eps)
    Kc = 2 * np.mean(Xc * y1) / np.sqrt(np.mean(abs(Xc) ** 2) + np.mean(abs(y1) ** 2))
    Ks = 2 * np.mean(Xs * y1) / np.sqrt(np.mean(abs(Xs) ** 2) + np.mean(abs(y1) ** 2))
    if Kc>1:
        Kc=1
    if Kc<-1:
        Kc=-1
    if Ks>1:
        Ks=1
    if Ks<-1:
        Ks=-1
    atan2_fi = (180 / pi) * math.atan2(Ks, Kc)
    if Ks < 0:
        atan2_fi = 360 + atan2_fi
    return atan2_fi
# вычисляем фазу между входным сигналом и опорным, 0 до 180
def get_fi_from_2_sig_acos(Xcos,y):
    pi = math.pi
    eps = np.finfo(float).eps
    Xc = Xcos / (max(abs(Xcos)) + eps)
    y1 = y / (max(abs(y)) + eps)
    Kc = 2 * np.mean(Xc * y1) / np.sqrt(np.mean(abs(Xc) ** 2) + np.mean(abs(y1) ** 2))
    if Kc>1:
        Kc=1
    if Kc<-1:
        Kc=-1
    acos_fi = (180 / pi) * math.acos(Kc)
    return acos_fi




def insert_new_fi_into_sp_matr(fi_3, fi_2, fi_1, Comp_Sp,Numfs, sr, Step):
    Fi_r1 = fi_1
    Fi_r2 = fi_2
    Fi_r3 = fi_3
    sp=[]
    sp1=[]
    sp2=[]
    sp3=[]
    CompSp=copy.deepcopy(Comp_Sp)
    NumSp = len(CompSp[1, :])
    Nfft = len(CompSp[:, 1])
    Ntm = math.floor(NumSp/2)
    #print(NumSp, Ntm)


    # первый раунд
    Mod_Fi = SetComplexRotationFi(CompSp[Numfs, Ntm - 3], Fi_r1)
    sp1 = cange_dat_in_complex_spec(CompSp[:, Ntm - 3], Numfs, Mod_Fi)
    CompSp[:, Ntm - 3]= sp1

    # второй раунд
    Mod_Fi = SetComplexRotationFi(CompSp[Numfs, Ntm - 2], Fi_r2)
    sp2 = cange_dat_in_complex_spec(CompSp[:, Ntm - 2], Numfs, Mod_Fi)
    CompSp[:, Ntm - 2] = sp2

    # третий раунд
    Mod_Fi = SetComplexRotationFi(CompSp[Numfs, Ntm - 1], Fi_r3)
    sp3 = cange_dat_in_complex_spec(CompSp[:, Ntm - 1], Numfs, Mod_Fi)
    CompSp[:, Ntm - 1] = sp3
    ## цикл


    for i in range(3):
        Oldfi = get_fi(CompSp[Numfs, Ntm - 1+i])
        dop_fi = get_new_fi(Oldfi, Numfs, Step, Nfft, sr)
        # меняем фазу гармоники
        Mod_Fi = SetComplexRotationFi(CompSp[Numfs, Ntm +i], dop_fi)
        sp = cange_dat_in_complex_spec(CompSp[:, Ntm +i ], Numfs, Mod_Fi)
        CompSp[:, Ntm + i] = sp

    # проверка - синтезируем сигнал

    ys2 = GetSignalFromSpectrum_hanning(CompSp, Step)

    # анаоизирем сигнал

    CompSpD, LogSpD, StepD = GetMatrixSpectrumNoWin(ys2, Nfft, Step)

    # смотрим фазу
    NewSfi = get_fi(CompSpD[Numfs, Ntm])
    #print(NewSfi)

    return CompSpD, NewSfi


def insert_phase_to_spectr_mat(CompSp,Num,sr,Step,NewFi,print_on=1):
    # меняем фазу в матрице спектра
    Fi_r2 = 0
    Fi_r3 = 0
    out_fi = []
    if print_on == 1:
        print('start')
    er = 1000
    for fi1 in range(1, 360, 10):
        if print_on==1:
            print('fi1=', fi1)
        for fi2 in range(1, 360, 10):
            if print_on==1:
                print('fi2=', fi2)
            for fi3 in range(1, 360, 10):
                Fi_r1 = fi1
                Fi_r2 = fi2
                Fi_r3 = fi3
                TNewSpMat, NewSfi = insert_new_fi_into_sp_matr(Fi_r3, Fi_r2, Fi_r1, CompSp, Num, sr,Step)
                er = abs(NewFi - NewSfi)
                # print(er)
                out_fi.append([Fi_r1, Fi_r2, Fi_r3, NewSfi])
                if er < 2:
                    if print_on==1:
                        print(Fi_r1, Fi_r2, Fi_r3, NewSfi)
                    break
            if er < 2:
                break
        if er < 2:
            break

    rezmat = np.array(out_fi)
    # plt.figure()
    # plt.plot(rezmat[:,1])

    return TNewSpMat


def get_phase_sep(sp_mat):
    eps=np.finfo(float).eps
    NumSp = len(sp_mat[1, :])
    Nfft = len(sp_mat[:, 1])
    phas_sp=np.zeros((Nfft,NumSp))
    fisp = np.zeros((Nfft,NumSp))

    for TimeSlot in range(NumSp):
        for Freq in range( Nfft):
            phas_sp[Freq,TimeSlot]=np.cos(math.atan2(sp_mat[Freq,TimeSlot].imag,sp_mat[Freq,TimeSlot].real))
    return phas_sp

def set_const_phase_on_nf(sp_mat,numf,Step,sr,tp_f=0):
    eps = np.finfo(float).eps
    NumSp = len(sp_mat[1, :])
    Nfft = len(sp_mat[:, 1])
    CompSp = copy.deepcopy(sp_mat)
    for TimeSlot in range(NumSp):
        sp=CompSp[:,TimeSlot]
        if TimeSlot>0:
            Oldfi = get_fi(sp[numf])
            dop_fi = get_new_fi(Oldfi, numf, Step, Nfft, sr)
            # меняем фазу гармоники
            if tp_f==0:
                Mod_Fi = SetComplexRotationFi(sp[numf], dop_fi)
            else:
                if tp_f<0:
                    Mod_Fi = SetComplexRotationFi(sp[numf], -dop_fi)
                else:
                    Mod_Fi = SetComplexRotationFi(sp[numf], tp_f)
            sp1 = cange_dat_in_complex_spec(sp, numf, Mod_Fi)
            CompSp[:,TimeSlot] = sp1
    return CompSp

def get_plot_spectrum(s,sr=0):
    sp = np.fft.fft(s)
    L=round(len(sp)/2)
    fv=[]
    spl = 20 * np.log10(abs(sp) + eps)
    if sr>0:
        fv = np.linspace(0, sr, len(sp))
        plt.figure()
        plt.plot(fv[0:L],spl[0:L])
    else:
        plt.figure()
        plt.plot(spl[0:L])
    return sp, fv

def get_part_of_spec_mat(sp_mat,nfmin,nfmax,ntmin,ntmax):
    NumSp = len(sp_mat[1, :])
    Nfft = len(sp_mat[:, 1])
    if nfmin>nfmax:
        c=nfmax
        nfmax=nfmin
        nfmin=c
    if ntmin>ntmax:
        c=ntmax
        ntmax=ntmin
        ntmin=c
    if nfmin<0:
        nfmin = 0
    if ntmin<0:
        ntmin = 0
    if nfmax>Nfft:
        nfmax=Nfft
    if ntmax>NumSp:
        ntmax=NumSp
    if (ntmin<NumSp) and (nfmin<Nfft):
        Comp_b = np.zeros((Nfft, NumSp), dtype=complex)
        Comp_b[nfmin:nfmax, ntmin: ntmax] = sp_mat[nfmin:nfmax, ntmin: ntmax]
        Comp_b[Nfft - nfmax:Nfft - nfmin, ntmin: ntmax] = sp_mat[Nfft - nfmax:Nfft - nfmin, ntmin: ntmax]
    else:
        Comp_b = np.zeros((Nfft, NumSp), dtype=complex)
    return Comp_b


def select_from_matr_sp(sp_mat,nf,nt,df,dt):
    NumSp = len(sp_mat[1, :])
    Nfft = len(sp_mat[:, 1])
    #out = np.zeros((Nfft, 2*dt+1), dtype=complex)
    out = copy.deepcopy(sp_mat[:,nt-dt:nt+dt+1])

    out[0:nf-df,:]=out[0:nf-df,:]*0
    out[nf+df+1:Nfft-(nf+df+1),:]=out[nf+df+1:Nfft-(nf+df+1),:]*0
    out[Nfft-(nf-df):,:]=out[Nfft-(nf-df):,:]*0

    #out[nf-df:nf+df+1,:] = sp_mat[nf-df:nf+df+1,nt-dt:nt+dt+1]
    #out[Nfft-nf - df-1:Nfft-nf+ df,:] = sp_mat[nf - df:nf + df + 1, nt - dt:nt + dt + 1]
    return out

def InsertBitToSpekMat(Nt,SpMat,sr,Step,f1,f2,grf_on=1):
    Ntm = Nt
    CompSp1=copy.deepcopy(SpMat)
    NumSp = len(CompSp1[1, :])
    Nfft = len(CompSp1[:, 1])

    # смотрим мгновенныей спектр
    sp = copy.deepcopy(SpMat[:, Ntm])
    #tls = list(abs(sp[nfmin:nfmax]))
    #fNum = nfmin + tls.index(max(tls))

    fNum=get_max_soectrum_in_window(sp, sr, f1, f2)

    #print('num f max=',fNum,' f = ', int((fNum/Nfft)*sr),' Hz')

    df = 2
    dt = 5
    sp_frame = select_from_matr_sp(CompSp1, fNum, Ntm, df, dt)
    ysf = GetSignalFromSpectrum_hanning(sp_frame, Step,0)
    phaz, intphaz = get_pll_estimate(fNum, sr, Nfft,  ysf[2*Step:len(ysf)-2*Step], 0)
        

    Fest = np.mean(np.array(intphaz[Step:Step+1000]))    
    Fest = (Nfft/(2*pi))*Fest
    fs1 = get_freq_from_num(Fest, Nfft, sr)

    print('точная частота сигнала', fs1)

    # инвертируем сигнал
    ytmp = 1*copy.deepcopy(ysf)

    ogib = get_ogib_signal(ytmp)
    
    p1 = int((len(ysf) - 1*Step) / 2)
    # ишем переход нуля
    ps=p1
    flag=1
    while flag==1:
      ps+=1
      if ytmp[ps]*ytmp[ps-1]<0:
        flag=0
      if (ps-p1)>Nfft/Ntm:
        flag=0
    p1=ps
    p2 = int((len(ysf) + 1*Step) / 2)
    ps=p2
    flag=1
    while flag==1:
      ps+=1
      if ytmp[ps]*ytmp[ps-1]<0:
        flag=0
      if (ps-p1)>Nfft/Ntm:
        flag=0
    p2=ps
    ytmp[p1:p2] = -1*ytmp[p1:p2]

    Nt1 = len(ytmp)
    tv1 = np.linspace(0, Nt1 / sr, Nt1)
    fs = get_freq_from_num(fNum, Nfft, sr)
    print('частота сигнала по спектру', fs)

    Oldfi = get_fi(CompSp1[fNum, Ntm-3])
    dop_fi = get_new_fi(Oldfi, fNum, Step, Nfft, sr)

    fi = get_fi(CompSp1[fNum, Ntm-3])
    #dop_fi = get_new_fi(0, fNum, Step, Nfft, sr)
    
    yop = np.cos(2 * math.pi * tv1 * fs + ((fi - dop_fi)* pi / 180))
    yop_f1 = yop
    yop_f1[p1:p2] = -yop[p1:p2]

    #yopmd[p1:p2] = -yopmd[p1:p2] # модификация синтезированного ( гармонического) сигнала
    #yopmd = yop_f1*max(ysf)*0.75
    yopmd = yop_f1*ogib
    

    #yopmd = ytmp # модификация реального сигнала
    # собираем сигнал в спектр

    #FragSp, FLogSp, Step = GetMatrixSpectrumNoWin(ysf, Nfft, Step)
    FragSp, FLogSp, Step = GetMatrixSpectrumNoWin(yopmd, Nfft, Step)

    CompSpD = copy.deepcopy(CompSp1)
    fp1=fNum - df
    fp2=fNum + df+1
    tp1=Ntm - (dt)
    tp2=Ntm + (dt + 1)

    #CompSpD[:,tp1-2:tp2+2]=FragSp[:, :]
    CompSpD[fp1:fp2, tp1:tp2] = FragSp[fp1:fp2,:]
    CompSpD[Nfft-fp2:Nfft - fp1, tp1:tp2] = FragSp[Nfft - fp2:Nfft - fp1,:]
    #ysmod = GetSignalFromSpectrum_hanning(CompSpD, Step,0)



    if grf_on==1:
        plt.figure()
        plt.plot(tv1,  yopmd)
        plt.plot(tv1, yop * max(yopmd),'k')

    #ModSp, MLogSp, Step = GetMatrixSpectrumNoWin(ysmod, Nfft, Step)
    #ModSp, MLogSp, Step = GetMatrixSpectrumNoWin(yopmd, Nfft, Step)
    return CompSpD
def get_freq_from_num(fNum, Nfft, sr):
    f=(fNum/Nfft)*sr
    return f
def DetectBit(CompSpR,Ntpm,sr,Step,f1, f2,gron=0):
    sp1 = copy.deepcopy(CompSpR[:, Ntpm])
    Nfft = len(CompSpR[:, 1])
    fNum = get_max_soectrum_in_window(sp1, sr, f1, f2)
    df = 2
    dt = 5
    sp_frame = select_from_matr_sp(CompSpR, fNum, Ntpm, df, dt)
    ysf = GetSignalFromSpectrum_hanning(sp_frame, Step, 0)
    Nt = len(ysf)
    tv1 = np.linspace(0, Nt / sr, Nt)
    fs = get_freq_from_num(fNum, Nfft, sr)
    fi = get_fi(sp_frame[fNum, 0])

    phaz, intphaz = get_pll_estimate(fNum, sr, Nfft,  ysf[2*Step:len(ysf)-2*Step], gron)

    Fest = np.mean(np.array(intphaz[Step:Step+1000]))
    print(Fest)
    Fest = (Nfft/(2*pi))*Fest
    if Fest>0:
      fs1 = get_freq_from_num(Fest, Nfft, sr)
    else:
      fs1 = fs
    

    
    print('Num F max = ', fNum, ' N max from phi = ', Fest )

    yop = np.cos(2 * math.pi * tv1 * fs + fi * pi / 180)
    ps = 0
    nTf = round((1 / fs) * sr)
    lis_fi = []
    lis_t = []
    if gron==1:
        plt.figure()
        plt.plot(tv1,ysf)
        plt.plot(tv1,yop*max(ysf),'r')
    while ps + 2 * nTf < Nt:
        yt1 = ysf[ps:ps + nTf]
        yt2 = yop[ps:ps + nTf]
        lis_fi.append(get_fi_from_2_sig_acos(yt1, yt2))
        lis_t.append(ps / sr)
        ps += nTf
    return lis_t, lis_fi, phaz


def get_pll_estimate(Nf,sr,Nfft,y1, gron=0):
    fs1 = ((Nf+2)  * sr / Nfft)
       # это работает ! не трогать
    # K_p = 0.02667
    # K_i = 0.00178
    # K_0 = 1

    # это работает на высоких и средних частотах ! не трогать
    K_p = 0.05667
    K_i = 0.001308
    K_0 = 1

    input_signal = copy.deepcopy(y1)
    integrator_out = 0
    phase_estimate = np.zeros(len(y1))
    e_D = []  # phase-error output
    e_F = []  # loop filter output
    sin_out = np.zeros(len(y1))
    cos_out = np.ones(len(y1))
    freq_out = []
    freq_out.append(0)
    freq_out1 = []
    freq_out1.append(0)

    for n in range(len(y1) - 1):
        # phase detector
        try:
            e_D.append(input_signal[n] * sin_out[n])
        except IndexError:
            e_D.append(0)
        # loop filter
        integrator_out += K_i * e_D[n]
        e_F.append(K_p * e_D[n] + integrator_out)
        # NCO
        try:
            phase_estimate[n + 1] = phase_estimate[n] + K_0 * e_F[n]
        except IndexError:
            phase_estimate[n + 1] = K_0 * e_F[n]
        sin_out[n + 1] = -np.sin(2 * np.pi * (Nf / Nfft) * (n + 1) + phase_estimate[n])
        cos_out[n + 1] = np.cos(2 * np.pi * (Nf / Nfft) * (n + 1) + phase_estimate[n])
        freq_out.append(2 * np.pi * (Nf / Nfft) * (n + 1) + phase_estimate[n])
        #freq_out1.append(freq_out[n - 1])
        # diff_frq.append(freq_out[n+1]-freq_out[n])
        # diff_frq1.append(freq_out1[n+1]-freq_out1[n])
    freq_out1=[]
    for i in range(len(freq_out)-1):
      freq_out1.append(freq_out[i+1]-freq_out[i])
    if gron==1:
      plt.figure()
      #plt.plot(input_signal)
      #plt.plot(cos_out)      
      plt.plot(freq_out1)
      print('ttt')
    return phase_estimate, freq_out1

def get_ogib_signal(y):
  #b, a = signal.butter(2, 0.01, 'lowpass')   # Конфигурационный фильтр 8 указывает порядок фильтра
  #filtedData = signal.filtfilt(b, a, abs(y))
  out = np.zeros(len(y))
  ps=0
  st=400
  while ps<len(y)-2*st:
      out[ps:ps+st]=max(y[ps:ps+st])
      ps=ps+st
  out[ps:] = max(y[ps:])
  return out
