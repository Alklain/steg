import matplotlib.pyplot as plt
import numpy as np
import math
import cmath
import copy
eps = np.finfo(float).eps
pi = math.pi


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

def get_plot_spectrum(s,sr=0):
    sp = np.fft.fft(s)
    L=round(len(sp)/2)
    fv=[]
    spl = 20 * np.log10(abs(sp) + eps)
    spt = abs(sp[0:L])
    if sr>0:
        fv = np.linspace(0, sr, len(sp))
        plt.figure()
        plt.plot(fv[0:L],spl[0:L])
    else:
        plt.figure()
        plt.plot(spl[0:L])
   
    return spt, fv


def get_max_sрectrum_in_window(sp,sr,f1,f2,w_on=1):
    Nfft=len(sp)
    nrf1 = int((f1 / sr) * Nfft)
    nrf2 = int((f2 / sr) * Nfft)
    N=nrf2-nrf1
    if w_on==1:
      w=np.kaiser(N, 4)
      tst=list(abs(sp[nrf1:nrf2])*w)
    else:
      tst=list(abs(sp[nrf1:nrf2]))  
    MaxP = tst.index(max(tst))
    #print(abs(sp[MaxP-1:MaxP+1]))
    return nrf1+MaxP


Nsmp=10000
sr=48000
Nfft = 2048
Step = 512

tv = np.linspace(0, Nsmp/sr, Nsmp)
fs = 15.*(sr/Nfft)
fs1 = 70.3*(sr/Nfft)
y1 = np.cos(2 * math.pi * tv * fs)+np.cos(2 * math.pi * tv * fs1)
# добавили шум
s = np.random.normal(0, 0.01, len(y1))
y1=y1+s


# считаем матрицу спектрограммы
CompSp, LogSp, Step = GetMatrixSpectrumNoWin(y1, Nfft, Step )
# восстановили сигнал для проверки
ys = GetSignalFromSpectrum_hanning(CompSp, Step)

sp1, fv = get_plot_spectrum(y1[0:Nfft],sr)

# ищем максимум
f1 = 1000
f2 = 2000
NumMaxf = get_max_sрectrum_in_window(sp1,sr,f1,f2,0)
f_nmax = (NumMaxf/Nfft)*sr
print(NumMaxf,f_nmax,fs1)

yt1 = np.cos(2 * math.pi * tv * fs1)
yt2 = np.cos(2 * math.pi * tv * f_nmax)


# строим грфики
#plt.figure()
#plt.plot(abs(lsp1[NumMaxf-3:NumMaxf+4]))
#print(round(abs(lsp1[NumMaxf-1])), round(abs(lsp1[NumMaxf])),
#                                        round(abs(lsp1[NumMaxf+1])))

plt.figure()
plt.plot(abs(sp1[NumMaxf-3:NumMaxf+4]))
print(round(abs(sp1[NumMaxf-1])), round(abs(sp1[NumMaxf])),
                                        round(abs(sp1[NumMaxf+1])))


plt.figure()
plt.plot(tv,yt1)
plt.plot(tv,yt2)
         
plt.figure()
plt.plot(tv,y1/max(abs(y1)))
plt.plot(tv,ys[:len(tv)]/max(abs(ys) ))
plt.show()
