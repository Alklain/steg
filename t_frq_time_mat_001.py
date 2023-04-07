import numpy as np
import random

def GetDistanse(a,b):
    dist=0
    for i in range(len(a)):
        if a[i]==b[i]:
            dist=dist
        else:
            dist+=1
    return dist
def TestMatrTrack(tack_mat=0, num_track=0, a=0, trh=1):
    t_pass=1
    for i in range (num_track):
       if GetDistanse(tack_mat[:,i],a)<trh:
           t_pass=0
           break
    return t_pass
    
def GenTrack(Num_time, Num_frq):
    a = np.zeros(Num_time)
    for i in range(Num_time):
        a[i]= random.randint(0,Num_frq)
    return a


Num_time=10
Num_frq=6
a = np.zeros(Num_time)

tack_mat=np.zeros((Num_time,100000))
num_track=0
num_attempt=0 # число попыток
while (num_track<10)and(num_attempt<100000):
    NewTrack=GenTrack(Num_time, Num_frq)
    if TestMatrTrack(tack_mat,num_track,NewTrack)==1:
        tack_mat[:,num_track]=NewTrack
        num_track+=1
    num_attempt+=1

for i in range(num_track):
    print(tack_mat[:,i])


