import numpy as np
import random
import matplotlib.pylab as plt

def GetDistanse(a, b):
    dist = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            dist = dist
        else:
            dist += 1
    return dist


def TestMatrTrack(tack_mat, num_track, a, trh=1):
    t_pass = 1
    for i in range(num_track):
        if GetDistanse(tack_mat[:, i], a) < trh:
            t_pass = 0
            break
    return t_pass

def GetMinDist(tack_mat, num_track):
    dist=10000
    for i in range(num_track):
        a=tack_mat[:,i]
        tempmat=np.delete(tack_mat,i,axis = 1)
        for j in range(num_track-1):
            d=GetDistanse(tempmat[:, i], a)
            if d<dist:
                dist=d
    return dist




def GenTrack(Num_time, Num_frq):
    a = np.zeros(Num_time)
    for i in range(Num_time):
        a[i] = int(random.randint(0, Num_frq))
    return a


Num_time = 20
Num_frq = 8
a = np.zeros(Num_time)

tack_mat = np.zeros((Num_time, 100000))
num_track = 0
num_attempt = 0  # число попыток
MaxNumTrack=256
while (num_track < MaxNumTrack) and (num_attempt < 1000000):
    NewTrack = GenTrack(Num_time, Num_frq)
    if TestMatrTrack(tack_mat, num_track, NewTrack,15) == 1:
        tack_mat[:, num_track] = NewTrack
        num_track += 1
    num_attempt += 1
    if num_attempt%100000==0:
        print(num_attempt)
        num_track

#for i in range(num_track):
#    print(tack_mat[:, i])

print(GetMinDist(tack_mat, num_track))
print(num_attempt)
print(num_track)
np.savetxt('C:/pythonProject/matrix.txt', tack_mat[:,0:num_track], delimiter = ',')
np.savetxt('C:/pythonProject/matrixt.txt', np.transpose(tack_mat[:,0:num_track]), delimiter = ',')

FrMat=np.loadtxt(open("C:/pythonProject/matrix.txt","rb"),delimiter=",",skiprows=0)
print(FrMat)

plt.plot(FrMat[:,1])
plt.plot(FrMat[:,10])
plt.show()