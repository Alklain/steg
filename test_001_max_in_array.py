import numpy as np

print('test')

def GetFerstPosMax(a,lw=1):
  max_pos=[]
  a=np.array(a)
  for i in range(lw,len(a)-lw-1):
    print(i)
    m1=max(a[i-lw:i])
    m2=max(a[i+1:i+lw+1])
    print('next')
    if (m1<a[i]) and (a[i]>m2):
      max_pos.append(i)
  return max_pos

a=[1, 2, 1, 5, 3, 9 ,1,7 , 5 ,3, 1]
print(GetFerstPosMax(a,2))
