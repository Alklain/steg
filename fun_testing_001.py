import numpy as np
def mtf(a,b):
    a=5
    b=6
    print(a + b)


    return a+b


a=50
b=40
print(a+b)

c= mtf(a,b)

print(a+b)
Mt=[]
for i in range(0,1,30):
    print(i)
    r=[i, 5, i/4]
    Mt.append(r)

A=np.array(Mt)
print(A)