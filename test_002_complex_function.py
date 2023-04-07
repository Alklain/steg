## временный файл для тестирования функций
from sndmrk import *

a=complex(1,0)
a=rotA(a,340)

print(get_fi(a))
b=complex(1,0)
b=rotA(b,340)

print(get_fi(b))
    
c=ComplexRotation(a, b, 30)

print(get_fi(c))
