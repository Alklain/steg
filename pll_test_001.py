import matplotlib.pyplot as plt
import numpy as np
import math
import cmath
import copy
eps = np.finfo(float).eps
pi = math.pi
# pll_time2.m 6/3/16 nr
# Digital PLL model in time using difference equations.
fn = 5 #kHz NCO initital freq error = -100ppm*8 MHz = -800 Hz
N= 30000 # number of samples
fref = 8e6#  Hz freq of ref signal
fs= 25e6#  Hz sample rate
Ts = 1/fs#  s sample time
nv = np.linspace(1, N, N) #% time index

t= nv*Ts*1000                #% ms
init_phase = 0.7 #  cycles initial phase of reference signal
ref_phase = fref*nv*Ts + init_phase # cycles phase of reference signal
ref_phase = ref_phase % 1 #cycles phase mod 1
Knco= 1/4096 # NCO gain constant
KI= .0032 #loop filter integrator gain
KL= 5.1 # loop filter linear (proportional) gain
fnco = fref*(1-100e-6) #;
#      % Hz NCO initial frequency
u=[]
u.append(0)
int = []
int.append(0)
phase_error=[]
phase_error.append(-init_phase)
vtune=[]
vtune.append(-init_phase*KL)
#% compute difference equations
y=[]
y.append(0)
for n in range(1,N):
  # NCO
  x = fnco*Ts + u[n-1] + vtune[n-1]*Knco # cycles NCO phase
  u.append(x % 1) #  cycles NCO phase mod 1
  s = np.sin(2*pi*u[n-1]) # NCO sine output
  y.append(round(2**15*s)/2**15) # quantized sine output
  # Phase Detector
  pe= ref_phase[n-1] - u[n-1] # phase error
  pe= 2*(((pe+1/2) % 1) - 1/2) #  wrap if phase crosses +/- 1/2 cycle
  phase_error.append(pe)
  #% Loop Filter
  int.append(KI*pe + int[n-1]) # integrator
  vtune.append(int[n] + KL*pe) # loop filter output

plt.plot(t,phase_error)

plt.figure()
plt.plot(t,vtune)
plt.figure()
plt.plot(t,y)

plt.show()
