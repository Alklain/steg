# PLL in an SDR

# Import the necessary packages and modules
import matplotlib.pyplot as plt
import numpy as np
import math
import cmath
import copy
eps = np.finfo(float).eps
pi = math.pi

sr=48000
Nfft = 2048
tv = np.linspace(0, Nfft/sr, Nfft)


Nf=55 # частоту менять от 15 до 250 - меньше плохо, выше тоже не хорошо
fs1 = (Nf+0.5)*(sr/Nfft)
y1 = np.cos(2 * math.pi * tv * fs1+pi/2)

p1 = round (( Nfft-512)/2)
p2 = round (( Nfft+512)/2)

y1[p1:p2] = -y1[p1:p2]

# это работает ! не трогать
#K_p = 0.02667
#K_i = 0.00178
#K_0 = 1

# это работает на высоких и средних частотах ! не трогать
K_p = 0.05667
K_i = 0.001308
K_0 = 1

input_signal = y1

integrator_out = 0
phase_estimate = np.zeros(len(y1))
e_D = [] #phase-error output
e_F = [] #loop filter output
sin_out = np.zeros(len(y1))
cos_out = np.ones(len(y1))
freq_out = []
freq_out.append(0)
freq_out1 = []
freq_out1.append(0)
diff_frq = []
diff_frq1 = []
mark=[]
intfi = 0
for n in range(len(y1)-1):
    # phase detector
    try:
        e_D.append(input_signal[n] * sin_out[n])
    except IndexError:
        e_D.append(0)
    #loop filter
    integrator_out += K_i * e_D[n]
    e_F.append(K_p * e_D[n] + integrator_out)
   #NCO
    try:
        phase_estimate[n+1] = phase_estimate[n] + K_0 * e_F[n]
    except IndexError:
        phase_estimate[n+1] = K_0 * e_F[n]
    sin_out[n+1] = -np.sin(2*np.pi*(Nf/Nfft)*(n+1) + phase_estimate[n])
    cos_out[n+1] = np.cos(2*np.pi*(Nf/Nfft)*(n+1) + phase_estimate[n])
    #intfi = intfi + 2*np.pi*(Nf/Nfft)*(n+1) + phase_estimate[n]
    #mark.append(intfi)    

    #if (intfi)> 2*np.pi:
    #    intfi = copy.deepcopy(intfi - 2*np.pi)
    #else:
    #    mark.append(0)

    
    freq_out.append(phase_estimate[n])
    freq_out1.append(phase_estimate[n]-freq_out[n-1])
    #diff_frq.append(freq_out[n+1]-freq_out[n])
    #diff_frq1.append(freq_out1[n+1]-freq_out1[n])



# Create a Figure
fig = plt.figure()

# Set up Axes
ax1 = fig.add_subplot(211)
ax1.plot(cos_out, label='PLL Output')
plt.grid()
ax1.plot(input_signal, label='Input Signal')
plt.legend()
ax1.set_title('Waveforms')

# Show the plot
#plt.show()

ax2 = fig.add_subplot(212)
ax2.plot(e_F)
plt.grid()
ax2.set_title('Filtered Error')

plt.figure()
plt.plot(input_signal)
plt.plot(cos_out)

plt.figure()
plt.plot(input_signal-cos_out )

plt.figure()
plt.plot(freq_out)
plt.plot(freq_out1)
#plt.figure()
#plt.plot(diff_frq)
#plt.plot(diff_frq1)
#plt.figure()
#plt.plot(mark)
#plt.plot(cos_out)

plt.show()
