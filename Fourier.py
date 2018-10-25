#Fourier.py

import numpy as np
import matplotlib.pyplot as plt

#Almacenar datos

signal = np.loadtxt('signal.dat', delimiter = ',')

incompletos = np.loadtxt('incompletos.dat', delimiter = ',')

#Grafica de signal.dat


c1 = signal[:,0]
c2 = signal[:,1]

plt.title('Signal')
plt.grid()
plt.plot(c1,c2, color = 'crimson')
plt.savefig('RojasLaura_signal.pdf')

#Transformada discreta de Fourier a los datos de signal

#def transformada ():
