#Fourier.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


#Almacenar datos

signal = np.loadtxt('signal.dat', delimiter = ',')

incompletos = np.loadtxt('incompletos.dat', delimiter = ',')

#Grafica de signal.dat

cs1 = signal[:,0]
cs2 = signal[:,1]

plt.title('Signal')
plt.grid()
plt.plot(cs1,cs2, color = 'crimson')
plt.savefig('RojasLaura_signal.pdf')

#Transformada discreta de Fourier a los datos de signal

#def transformada (cs1,cs2):




#Grafica de transformada de Fourier



#Frecuencias principales




#Filtro, transformada inversa y grafica



#Mensaje porque no se puede hacer la transformada para los datos de incompletos



#Interpolacion cuadratica y cubica de incompletos con 512 puntos - transformada de Fourier de los datos interpolados

ci1 = incompletos[:,0]
ci2 = incompletos[:,1]

def interpol_cuadratica(ci1,ci2):
      cuadratica = interp1d(ci1, ci2, kind='quadratic')
      return cuadratica

cuadratica = interpol_cuadratica(ci1,ci2)




def interpol_cubica(ci1,ci2):
    cubica = interp1d(ci1, ci2, kind='cubic')
    return cubica

cubica = interpol_cubica(ci1,ci2)


#Grafica de las 3 transformadas (2 de signal y  de incompletos)


#Diferencias encontradas entre la transformada de Fourier de la señal original y las de las interpolaciones


#Filtro con 1000Hz y 500Hz


#Grafica para cada filtro
