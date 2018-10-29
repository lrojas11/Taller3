#Fourier.py

import numpy as np
from scipy.fftpack import fftfreq, ifft
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


#Almacenar datos

signal = np.loadtxt('signal.dat', delimiter = ',')

incompletos = np.loadtxt('incompletos.dat', delimiter = ',')

#Grafica de signal.dat

cs1 = signal[:,0]
cs2 = signal[:,1]

plt.title('Signal')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.grid()
plt.plot(cs1,cs2, color = 'crimson')
plt.savefig('RojasLaura_signal.pdf')

#Transformada discreta de Fourier a los datos de signal


def transformada (cs1,cs2):
    N = len(cs2)
    lista_real = []
    lista_imaginaria = []
    for k in range(N):
        suma_real= 0
        suma_imaginaria = 0
        for n in range (N):
            suma_real = suma_real + np.cos((-2*np.pi*k*n)/N) * cs2[n]
            suma_imaginaria = suma_real + np.sin((-2*np.pi*k*n)/N) * cs2[n]
        lista_real.append (suma_real)
        lista_imaginaria.append (suma_imaginaria)
    return lista_real,lista_imaginaria
lista_real, lista_imaginaria = transformada (cs1,cs2)



#Grafica de transformada de Fourier

periodo = cs1[1]-cs1[0]
N = len(cs1)
frecuencias = fftfreq(N,periodo)
modulo = np.array(lista_real)**2 + np.array(lista_imaginaria)**2

plt.figure()
plt.plot(frecuencias, modulo)
plt.xlabel('Frecuencia')
plt.ylabel('Amplitud')
plt.savefig('RojasLaura_TF.pdf')


#Frecuencias principales

print('Las frecuencias principales son:', frecuencias[4],frecuencias[6], frecuencias[10])


#Filtro, transformada inversa y grafica

def filtro (lista_real,lista_imaginaria,frecuencias):
    N = len(frecuencias)
    for i in range (N):
        if (frecuencias[i] > 1000):
            lista_real[i] = 0
            lista_imaginaria [i] = 0
        if (frecuencias[i] < -1000):
            lista_real[i] = 0
            lista_imaginaria[i] = 0
    return lista_real,lista_imaginaria

lista_real, lista_imaginaria = filtro(lista_real,lista_imaginaria,frecuencias)

def transformada_inversa (lista_real,lista_imaginaria):
    inversa = ifft (np.array(lista_real) + 1j*np.array(lista_imaginaria))
    return inversa.real

inversa = transformada_inversa(lista_real,lista_imaginaria)

plt.figure()
plt.plot (cs1, inversa)
plt.show()


#Mensaje porque no se puede hacer la transformada para los datos de incompletos

print ('Porque los datos no están muestrados uniformemente')

#Interpolacion cuadratica y cubica de incompletos con 512 puntos - transformada de Fourier de los datos interpolados

ci1 = incompletos[:,0]
ci2 = incompletos[:,1]

xmin = min(ci1)
xmax = max(ci1)

x = np.linspace(xmin,xmax,512)

def interpol_cuadratica(ci1,ci2):
      cuadratica = interp1d(ci1, ci2, kind='quadratic')
      return cuadratica

cuadratica = interpol_cuadratica(ci1,ci2)
y = cuadratica(x)


def interpol_cubica(ci1,ci2):
    cubica = interp1d(ci1, ci2, kind='cubic')
    return cubica

cubica = interpol_cubica(ci1,ci2)
y1 = cubica(x)

real_incompletos_cuadratica, imaginaria_incompletos_cuadratica = transformada (x,y)

real_incompletos_cubica, imaginaria_incompletos_cubica = transformada(x, y1)

#Grafica de las 3 transformadas (2 de signal y  de incompletos)




#Diferencias encontradas entre la transformada de Fourier de la señal original y las de las interpolaciones


#Filtro con 1000Hz y 500Hz


#Grafica para cada filtro
