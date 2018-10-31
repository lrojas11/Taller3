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

plt.title('Señal')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.grid()
plt.plot(cs1,cs2, color = 'crimson')
plt.savefig('RojasLaura_signal.pdf')

#Transformada discreta de Fourier a los datos de signal

def transformada (cs2):
    N = len(cs2)
    lista = []
    for k in range(N):
        suma = 0
        for n in range (N):
            suma += np.exp((-1j*-2*np.pi*k*n)/N) * cs2[n]    #Formula
        lista.append (suma)
    return lista
lista = transformada (cs2)


#Grafica de transformada de Fourier

periodo = cs1[1]-cs1[0]
N = np.shape(signal)
frecuencias = fftfreq(N[0],periodo)
modulo = np.abs(lista)

plt.figure()
plt.plot(frecuencias, modulo, color = 'mediumpurple')
plt.grid()
plt.xlim(-5000,5000)
plt.title('Transformada de Fourier - Señal')
plt.xlabel('Frecuencia')
plt.ylabel('Amplitud')
plt.savefig('RojasLaura_TF.pdf')


#Frecuencias principales

print('Las frecuencias principales son:', frecuencias[4],frecuencias[6], frecuencias[10])


#Filtro, transformada inversa y grafica

def filtro (lista,frecuencias):
    N = len(frecuencias)
    for i in range (N):
        if (frecuencias[i] > 1000):
            lista[i] = 0
        if (frecuencias[i] < -1000):
            lista[i] = 0
    return lista

lista = filtro(lista,frecuencias)

def transformada_inversa (lista):
    inversa = ifft (np.array(lista))
    return inversa.real

inversa = transformada_inversa(lista)


plt.figure()
plt.plot (cs1, inversa, color = 'dodgerblue')
plt.grid()
plt.title('Señal filtrada - Transformada inversa')
plt.xlabel('Frecuencia')
plt.ylabel('Amplitud')
plt.savefig('RojasLaura_Filtrada.pdf')


#Mensaje porque no se puede hacer la transformada para los datos de incompletos

print ('Porque los datos no están muestrados uniformemente, por lo que no tendría sentido realizar la transformada')

#Interpolacion cuadratica y cubica de incompletos con 512 puntos - transformada de Fourier de los datos interpolados

ci1 = incompletos[:,0]
ci2 = incompletos[:,1]

xmin = ci1[0]
xmax = ci1[-1]

x = np.linspace(xmin,xmax,512)

#Interpolaciones
interpol_cuadratica = interp1d(ci1, ci2, kind='quadratic')

interpol_cubica = interp1d(ci1, ci2, kind='cubic')

cuadratica = interpol_cuadratica(x)

cubica = interpol_cubica(x)

#Transformadas a las Interpolaciones

transf_cuadratica = transformada(cuadratica)
transf_cubica = transformada(cubica)

#Grafica de las 3 transformadas

periodo2 = ci1[1]-ci1[0]

frecuencia_interpol = fftfreq(len(x), periodo2)


plt.figure()
plt.subplot(3,1,1)
plt.plot(frecuencias, modulo, color = 'mediumorchid', label = 'Señal original')
plt.subplot(3,1,2)
plt.plot(frecuencia_interpol, np.abs(transf_cuadratica), color = 'lightcoral', label = 'Interpolación cuadrática')
plt.subplot(3,1,3)
plt.plot(frecuencia_interpol, np.abs(transf_cubica), label = 'Interpolación cúbia', color = 'lightseagreen' )
plt.savefig('RojasLaura_TF_interpola.pdf')


#Diferencias encontradas entre la transformada de Fourier de la señal original y las de las interpolaciones



#Filtro con 1000Hz y 500Hz



#Grafica para cada filtro
