#Fourier2D.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import fftpack

#Parte real e imaginaria

transformada = fftpack.fft2(ndimage.imread('Arboles.png'))

#Calcular modulo para graficar

transformada_modulo = transformada.real**2 + transformada.imag**2
transformada_modulo = transformada_modulo**0.5
transformada_modulo = np.log(transformada_modulo)

#Grafica de la transformada

fig = plt.figure()
plt.imshow(transformada_modulo, cmap=plt.cm.plasma)
plt.axis('off')
plt.savefig('RojasLaura_FT2D.pdf')

#Filtro - reducir las magnitudes dentro de 4 elipses horizontales ubicadas en los picos de interes

for i in range(np.size(transformada,0)):
    for j in range(np.size(transformada,1)):
        #Reducir la magnitud en los 4 picos con elipses
        if ((i-10)/0.5)**2 + ((j-30)/2)**2 < 10**2:
            transformada[i,j] = transformada[i,j]*((i-10)**2+(j-30)**2)/20**2
        if ((i-60)/0.5)**2 + ((j-60)/2)**2 < 10**2:
            transformada[i,j] = transformada[i,j]*((i-60)**2+(j-60)**2)/20**2
        if ((i-190)/0.5)**2 + ((j-200)/2)**2 < 10**2:
            transformada[i,j] = transformada[i,j]*((i-190)**2+(j-200)**2)/20**2
        if ((i-245)/0.5)**2 + ((j-230)/2)**2 < 10**2:
            transformada[i,j] = transformada[i,j]*((i-245)**2+(j-230)**2)/20**2

#Calcular modulo para graficar

transformada_modulo = transformada.real**2 + transformada.imag**2
transformada_modulo = transformada_modulo**0.5
transformada_modulo = np.log(transformada_modulo)

#Grafica de la transformada filtrada

ig = plt.figure()
plt.imshow(transformada_modulo, cmap=plt.cm.plasma)
plt.axis('off')
plt.savefig('RojasLaura_FT2D_filtrada.pdf')

plt.figure()
plt.imshow(fftpack.ifft2(transformada).real, cmap=plt.cm.gray)
plt.axis('off')
plt.savefig('RojasLaura_Imagen_filtrada.pdf')
