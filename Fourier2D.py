#Fourier2D.py

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import fftpack

#Parte real e imaginaria

transformada = fftpack.fft2(ndimage.imread('arbol.PNG'))

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
