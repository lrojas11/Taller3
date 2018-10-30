import matplotlib.pyplot as plt
import pylab
import numpy as np
from matplotlib import rc
import requests


#A traves del modulo request de python se genera pedido a través del modulo http a la pagina y el resultado se guarda en formato txt
url = 'http://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/WDBC.dat'
r = requests.get(url)
with open("WDBC.txt",'wb') as f:
    f.write(r.content)

#Se cargan los datos de los pacientes, omitiendo los valores de ID y el string (Benigno o Maligno) a una variable de parametros('param')
param=np.loadtxt('WDBC.txt',delimiter=',',usecols=(2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31))

#Guarda los valores de los tipos de tumor (Benigno o Maligno (B-M))
tipo= np.genfromtxt('WCDB.txt',delimiter =',', dtype = 'str')
M_B = tipo[:,1]

#Guardar las variables de cada parametro en Arrays
T1= param[:,0]
T2= param[:,1]
T3= param[:,2]
T4= param[:,3]
T5= param[:,4]
T6= param[:,5]
T7= param[:,6]
T8= param[:,7]
T9= param[:,8]
T10= param[:,9]
T11= param[:,10]
T12= param[:,11]
T13= param[:,12]
T14= param[:,13]
T15= param[:,14]
T16= param[:,15]
T17= param[:,16]
T18= param[:,17]
T19= param[:,18]
T20= param[:,19]
T21= param[:,20]
T22= param[:,21]
T23= param[:,22]
T24= param[:,23]
T25= param[:,24]
T26= param[:,25]
T27= param[:,26]
T28= param[:,27]
T29= param[:,28]
T30= param[:,29]

#Se normalizan los datos para realizar PCA
T1_norm = T1 / T1.max(axis=0)
T2_norm = T2 / T2.max(axis=0)
T3_norm = T3 / T3.max(axis=0)
T4_norm = T4 / T4.max(axis=0)
T5_norm = T5 / T5.max(axis=0)
T6_norm = T6 / T6.max(axis=0)
T7_norm = T7 / T7.max(axis=0)
T8_norm = T8 / T8.max(axis=0)
T9_norm = T9 / T9.max(axis=0)
T10_norm = T10 / T10.max(axis=0)
T11_norm = T11 / T11.max(axis=0)
T12_norm = T12 / T12.max(axis=0)
T13_norm = T13 / T13.max(axis=0)
T14_norm = T14 / T14.max(axis=0)
T15_norm = T15 / T15.max(axis=0)
T16_norm = T16 / T16.max(axis=0)
T17_norm = T17 / T17.max(axis=0)
T18_norm = T18 / T18.max(axis=0)
T19_norm = T19 / T19.max(axis=0)
T20_norm = T20 / T20.max(axis=0)
T21_norm = T21 / T21.max(axis=0)
T22_norm = T22 / T22.max(axis=0)
T23_norm = T23 / T23.max(axis=0)
T24_norm = T24 / T24.max(axis=0)
T25_norm = T25 / T25.max(axis=0)
T26_norm = T26 / T26.max(axis=0)
T27_norm = T27 / T27.max(axis=0)
T28_norm = T28 / T28.max(axis=0)
T29_norm = T29 / T29.max(axis=0)
T30_norm = T30 / T30.max(axis=0)

#Matriz con todos los datos organizados y normalizados
D= np.column_stack((T1_norm ,T2_norm ,T3_norm ,T4_norm ,T5_norm ,T6_norm ,T7_norm ,T8_norm ,T9_norm ,T10_norm ,T11_norm ,T12_norm ,T13_norm ,T14_norm ,T15_norm ,T16_norm ,T17_norm ,T18_norm ,T19_norm ,T20_norm ,T21_norm ,T22_norm ,T23_norm ,T24_norm ,T25_norm ,T26_norm ,T27_norm ,T28_norm ,T29_norm ,T30_norm))


#Calcular la matriz de covarianza para los datos de las bandas, se usa la transpuesta
matriz_cov = np.cov(D.T)
print('La Matriz de Covarianza')
print(matriz_cov)

#Se busca busca obtener los DOS componentes principales , queremos que el primer autovector corresponda al autovalor mas grande
valores, vectores = np.linalg.eig(matriz_cov)
#Reorganiza valores(basado del PCA.ipynb / Metodos computacionales Uniandes)
orden = vectores[:,0].copy()
vectores[:,0] = vectores[:,1]
vectores[:,1] = orden[:]

orden = valores[0]
valores[0] =valores[1]
valores[1] = orden

#Obtener e imprimir en la consola los DOS componentes principales en orden.
#Los vectores propios con los valores propios mÃ¡s bajos llevan la menor informaciÃ³n sobre la distribuciÃ³n de los datos
#Esos son los que no se toman en cuenta
print('Los parametros mas importantes en la base de componentes de los autovetores van a ser los dos mayores, por esta razon se reorganizan los autovectores y autovalores y se extraen estos')
#print ("Valores Propios de mayor a menor ", valores)
#print ("vectores Propios de mayor a menor ", vectores)
print(vectores[:, [0, 1]])

#Graficar los datos nuevamente en el sistema de referencia de los dos componentes principales.

#PRODUCTO PUNTO ENTRE LOS VECTORES Y LA MATRIZ DE LOS DATOS ORGANIZADOS
comp_principal = np.dot(vectores.T, D.T)

fig1 = plt.figure(figsize=(20, 8))
plt.rc('font', family='serif')
fig1.suptitle(r'PCA Cancer Cells Diagnose', fontsize=16,color='Black')
#Rotulación de ejes de la gráfica y rejilla
plt.rc('font', family='serif')
plt.xlabel(r'Componente principal 1',fontsize=16,  color='Black')
plt.rc('font', family='serif')
plt.ylabel(r'Componente principal 2',fontsize=16)
plt.grid(True)
ax = plt.axes()
plt.scatter(comp_principal[0,:][np.where(M_B == 'M')], comp_principal[1,:][np.where(M_B == 'M')], label = 'MALIGNO',color='red',edgecolors='black',s=100)
plt.scatter(comp_principal[0,:][np.where(M_B == 'B')], comp_principal[1,:][np.where(M_B == 'B')], label = 'BENIGNO', color='green',edgecolors='black',s=100)
x_line = np.linspace(-1,1)
fig1.savefig('ROJASLAURA_PCA.pdf')


print('Si es posible realizar un diagnostico a tiempo para la deteccion prematura de las celulas cancerigenas usando PCA, \n pues al seleccionar un analisis de dos dimensiones se hace más sencillo evaluar la matriz de covarianza con \n mayor precision y un menos tiempo es necesario para determinar las caracteristicas claves')
