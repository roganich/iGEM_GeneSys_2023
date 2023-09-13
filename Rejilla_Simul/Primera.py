#%%
#Importar librerias y funciones auxiliares
from Aux import apply_kernel, find_path
import numpy as np
import matplotlib.pyplot as plt

#Parametros biologicos de densidades

densidad_shewanella = 7/10
densidad_conexiones = 5/10
densidad_cro = 10/10
N = 100 #Dimensionalidad de rejilla

matrix_binaria = np.random.rand(N,N) #Rejilla de simulacion
matrix_binaria = (matrix_binaria < densidad_shewanella).astype(int)


nueva_matriz = apply_kernel(matrix_binaria, densidad_conexiones)

nueva_matriz = list(nueva_matriz)
camino = find_path(nueva_matriz)

if camino: 
    print("Existe camino de activacion")
else: 
    print("No existe camino de conexion")
plt.figure(figsize=(15,15))
plt.imshow((-1)*np.array(nueva_matriz), cmap = "viridis_r")
plt.axis("off")
# %%
