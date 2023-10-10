#%%
#Importar librerias y funciones auxiliares
from Aux import apply_kernel, find_path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display

#Parametros biologicos de densidades
def simulacion_rejilla(concentracion_crom):
    densidad_E_Coli = 3/9
    densidad_shewanella = 4/9
    densidad_fluido = 1 - densidad_E_Coli-densidad_shewanella

    densidad_conexiones = 8/8   #Relacionado con la cantidad de tubos de conexion que tiene (completamente biologico)
    N = 100 #Dimensionalidad de rejilla
    matrix_binaria = np.random.rand(N,N) #Rejilla de simulacion

    adjusted_array = np.zeros((N, N), dtype=int)

    # Aplica las reglas
    for i in range(N):
        for j in range(N):
            value = matrix_binaria[i, j]
            if value < densidad_E_Coli:
                adjusted_array[i, j] = 2
            elif densidad_E_Coli <= value <= densidad_E_Coli + densidad_shewanella:
                adjusted_array[i, j] = 1
            else:
                adjusted_array[i, j] = 0


    capacidad_de_conectividad = densidad_conexiones*concentracion_crom
    nueva_matriz = apply_kernel(adjusted_array, capacidad_de_conectividad)

    nueva_matriz = list(nueva_matriz)
    camino = find_path(nueva_matriz)

    plt.figure(figsize=(15,15))
    plt.imshow((-1)*np.array(nueva_matriz), cmap = "viridis")

    if camino: 
        plt.title(r"Existe camino de medición", fontsize = 24)
    else: 
        plt.title(r"No existe camino de medición", fontsize = 24)
    plt.axis("off")
    plt.savefig("Camino_simulacion.jpg", dpi = 200)
concentracion_slider = widgets.FloatSlider(
    value=0.0,
    min=0.0,
    max=1,
    step=0.01,
    description='Concentracion:')

interact(simulacion_rejilla, concentracion_crom=concentracion_slider)
display(concentracion_slider)
# %%
densidad_E_Coli = 1/9
densidad_shewanella = 6/9
densidad_fluido = 1 - densidad_E_Coli-densidad_shewanella

densidad_conexiones = 8/8   #Relacionado con la cantidad de tubos de conexion que tiene (completamente biologico)
N = 50 #Dimensionalidad de rejilla
matrix_binaria = np.random.rand(N,N) #Rejilla de simulacion

#matrix_binaria = (matrix_binaria < densidad_shewanella).astype(int)

adjusted_array = np.zeros((N, N), dtype=int)

# Aplica las reglas
for i in range(N):
    for j in range(N):
        value = matrix_binaria[i, j]
        if value < densidad_E_Coli:
            adjusted_array[i, j] = 2
        elif densidad_E_Coli <= value <= densidad_E_Coli + densidad_shewanella:
            adjusted_array[i, j] = 1
        else:
            adjusted_array[i, j] = 0

plt.imshow(adjusted_array)
# %%
adjusted_array
# %%
