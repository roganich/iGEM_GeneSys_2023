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
    densidad_shewanella = 10/10 #Relacionado con la densidad espacial en el fluido
    densidad_conexiones = 6/8   #Relacionado con la cantidad de tubos de conexion que tiene (completamente biologico)
    N = 100 #Dimensionalidad de rejilla
    concentracion_crom = concentracion_crom/100
    matrix_binaria = np.random.rand(N,N) #Rejilla de simulacion
    matrix_binaria = (matrix_binaria < densidad_shewanella).astype(int)

    capacidad_de_conectividad = densidad_conexiones*concentracion_crom
    nueva_matriz = apply_kernel(matrix_binaria, capacidad_de_conectividad)

    nueva_matriz = list(nueva_matriz)
    camino = find_path(nueva_matriz)


    plt.figure(figsize=(15,15))
    plt.imshow((-1)*np.array(nueva_matriz), cmap = "viridis_r")
    if camino: 
        plt.title(r"Existe camino de medición", fontsize = 24)
    else: 
        plt.title(r"No existe camino de medición", fontsize = 24)
    plt.axis("off")

# %%
concentracion_slider = widgets.FloatSlider(
    value=0.0,
    min=0.0,
    max=100,
    step=0.1,
    description='Concentracion:')
interact(simulacion_rejilla, concentracion_crom=concentracion_slider)

# %%
