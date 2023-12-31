#Importar librerias y funciones auxiliares
from Aux import apply_kernel, minDistance, findShortestPaths, find_path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import math
import os

main_path = os.getcwd()
parameters_path = 'parameters'
results_path = 'results'

if os.path.exists(os.path.join(main_path,results_path)) == False:
    os.mkdir(os.path.join(main_path, results_path))

if os.path.exists(os.path.join(main_path, parameters_path)) == False:
    os.mkdir(os.path.join(main_path, parameters_path))

if os.path.exists(os.path.join(main_path, results_path, 'GIF')) == False:
    os.mkdir(os.path.join(main_path, results_path, 'GIF'))

resistencias_finales = []
valores_posibles_citocromo = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

for concentracion_individual_cromo in valores_posibles_citocromo:
    resistencias_calculadas = []
    for hola in tqdm(range(100)):
        concentracion_crom = concentracion_individual_cromo
        densidad_E_Coli = 1/9 #Densidad de concentración de E coli
        densidad_shewanella = 7/9 #Densidad de concentración de Shewanella
        densidad_fluido = 1 - densidad_E_Coli-densidad_shewanella #Densidad de puntos en los que no hay nada

        densidad_conexiones = 8/8   #Densidad de filamenticos relacionado con la cantidad de tubos de conexion que tiene (completamente biologico)
        N = 20 #Dimensionalidad de rejilla relacionado con la cantidad de bichos simulados
        matrix_binaria = np.random.rand(N,N) #Rejilla de simulacion con la que se determina cosas

        adjusted_array = np.zeros((N, N), dtype=int) #Rejilla vacia que se va a ir llenando

        # Aplica las reglas
        for i in range(N):
            for j in range(N):
                value = matrix_binaria[i, j]
                if value < densidad_E_Coli:
                    adjusted_array[i, j] = 10
                elif densidad_E_Coli <= value <= densidad_E_Coli + densidad_shewanella:
                    adjusted_array[i, j] = 1
                else:
                    adjusted_array[i, j] = 0
                    
        capacidad_de_conectividad = densidad_conexiones*concentracion_crom
        nueva_matriz = apply_kernel(adjusted_array, capacidad_de_conectividad)

        nueva_matriz = list(nueva_matriz) #Nueva matriz final que tenemos con las conexiones dentro de la rejilla

        plt.imshow((-1)*np.array(nueva_matriz), cmap = "viridis")

        sources = []
        for posicion, i in enumerate(nueva_matriz[0]):
            if i == 1:
                sources.append((0, posicion))
        destinations = []
        for posicion, i in enumerate(nueva_matriz[-1]):
            N = len(nueva_matriz)-1
            if i == 1:
                destinations.append((N, posicion))

        shortest_paths = findShortestPaths(nueva_matriz, sources, destinations)

        resistencias_no_nulas = []
        for resistencia in np.array(shortest_paths).flatten():
            if resistencia != -1:
                resistencias_no_nulas.append(resistencia)
        inverso_resistencias = 1/np.array(resistencias_no_nulas)

        suma_inverso = np.sum(inverso_resistencias)
        resistencia_total = 1/suma_inverso
        if not math.isinf(resistencia_total):
            resistencias_calculadas.append(resistencia_total)
    resistencia_total_cromo_individual = np.mean(resistencias_calculadas)
    if not math.isinf(resistencia_total):
        resistencias_finales.append(resistencia_total_cromo_individual)
    else: 
        resistencias_finales.append(N*N)
#%%
np.array(resistencias_finales)

volateje_base = 100*(10**(-6))
resistencia_base = 1*(10**3)
corriente = volateje_base/(resistencia_base*np.array(resistencias_finales))

plt.figure(figsize = (9,6))
plt.title("Simulation of current as function of the concentration [C]", fontsize = 16)
plt.xlabel("Cytochrome concentration [C]", fontsize = 14)
plt.ylabel("Mean current [mA]", fontsize = 14)
plt.plot(valores_posibles_citocromo, corriente, linestyle = "--", label = "")
plt.scatter(valores_posibles_citocromo, corriente, color = "red", label = "Simulation data", marker= "o", alpha= 0.7)
plt.legend(fontsize = 14)
plt.savefig(os.path.join(main_path, results_path, 'plot_current_microplastic.jpg'), dpi = 1000)
#%%


plt.figure(figsize=(20,20))
plt.imshow((-1)*np.array(nueva_matriz), cmap = "viridis")
plt.axis("off")

#plt.savefig(os.path.join(main_path, results_path, "simulation_grid.jpg"), dpi = 1000)

#%%
def funcion(concentracion, posicion):
    concentracion_crom = concentracion
    densidad_E_Coli = 1/9 #Densidad de concentración de E coli
    densidad_shewanella = 6/9 #Densidad de concentración de Shewanella
    densidad_fluido = 1 - densidad_E_Coli-densidad_shewanella #Densidad de puntos en los que no hay nada

    densidad_conexiones = 8/8   #Densidad de filamenticos relacionado con la cantidad de tubos de conexion que tiene (completamente biologico)
    N = 50 #Dimensionalidad de rejilla relacionado con la cantidad de bichos simulados
    matrix_binaria = np.random.rand(N,N) #Rejilla de simulacion con la que se determina cosas

    adjusted_array = np.zeros((N, N), dtype=int) #Rejilla vacia que se va a ir llenando

    # Aplica las reglas
    for i in range(N):
        for j in range(N):
            value = matrix_binaria[i, j]
            
            if value < densidad_E_Coli:
                adjusted_array[i, j] = 0
            elif densidad_E_Coli <= value and value <= densidad_E_Coli + densidad_shewanella:
                adjusted_array[i, j] = 1
            else:
                adjusted_array[i, j] = 0
                
    capacidad_de_conectividad = densidad_conexiones*concentracion_crom
    nueva_matriz = apply_kernel(adjusted_array, capacidad_de_conectividad)

    nueva_matriz = list(nueva_matriz) #Nueva matriz final que tenemos con las conexiones dentro de la rejilla
    plt.figure()
    plt.title(f"Cytochrome-dependent activation of the grid \n Cytochrome Concentration : {round(concentracion,2)}")
    plt.imshow(nueva_matriz)
    plt.axis("off")
    plt.close()
    plt.savefig(os.path.join(main_path,results_path,'GIF',f'image_{posicion}.jpg'))

concentraciones = np.linspace(0,1,100)
for posicion, concentracion_propia in enumerate(concentraciones): 
    funcion(concentracion_propia, posicion)
    
capacidad_de_conectividad
