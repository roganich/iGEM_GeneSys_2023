#%%
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
from tqdm import tqdm

class QItem:
    def __init__(self, row, col, dist):
        self.row = row
        self.col = col
        self.dist = dist

def minDistance(grid, src, dest):
    visited = [[False for _ in range(len(grid[0]))] for _ in range(len(grid))]
    source = QItem(src[0], src[1], 0)
    visited[source.row][source.col] = True
    queue = [source]

    while queue:
        source = queue.pop(0)

        if (source.row, source.col) == dest:
            return source.dist

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = source.row + dr, source.col + dc
            if (
                0 <= new_row < len(grid)
                and 0 <= new_col < len(grid[0])
                and grid[new_row][new_col] == 1
                and not visited[new_row][new_col]
            ):
                queue.append(QItem(new_row, new_col, source.dist + 1))
                visited[new_row][new_col] = True

    return -1

def findShortestPaths(grid, sources, destinations):
    shortest_paths = []

    for source in sources:
        paths = []
        for destination in destinations:
            distance = minDistance(grid, source, destination)
            paths.append(distance)
        shortest_paths.append(paths)

    return shortest_paths
resistencias_calculadas = []
for i in tqdm(range(10)):
    concentracion_crom = 0.7
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

    nueva_matriz = list(nueva_matriz)
    #plt.imshow((-1)*np.array(nueva_matriz), cmap = "viridis")

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

    sumas = []
    for camino in shortest_paths:
        camino_filtrado = [elemento for elemento in camino if elemento != -1]
        inversos = [1/resistencia for resistencia in camino_filtrado]
        suma = sum(inversos)
        sumas.append(suma)

    # Filtra los valores de resistencia que no son cero
    resistencias_no_cero = [resistencia for resistencia in sumas if resistencia != 0]

    if resistencias_no_cero:
        # Calcula los inversos de las resistencias que no son cero
        inversos = [1/resistencia for resistencia in resistencias_no_cero]

        # Suma los inversos
        suma_inversos = sum(inversos)

        # Calcula el inverso de la suma total
        resistencia_equivalente = 1 / suma_inversos
    resistencias_calculadas.append(resistencia_equivalente)

# %%
np.mean(resistencias_calculadas)
# %%
np.std(resistencias_calculadas)
# %%
