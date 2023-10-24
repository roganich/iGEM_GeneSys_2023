
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
import math

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

resistencias_finales = []
valores_posibles_citocromo = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

for concentracion_individual_cromo in valores_posibles_citocromo:
    resistencias_calculadas = []
    for hola in tqdm(range(10)):
        concentracion_crom = concentracion_individual_cromo
        densidad_E_Coli = 1/9 #Densidad de concentración de E coli
        densidad_shewanella = 7/9 #Densidad de concentración de Shewanella
        densidad_fluido = 1 - densidad_E_Coli-densidad_shewanella #Densidad de puntos en los que no hay nada

        densidad_conexiones = 8/8   #Densidad de filamenticos relacionado con la cantidad de tubos de conexion que tiene (completamente biologico)
        N = 10 #Dimensionalidad de rejilla relacionado con la cantidad de bichos simulados
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

np.array(resistencias_finales)

corriente = 1/np.array(resistencias_finales)

plt.figure(figsize = (9,6))
plt.title("Simulación corriente en función de concentración [C]", fontsize = 16)
plt.xlabel("Concentración citocromo [C]", fontsize = 14)
plt.ylabel("Corriente medida [A]", fontsize = 14)
plt.plot(valores_posibles_citocromo, corriente)
plt.scatter(valores_posibles_citocromo, corriente, color = "red")
plt.savefig("Resultado_simulacion.jpg", dpi = 1000)

