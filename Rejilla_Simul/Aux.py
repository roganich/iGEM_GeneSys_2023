from collections import deque
import numpy as np

def apply_kernel(matrix, densidad_tubulinas):

    rows, cols = matrix.shape
    
    # Crear una nueva matriz de dimensiones (3 * rows, 3 * cols) llena de ceros
    result = np.zeros((3 * rows, 3 * cols), dtype=float)
    
    # Recorrer la matriz binaria
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == 1:
                kernel = np.random.rand(3,3) #Generar kernel aleatorio para cada decision
                kernel = (kernel < densidad_tubulinas).astype(int) #Definir de acuerdo con la misma densidad para todos
                kernel[1,1] = 1
                # Si el valor en la matriz binaria es 1, colocar el kernel en la posición correspondiente
                result[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3] = kernel
    
    return result
"""
def find_path(matrix):
    if not matrix:
        return None

    rows, cols = len(matrix), len(matrix[0])

    def is_valid(x, y):
        return 0 <= x < rows and 0 <= y < cols and matrix[x][y] == 1

    moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    # Bordes derecha e izquierda
#    for i in range(rows):
#        if matrix[i][0] == 1:
#            # Comenzar una búsqueda desde el borde izquierdo
#            queue = deque([(i, 0)])
#            visited = set([(i, 0)])

#            while queue:
#                x, y = queue.popleft()

#                if y == cols - 1:
#                    return True  # Se encontró un camino hacia el borde derecho

                # Explorar las celdas vecinas 
#                for dx, dy in moves:
#                    nx, ny = x + dx, y + dy
#                    if is_valid(nx, ny) and (nx, ny) not in visited:
#                        queue.append((nx, ny))
#                        visited.add((nx, ny))

    # Bordes superior e inferior
    for j in range(cols):
        if matrix[0][j] == 1:
            # Comenzar una búsqueda desde el borde superior
            queue = deque([(0, j)])
            visited = set([(0, j)])

            while queue:
                x, y = queue.popleft()

                if x == rows - 1:
                    return True  # Se encontró un camino hacia el borde inferior

                # Explorar las celdas vecinas 
                for dx, dy in moves:
                    nx, ny = x + dx, y + dy
                    if is_valid(nx, ny) and (nx, ny) not in visited:
                        queue.append((nx, ny))
                        visited.add((nx, ny))

    return False
""" 
def find_path(matrix):
    if not matrix:
        return None

    rows, cols = len(matrix), len(matrix[0])

    def is_valid(x, y):
        return 0 <= x < rows and 0 <= y < cols and matrix[x][y] == 1

    moves = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)]

    # Bordes superior e inferior
    for j in range(cols):
        if matrix[0][j] == 1:
            # Comenzar una búsqueda desde el borde superior
            queue = deque([(0, j)])
            visited = set([(0, j)])

            while queue:
                x, y = queue.popleft()

                matrix[x][y] = 10  # Marcar el punto como parte del camino principal

                if x == rows - 1:
                    return matrix  # Devolver la matriz modificada con el camino principal

                # Explorar las celdas vecinas 
                for dx, dy in moves:
                    nx, ny = x + dx, y + dy
                    if is_valid(nx, ny) and (nx, ny) not in visited:
                        queue.append((nx, ny))
                        visited.add((nx, ny))

    return None