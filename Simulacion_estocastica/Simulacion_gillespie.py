#%%
#Importe de librerias

import numpy as np
from tqdm import tqdm
from numba import jit,njit
import pandas as pd

@njit
def funcion_creacion_ARNmX():
    return Kx

@njit
def funcion_creacion_ARNmY(cantidad_X):
    return Ky*((cantidad_X**Hill)/(cantidad_X**Hill + Kxy**Hill))

@njit
def funcion_creacion_ARNmZ(cantidad_X, cantidad_Y):
    creacion_ARNmZ = Kz*((cantidad_X**Hill)/(cantidad_X**Hill + Kxz**Hill))*((cantidad_Y**Hill)/(cantidad_Y**Hill + Kyz**Hill))
    return creacion_ARNmZ

@njit
def funcion_creacion_X(cantidad_mX):
    return Kpx*cantidad_mX

@njit
def funcion_creacion_Y(cantidad_mY):
    return Kpy*cantidad_mY

@njit
def funcion_creacion_Z(cantidad_mZ):
    return Kpz*cantidad_mZ

@njit
def funcion_degradacion_ARNmX(cantidad_mX):
    return gammamx*cantidad_mX

@njit
def funcion_degradacion_ARNmY(cantidad_mY):
    return gammamy*cantidad_mY

@njit
def funcion_degradacion_ARNmZ(cantidad_mZ):
    return gammamz*cantidad_mZ

@njit
def funcion_degradacion_X(cantidad_X):
    return muX * cantidad_X 

@njit
def funcion_degradacion_Y(cantidad_Y):
    return muY * cantidad_Y 

@njit
def funcion_degradacion_Z(cantidad_Z):
    return muZ * cantidad_Z 

@njit
def modelo_constitutivo(cantidad_mX, cantidad_mY, cantidad_mZ, cantidad_X, cantidad_Y,cantidad_Z):

    propensidad_creacion_ARNmX = funcion_creacion_ARNmX()
    propensidad_creacion_ARNmY = funcion_creacion_ARNmY(cantidad_X)
    propensidad_creacion_ARNmZ = funcion_creacion_ARNmZ(cantidad_X, cantidad_Y)

    propensidad_creacion_proteinaX = funcion_creacion_X(cantidad_mX)
    propensidad_creacion_proteinaY = funcion_creacion_Y(cantidad_mY)
    propensidad_creacion_proteinaZ = funcion_creacion_Z(cantidad_mZ)

    propensidad_degradacion_ARNmX = funcion_degradacion_ARNmX(cantidad_mX)
    propensidad_degradacion_ARNmY = funcion_degradacion_ARNmY(cantidad_mY)
    propensidad_degradacion_ARNmZ = funcion_degradacion_ARNmZ(cantidad_mZ)

    propensidad_degradacion_proteinaX = funcion_degradacion_X(cantidad_X)
    propensidad_degradacion_proteinaY = funcion_degradacion_Y(cantidad_Y)
    propensidad_degradacion_proteinaZ = funcion_degradacion_Z(cantidad_Z)

    return propensidad_creacion_ARNmX, propensidad_creacion_ARNmY, propensidad_creacion_ARNmZ, propensidad_creacion_proteinaX, propensidad_creacion_proteinaY, propensidad_creacion_proteinaZ, propensidad_degradacion_ARNmX, propensidad_degradacion_ARNmY, propensidad_degradacion_ARNmZ, propensidad_degradacion_proteinaX, propensidad_degradacion_proteinaY, propensidad_degradacion_proteinaZ

@njit('f8[:](f8[:],f8)')
def Gillespie(trp0,tmax):
    """
    Esta funcion se emplea solamente para hacer la evolución de un paso individual en la celula. Evoluciona no un paso temporal, 
    pero si temporalmente la cantidad de veces que pueda evolucionar antes del tmax en una corrida
    """
    
    t,ARNmX, ARNmY, ARNmZ, proteinaX, proteinaY, proteinaZ =trp0 

    while t < tmax:
        s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8, s_9, s_10, s_11, s_12 = modelo_constitutivo(ARNmX, ARNmY, ARNmZ, proteinaX, proteinaY, proteinaZ)
        S_T = s_1 + s_2 + s_3 + s_4 + s_5 + s_6 + s_7 + s_8 + s_9 + s_10 + s_11 + s_12

        τ = (-1/S_T)*np.log(np.random.rand())
        x = np.random.rand()

        if x <= (s_1)/S_T:
            ARNmX += 1

        elif x<= (s_1 + s_2)/S_T:
            ARNmY += 1
        
        elif x <= (s_1 + s_2 + s_3)/S_T :
            ARNmZ+=1
        
        elif x <= (s_1 + s_2 + s_3 + s_4)/S_T :
            proteinaX+=1
        
        elif x <= (s_1 + s_2 + s_3 + s_4 + s_5)/S_T :
            proteinaY+= 1

        elif x <= (s_1 + s_2 + s_3 + s_4 + s_5 + s_6)/S_T :
            proteinaZ += 1
        
        elif x <= (s_1 + s_2 + s_3 + s_4 + s_5 + s_6 + s_7)/S_T :
            ARNmX-= 1
        
        elif x <= (s_1 + s_2 + s_3 + s_4 + s_5 + s_6 + s_7 + s_8)/S_T :
            ARNmY-=1

        elif x <= (s_1 + s_2 + s_3 + s_4 + s_5 + s_6 + s_7 + s_8 + s_9)/S_T :
        
            ARNmZ-= 1

        elif x <= (s_1 + s_2 + s_3 + s_4 + s_5 + s_6 + s_7 + s_8 + s_9 + s_10)/S_T :

            proteinaX-=1

        elif x <= (s_1 + s_2 + s_3 + s_4 + s_5 + s_6 + s_7 + s_8 + s_9 + s_10 + s_11)/S_T :
            proteinaY-=1

        else: 
            proteinaZ-=1

        t+=τ
    return np.array([t,ARNmX, ARNmY, ARNmZ, proteinaX, proteinaY, proteinaZ]) 

@njit('f8[:,:](f8[:],f8[:])')
def Estado_celula(X0,tiempos):

    
    X = np.zeros((len(tiempos),len(X0)))
    X[0] = X0
    
    for i in range(1,len(tiempos)):
        X[i] = Gillespie(X[i-1],tiempos[i])
    
    return X

x0 = np.array([0., 0., 0., 0., 0., 0., 0.])

num_cel = 1000 #número de células 
celulas = np.array([Estado_celula(x0,np.arange(0.,700.,2.)) for i in tqdm(range(num_cel))])

celulas_prom = np.mean(celulas,axis=0) #axis = 0 saca el promedio componente a componente de cada célula.
