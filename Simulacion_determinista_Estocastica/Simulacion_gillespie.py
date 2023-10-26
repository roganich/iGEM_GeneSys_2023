#%%
#Importe de librerias
import numpy as np
from tqdm import tqdm
from numba import jit,njit
import pandas as pd
import os
#%%
main_path = os.getcwd()
parameters_path = 'parameters'
results_path = 'results'

df_params = pd.read_csv(os.path.join('params.csv'))

params_vals = list(df_params['value'])
variable_names = ['p_LuxI', 'p_mtrC']
cond_init = [0.1]*len(variable_names)

sigma_pNahR, sigma_rNahR, gamma_rNahR, gamma_pNahR, m, K, K1, K2, gamma_Ec, gamma_out, gamma_sh, sigma_pLuxI, gamma_rLuxI, alpha_LuxI, beta_LuxI, K_LuxI, h_LuxI, gamma_pLuxI, beta_LuxRAHL, K_LuxRAHL, sigma_pLuxR, sigma_rLuxR, gamma_rLuxR, gamma_pLuxR, sigma_pmtrC, gamma_rmtrC, alpha_mtrC, beta_mtrC, K_mtrC, h_mtrC, gamma_pmtrC, V_T, S, E, sigma_AHL = params_vals

v_sh = np.pi*(0.55**2)*2.5
v_ec = np.pi*(0.5**2)*1.5 

V_sh = S*v_sh
V_ec = E*v_ec

constante_grande = (S*K2*K1*sigma_AHL)/(gamma_Ec + gamma_out*K1*((V_T-E*v_ec-S*v_sh)/E*v_ec) + gamma_sh*K2*K1*S*v_sh/E*v_ec)*(v_sh/v_ec)


@njit
def funcion_creacion_pNahR():
    return (sigma_pNahR*sigma_rNahR)/gamma_rNahR

@njit
def funcion_destruccion_pNahR(cantidad_pNahR):
    return gamma_pNahR*cantidad_pNahR

@njit
def funcion_mNahR(cantidad_pNahR, cantidad_m):
    return (cantidad_pNahR*cantidad_m)/(K + cantidad_m)

@njit
def funcion_creacion_pLuxI(cantidad_pNahR, cantidad_m):
    return (sigma_pLuxI/gamma_rLuxI)*(alpha_LuxI + (beta_LuxI)/(1 + (funcion_mNahR(cantidad_pNahR, cantidad_m)/K_LuxI)**(-h_LuxI))) 

@njit
def funcion_destruccion_pLuxI(cantidad_pLuxI):
    return gamma_pLuxI*cantidad_pLuxI

@njit
def cantidad_AHL(cantidad_pLuxI):
    return constante_grande*cantidad_pLuxI

@njit
def funcion_creacion_pLuxR():
    return S*(sigma_pLuxR*sigma_rLuxR)/gamma_rLuxR

@njit
def funcion_destruccion_pLuxR(cantidad_pLuxR):
    return gamma_pLuxR*cantidad_pLuxR

@njit
def funcion_LuxRAHL(cantidad_pLuxR, cantidad_pLuxI):
    return (cantidad_pLuxR*cantidad_AHL(cantidad_pLuxI))/(K_LuxRAHL*V_sh + cantidad_AHL(cantidad_pLuxI))

@njit
def funcion_creation_pmtrc(cantidad_pLuxR, cantidad_pLuxI):
    return S*(sigma_pmtrC/gamma_rmtrC)*(alpha_mtrC + (beta_mtrC)/(1 + (funcion_LuxRAHL(cantidad_pLuxR, cantidad_pLuxI)/K_mtrC)**(-h_mtrC))) 

@njit
def funcion_destruccion_pmtrc(cantidad_pmtrc):
    return gamma_pmtrC*cantidad_pmtrc


@njit
def modelo_constitutivo(cantidad_pNahR, cantidad_m,cantidad_pLuxI, cantidad_pLuxR, cantidad_pmtrc):

    propensidad_creacion_pNahR = funcion_creacion_pNahR()
    propensidad_destruccion_pNahR =  funcion_destruccion_pNahR(cantidad_pNahR)

    propensidad_creacion_pLuxI = funcion_creacion_pLuxI(cantidad_pNahR, cantidad_m)
    propensidad_destruccion_pLuxI = funcion_destruccion_pLuxI(cantidad_pLuxI)

    propensidad_creacion_pLuxR = funcion_creacion_pLuxR()
    propensidad_destruccion_pLuxR = funcion_destruccion_pLuxR(cantidad_pLuxR)

    propensidad_creacion_pmtrc = funcion_creation_pmtrc(cantidad_pLuxR, cantidad_pLuxI)
    propensidad_destruccion_pmtrc = funcion_destruccion_pmtrc(cantidad_pmtrc)


    return propensidad_creacion_pNahR, propensidad_destruccion_pNahR, propensidad_creacion_pLuxI, propensidad_destruccion_pLuxI , propensidad_creacion_pLuxR, propensidad_destruccion_pLuxR, propensidad_creacion_pmtrc , propensidad_destruccion_pmtrc

#@njit('f8[:](f8[:],f8)')
@njit
def Gillespie(trp0,tmax):
    """
    Esta funcion se emplea solamente para hacer la evolución de un paso individual en la celula. Evoluciona no un paso temporal, 
    pero si temporalmente la cantidad de veces que pueda evolucionar antes del tmax en una corrida
    """
    
    t, cantidad_pNahR, cantidad_m, cantidad_pLuxI, cantidad_pLuxR, cantidad_pmtrc =trp0 

    while t < tmax:
        s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8 = modelo_constitutivo(cantidad_pNahR, cantidad_m,cantidad_pLuxI, cantidad_pLuxR, cantidad_pmtrc)
        S_T = s_1 + s_2 + s_3 + s_4 + s_5 + s_6 + s_7 + s_8 

        τ = (-1/S_T)*np.log(np.random.rand())
        x = np.random.rand()

        if x <= (s_1)/S_T:
            cantidad_pNahR += 1

        elif x<= (s_1 + s_2)/S_T:
            cantidad_pNahR -= 1
        
        elif x <= (s_1 + s_2 + s_3)/S_T :
            cantidad_pLuxI +=1
        
        elif x <= (s_1 + s_2 + s_3 + s_4)/S_T :
            cantidad_pLuxI -=1
        
        elif x <= (s_1 + s_2 + s_3 + s_4 + s_5)/S_T :
            cantidad_pLuxR += 1

        elif x <= (s_1 + s_2 + s_3 + s_4 + s_5 + s_6)/S_T :
            cantidad_pLuxR -= 1
        
        elif x <= (s_1 + s_2 + s_3 + s_4 + s_5 + s_6 + s_7)/S_T :
            cantidad_pmtrc += 1
        
        else:
            cantidad_pmtrc -= 1

        t+=τ

    return np.array([t, cantidad_pNahR, cantidad_m, cantidad_pLuxI, cantidad_pLuxR, cantidad_pmtrc]) 

#@njit('f8[:,:](f8[:],f8[:])')
@njit
def Estado_celula(X0,tiempos):

    X = np.zeros((len(tiempos),len(X0)))
    X[0] = X0
    
    for i in range(1,len(tiempos)):
        X[i] = Gillespie(X[i-1],tiempos[i])
    
    return X
#%%
x0 = np.array([0., 0., 0., 0., 0., 0.])

num_cel = 10 #número de células 
celulas = np.array([Estado_celula(x0,np.arange(0.,700.,1.)) for i in tqdm(range(num_cel))])

celulas_prom = np.mean(celulas,axis=0) #axis = 0 saca el promedio componente a componente de cada célula.
#%%
import matplotlib.pyplot as plt
plt.plot(celulas[:,:,0])
# %%
print(len(celulas_prom[:,0]))
# %%
constante_grande
# %%
celulas[:,:,0].shape
# %%
for i in range(0,9):
    plt.plot(celulas[:,:,5][i])
# %%
