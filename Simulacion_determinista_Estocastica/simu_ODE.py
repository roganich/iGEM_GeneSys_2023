#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import os

def model_biosensor(variables, t, params):
    p_LuxI, p_mtrC = variables
    sigma_pNahR, sigma_rNahR, gamma_rNahR, gamma_pNahR, m, K, K1, K2, gamma_Ec, gamma_out, gamma_sh, sigma_pLuxI, gamma_rLuxI, alpha_LuxI, beta_LuxI, K_LuxI, h_LuxI, gamma_pLuxI, beta_LuxRAHL, K_LuxRAHL, sigma_pLuxR, sigma_rLuxR, gamma_rLuxR, gamma_pLuxR, sigma_pmtrC, gamma_rmtrC, alpha_mtrC, beta_mtrC, K_mtrC, h_mtrC, gamma_mtrC, V_T, S, E, sigma_AHL = params
    
    '''
    E = numero de E. coli
    S = numero de S. oneidensis
    v_ec = volumen de 1 E. coli
    v_sh = volumen de 1 S. oneidensis
    V_ec = E * v_ec
    V_sh = S * v_sh
    '''

    #los tamaños estan en micrometros (10^-6)
    v_sh = np.pi*(0.55**2)*2.5
    v_ec = np.pi*(0.5**2)*1.5 
    V_sh = S*v_sh
    
    V_ec = E*v_ec

    #V_out = V_T - V_ec - V_sh
    
    p_NahR = (sigma_pNahR*sigma_rNahR)/(gamma_rNahR*gamma_pNahR)
    microNahR = (p_NahR*m/(K+m))
    
    dp_LuxIdt = sigma_pLuxI/gamma_rLuxI*(alpha_LuxI + beta_LuxI/(1+ (microNahR/K_LuxI)**-h_LuxI)) - gamma_pLuxI*p_LuxI

    AHL_sh = (S*K1*K2*sigma_AHL*p_LuxI)/(gamma_Ec + gamma_out*K1*((V_T-E*v_ec-S*v_sh)/E*v_ec) + gamma_sh*K2*K1*S*v_sh/E*v_ec)*(v_sh/v_ec)
    p_LuxR = S*sigma_pLuxR*sigma_rLuxR/(gamma_pLuxR*gamma_rLuxR)
    luxRAHL = (p_LuxR*AHL_sh)/((K_LuxRAHL*V_sh)+AHL_sh)

    dp_mtrCdt = S*(sigma_pmtrC/gamma_rmtrC)*(alpha_mtrC + (beta_mtrC)/(1+(luxRAHL/K_mtrC)**(-h_mtrC))) - gamma_mtrC*p_mtrC 

    dXdt = [dp_LuxIdt, dp_mtrCdt]

    return dXdt
#%%
df_params = pd.read_csv(os.path.join('params.csv'))

params_vals = list(df_params['value'])

Tmax = 700
nums = Tmax*5
vecTime = np.linspace(0, Tmax, nums)

variable_names = ['p_LuxI', 'p_mtrC']
cond_init = [0.1]*len(variable_names)

simu = odeint(model_biosensor, cond_init, vecTime, args=(params_vals,))

fig, axes = plt.subplots(1,2, figsize=(9,6))
titulos = ["p_LuxI vs time", "p_mtrC vs time"]
for idx, name in enumerate(variable_names):
    axes[idx].set_title(fr"{titulos[idx]}")
    axes[idx].plot(vecTime, simu[:,idx])

    axes[idx].set_ylabel(fr"{name}")
    axes[idx].set_xlabel(r'Time (min)')

plt.tight_layout()
#plt.savefig('simu_ODE.jpeg')
plt.show()
plt.close()

# %%
