import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import os

def model(variables, t, params):
    p_NahR, C_mNahR, p_LuxI, AHL, p_LuxR, C_LuxRAHL, p_mtrC = variables
    sigma_pNahR, sigma_rNahR, gamma_rNahR, gamma_pNahR, beta_mNahR, K_mNahR, sigma_pLuxI, gamma_rLuxI, alpha_LuxI, beta_LuxI, K_LuxI, h_LuxI, gamma_pLuxI, eta_pLuxI, beta_LuxRAHL, K_LuxRAKL, sigma_pLuxR, sigma_rLuxR, gamma_rLuxR, gamma_pLuxR, sigma_pmtrC, gamma_rmtrC, alpha_mtrC, beta_mtrC, K_mtrC, h_mtrC, gamma_mtrC = params
    
    #r_NahR = sigma_rNahR/gamma_rNahR
    #r_LuxI = (1/gamma_rLuxI)*(alpha_LuxI+(beta_LuxI*C_mNahR/(K_LuxI+C_mNahR**h_LuxI)))
    #r_luxr = sigma_rLuxR/gamma_rLuxR
    #r_mtrC = (1/gamma_rmtrC)*(alpha_mtrC+(beta_mtrC*C_LuxRAHL/(h_mtrC + C_LuxRAHL**h_mtrC)))

    dp_Nahrdt = (sigma_pNahR*sigma_rNahR/gamma_rNahR) - gamma_pNahR*p_NahR
    dC_mNahRdt = 0
    dmdt = 0
    dp_LuxIdt = sigma_pLuxI/gamma_rLuxI*(alpha_LuxI+ (beta_LuxI*C_mNahR/(K_LuxI+C_LuxRAHL**h_LuxI))) - gamma_pLuxI*p_LuxI - eta_pLuxI*p_LuxI
    dAHLdt = 0
    dp_LuxRdt = sigma_pLuxR*(sigma_rLuxR/gamma_rLuxR) - gamma_pLuxR*p_LuxR
    dC_luxRAHLdt = 0
    dp_mtrCdt = (sigma_pmtrC/gamma_rmtrC)*(alpha_mtrC + (beta_mtrC*C_LuxRAHL/(K_mtrC + C_LuxRAHL**h_mtrC))) - gamma_mtrC*p_mtrC

    dXdt = [dp_Nahrdt, dC_mNahRdt, dp_LuxIdt, dAHLdt, dp_LuxRdt, dC_luxRAHLdt, dp_mtrCdt]

    return dXdt

main_path = os.getcwd()
parameters_path = 'parameters'
results_path = 'results'

df_params = pd.read_csv(os.path.join(main_path, parameters_path, 'params.csv'))

params_vals = list(df_params['value'])

Tmax = 150
nums = Tmax*5
vecTime = np.linspace(0, Tmax, nums)

params_names = ['p_Nahr', 'C_mNahR', 'p_LuxI', 'AHL', '_LuxR', 'C_luxRAHL', 'p_mtrC']
cond_init = [0.1]*len(params_names)

simu = odeint(model, cond_init, vecTime, args=(params_vals,))

fig, axes = plt.subplots(2,4, figsize=(10,10))

for idx, name in enumerate(params_names):
    if idx < 4:
        axes[0,idx].plot(vecTime, simu[:,idx])
        axes[0,idx].grid()
        axes[0,idx].set_ylabel(name)
    else:
        axes[1,idx-4].plot(vecTime, simu[:,idx])
        axes[1,idx-4].grid()
        axes[1,idx-4].set_ylabel(name)
        axes[1,idx-4].set_xlabel('Time') 

plt.savefig(os.path.join(main_path, results_path, 'simu_ODE.jpeg'))
plt.close()

