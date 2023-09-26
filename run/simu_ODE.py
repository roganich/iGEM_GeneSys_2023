import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import os

def model(variables, t, params):
    p_LuxI, p_LuxR, p_mtrC = variables
    sigma_pNahR, sigma_rNahR, gamma_rNahR, gamma_pNahR, m, K, K1, K2, gamma_Ec, gamma_out, gamma_sh, sigma_pLuxI, gamma_rLuxI, alpha_LuxI, beta_LuxI, K_LuxI, h_LuxI, gamma_pLuxI, beta_LuxRAHL, K_LuxRAHL, sigma_pLuxR, sigma_rLuxR, gamma_rLuxR, gamma_pLuxR, sigma_pmtrC, gamma_rmtrC, alpha_mtrC, beta_mtrC, K_mtrC, h_mtrC, gamma_mtrC = params
    
    p_NahR = sigma_pNahR*sigma_rNahR/(gamma_rNahR*gamma_pNahR)
    microNahR = p_NahR*m/(K+m)
    AHL_sh = sigma_pLuxI*K1*K2/(gamma_Ec + gamma_out*K1 + gamma_sh*K1*K2)
    luxRAHL = beta_LuxRAHL*AHL_sh/(K_LuxRAHL+AHL_sh)

    dp_LuxIdt = sigma_pLuxI/gamma_rLuxI*(alpha_LuxI + beta_LuxI/(1+ (microNahR/K_LuxI)**-h_LuxI)) - gamma_pLuxI*p_LuxI
    dp_LuxRdt =  sigma_pLuxR*sigma_rLuxR/gamma_rLuxR - gamma_pLuxR*p_LuxR
    dp_mtrCdt = sigma_pmtrC/gamma_rmtrC * (alpha_mtrC + beta_mtrC/(1+(luxRAHL/K_mtrC)**(-h_mtrC))) - gamma_mtrC*p_mtrC

    #dp_Nahrdt = (sigma_pNahR*sigma_rNahR/gamma_rNahR) - gamma_pNahR*p_NahR
    #dp_LuxIdt = sigma_pLuxI/gamma_rLuxI*(alpha_LuxI+ (beta_LuxI*C_mNahR/(K_LuxI+C_LuxRAHL**h_LuxI))) - gamma_pLuxI*p_LuxI - eta_pLuxI*p_LuxI
    #dp_LuxRdt = sigma_pLuxR*(sigma_rLuxR/gamma_rLuxR) - gamma_pLuxR*p_LuxR
    #dp_mtrCdt = (sigma_pmtrC/gamma_rmtrC)*(alpha_mtrC + (beta_mtrC*C_LuxRAHL/(K_mtrC + C_LuxRAHL**h_mtrC))) - gamma_mtrC*p_mtrC

    dXdt = [dp_LuxIdt, dp_LuxRdt, dp_mtrCdt]

    return dXdt

main_path = os.getcwd()
parameters_path = 'parameters'
results_path = 'results'

df_params = pd.read_csv(os.path.join(main_path, parameters_path, 'params.csv'))

params_vals = list(df_params['value'])

Tmax = 150
nums = Tmax*5
vecTime = np.linspace(0, Tmax, nums)

variable_names = ['p_LuxI', 'p_LuxR', 'p_mtrC']
cond_init = [0.1]*len(variable_names)

simu = odeint(model, cond_init, vecTime, args=(params_vals,))

fig, axes = plt.subplots(1,3, figsize=(10,10))

for idx, name in enumerate(variable_names):
    axes[idx].plot(vecTime, simu[:,idx])
    axes[idx].grid()
    axes[idx].set_ylabel(name)
    axes[idx].set_xlabel('Time')

plt.tight_layout()
plt.savefig(os.path.join(main_path, results_path, 'simu_ODE.jpeg'))
plt.close()


