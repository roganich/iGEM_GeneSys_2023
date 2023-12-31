import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import os

main_path = os.getcwd()
parameters_path = 'parameters'
results_path = 'results'
run_path = 'run'

if os.path.exists(os.path.join(main_path,results_path)) == False:
    os.mkdir(os.path.join(main_path, results_path))

if os.path.exists(os.path.join(main_path, parameters_path)) == False:
    os.mkdir(os.path.join(main_path, parameters_path))

def model_biosensor(variables, t, params,microplastic):
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
    microNahR = (p_NahR*microplastic/(K+microplastic))

    
    dp_LuxIdt = sigma_pLuxI/gamma_rLuxI*(alpha_LuxI + beta_LuxI/(1+ (microNahR/K_LuxI)**-h_LuxI)) - gamma_pLuxI*p_LuxI

    AHL_sh = (S*K1*K2*sigma_AHL*p_LuxI)/(gamma_Ec + gamma_out*K1*((V_T-E*v_ec-S*v_sh)/E*v_ec) + gamma_sh*K2*K1*S*v_sh/E*v_ec)*(v_sh/v_ec)
    p_LuxR = S*sigma_pLuxR*sigma_rLuxR/(gamma_pLuxR*gamma_rLuxR)
    luxRAHL = (p_LuxR*AHL_sh)/((K_LuxRAHL*V_sh)+AHL_sh)

    dp_mtrCdt = S*(sigma_pmtrC/gamma_rmtrC)*(alpha_mtrC + (beta_mtrC)/(1+(luxRAHL/K_mtrC)**(-h_mtrC))) - gamma_mtrC*p_mtrC 

    dXdt = [dp_LuxIdt, dp_mtrCdt]

    return dXdt

df_params = pd.read_csv(os.path.join(main_path, parameters_path,'params_biosensor.csv'))

params_vals = list(df_params['value'])

Tmax = 700
nums = Tmax*5
vecTime = np.linspace(0, Tmax, nums)

variable_names = ['p_LuxI', 'p_mtrC']
cond_init = [0.1]*len(variable_names)

p_mtrC_valores = []
for micro in [1,100,1000,10000,100000,1000000,10000000,100000000]:

    simu = odeint(model_biosensor, cond_init, vecTime, args=(params_vals,micro))
    
    p_mtrC_valores.append(simu[:,1])

fig, axes = plt.subplots(1,2, figsize=(9,6))
colors = ['darkgreen', 'darkmagenta']
titulos = ["p_LuxI vs time", "p_mtrC vs time"]
for idx, name in enumerate(variable_names):
    axes[idx].set_title(fr"{titulos[idx]}")
    axes[idx].plot(vecTime, simu[:,idx], color=colors[idx])
    axes[idx].grid()

    axes[idx].set_ylabel(fr"{name}")
    axes[idx].set_xlabel(r'Time (min)')

plt.tight_layout()
plt.savefig(os.path.join(main_path, results_path, 'simulation_biosensor.jpg'), dpi=500)


plt.figure()
plt.plot(p_mtrC_valores[7], color='dodgerblue')
plt.xlabel('Microplastic concentration')
plt.ylabel('mtrC')
plt.grid()
plt.title('Sensibility of the cytochrome to the microplastic concentration')
plt.savefig(os.path.join(main_path,results_path,'sensibility_mtrC.jpg'), dpi=500)

