import numpy as np
import matplotlib.pyplot as plt
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

df_params = pd.read_csv(os.path.join(main_path, parameters_path,'params_biosensor.csv'))
params_vals = list(df_params['value'])
sigma_pNahR, sigma_rNahR, gamma_rNahR, gamma_pNahR, m, K, K1, K2, gamma_Ec, gamma_out, gamma_sh, sigma_pLuxI, gamma_rLuxI, alpha_LuxI, beta_LuxI, K_LuxI, h_LuxI, gamma_pLuxI, beta_LuxRAHL, K_LuxRAHL, sigma_pLuxR, sigma_rLuxR, gamma_rLuxR, gamma_pLuxR, sigma_pmtrC, gamma_rmtrC, alpha_mtrC, beta_mtrC, K_mtrC, h_mtrC, gamma_mtrC, V_T, S, E, sigma_AHL = params_vals 

v_sh = np.pi*(0.55**2)*2.5
v_ec = np.pi*(0.5**2)*1.5 
V_sh = S*v_sh

V_ec = E*v_ec

#V_out = V_T - V_ec - V_sh
def modificacion_estado_estacionario(microplastic):
    p_NahR = (sigma_pNahR*sigma_rNahR)/(gamma_rNahR*gamma_pNahR)
    microNahR = (p_NahR*microplastic/(K+microplastic))

    p_LuxI = (sigma_pLuxI/gamma_rLuxI*(alpha_LuxI + beta_LuxI/(1+ (microNahR/(K_LuxI))**-h_LuxI)))/gamma_pLuxI

    AHL_sh = (S*K1*K2*sigma_AHL*p_LuxI)/(gamma_Ec + gamma_out*K1*((V_T-E*v_ec-S*v_sh)/E*v_ec) + gamma_sh*K2*K1*S*v_sh/E*v_ec)*(v_sh/v_ec)
    p_LuxR = S*sigma_pLuxR*sigma_rLuxR/(gamma_pLuxR*gamma_rLuxR)
    luxRAHL = (p_LuxR*AHL_sh)/((K_LuxRAHL*V_sh*(1e-18))+AHL_sh)
    p_mtrC = (S*(sigma_pmtrC/gamma_rmtrC)*(alpha_mtrC + (beta_mtrC)/(1+(luxRAHL/(S*K_mtrC))**(-h_mtrC))))/gamma_mtrC

    return [p_NahR, microNahR, p_LuxI, AHL_sh, p_LuxR, luxRAHL, p_mtrC]

valores_totales_p_NahR = []
valores_totales_microNahR = []

valores_totales_p_LuxI = []
valores_totales_AHL_sh = []
valores_totales_p_LuxR = []
valores_totales_luxRAHL = []
valores_totales_p_mtrC = []

valores_micro  = np.logspace(19.5,22.5, num = 1000)

for micro in valores_micro:
    resultados = modificacion_estado_estacionario(micro)
    valores_totales_p_NahR.append(resultados[0])
    valores_totales_microNahR.append(resultados[1])

    valores_totales_p_LuxI.append(resultados[2])
    valores_totales_AHL_sh.append(resultados[3])
    valores_totales_p_LuxR.append(resultados[4])
    valores_totales_luxRAHL.append(resultados[5])
    valores_totales_p_mtrC.append(resultados[6])

plt.figure(figsize=(8,5))
plt.title(r"P_mtrc vs [m]", fontsize = 16)
plt.xlabel(r"Microplastic concentration $\frac{[m]}{m^3}$", fontsize = 16)
plt.ylabel(r"P_mtrc", fontsize = 16)
plt.ticklabel_format(style="sci")
plt.plot(valores_micro[0:-1], valores_totales_p_mtrC[0:-1], label = "")
plt.xscale("log")
plt.yscale("log")
plt.axvline(x = 1.8066e20, color = "green", label = r"K", linestyle = "--", alpha = 0.5)
plt.axvline(x = 2.8e20, color = "red", label = r"[m] critic value", linestyle = "--", alpha = 0.5)
plt.axhline(y = valores_totales_p_mtrC[-1], color = "black", linestyle = "--", alpha = 0.5, label = r"Stationary Value")
plt.legend(fontsize = 14)