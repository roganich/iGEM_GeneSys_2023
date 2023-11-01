import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pandas as pd
import os

def model_biosensor(variables, t, params):
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

def model_ToxAnti(variables, t, params):
    p_ccdB_e,p_ccdB_s,p_ccdA_e,p_ccdA_s,p_Trar,p_Bjal,p_Bjar,p_Esal,c_TrarEsal,c_BjarBjal,Ec,So = variables
    #p_ccdB_e,p_ccdB_s,p_ccdA_e,p_ccdA_s = variables

    alpha_ccdB_e,beta_ccdB_e,IPTG,K_ccdB_e,h_ccdB_e,gamma_rccdB_e,alpha_ccdB_s,beta_ccdB_s,K_ccdB_s,h_ccdB_s,gamma_rccdB_s,alpha_ccdA_e,beta_ccdA_e,K_ccdA_e,h_ccdA_e,gamma_rccdA_e,alpha_ccdA_s,beta_ccdA_s,K_ccdA_s,h_ccdA_s,gamma_rccdA_s,sigma_ccdB,gamma_pccdB,delta_ccdAccdB,sigma_Trar,Ar_Trar,gamma_rTrar,gamma_pTrar,sigma_Bjal,Ar_Bjal,gamma_rBjal,gamma_pBjal,sigma_Bjar,Ar_Bjar,gamma_rBjar,gamma_pBjar,sigma_Esal,Ar_Esal,gamma_rEsal,gamma_pEsal,s_BjarBjal,s_TrarEsal,gamma_pccdA,sigma_ccdA,sigma_Ec,sigma_So,p_So,K,p_Ec = params


    r_ccdB_e = (alpha_ccdB_e + beta_ccdB_e/(1 + (IPTG/K_ccdB_e)**-h_ccdB_e))/gamma_rccdB_e
    #Cambiar por tasas en Shawanella
    r_ccdB_s = (alpha_ccdB_s + beta_ccdB_s/(1 + (IPTG/K_ccdB_s)**-h_ccdB_s))/gamma_rccdB_s
    
    # proteína de ccdA para S. oneidensis y E. coli
    r_ccdA_e = (alpha_ccdA_e + beta_ccdA_e/(1 + (c_BjarBjal/K_ccdA_e)**-h_ccdA_e))/gamma_rccdA_e
    r_ccdA_s = (alpha_ccdA_s + beta_ccdA_s/(1 + (c_TrarEsal/K_ccdA_s)**-h_ccdA_s))/gamma_rccdA_s

    # proteína de ccdB para S. oneidensis y E. coli
    dp_ccdB_e = sigma_ccdB*r_ccdB_e - gamma_pccdB*p_ccdB_e - delta_ccdAccdB*p_ccdB_e*p_ccdA_e
    dp_ccdB_s = sigma_ccdB*r_ccdB_s - gamma_pccdB*p_ccdB_s - delta_ccdAccdB*p_ccdB_s*p_ccdA_s

    #dp_ccdB_e = sigma_ccdB*1000 - gamma_pccdB*p_ccdB_e - delta_ccdAccdB*p_ccdB_e*p_ccdA_e ##########
    #dp_ccdB_s = sigma_ccdB*2000 - gamma_pccdB*p_ccdB_s - delta_ccdAccdB*p_ccdB_e*p_ccdA_s ##########

    # proteína de ccdA para S. oneidensis y E. coli
    dp_ccdA_e = sigma_ccdA*r_ccdA_e - gamma_pccdA*p_ccdA_e - delta_ccdAccdB*p_ccdB_e*p_ccdA_e
    dp_ccdA_s = sigma_ccdA*r_ccdA_s - gamma_pccdA*p_ccdA_s - delta_ccdAccdB*p_ccdB_s*p_ccdA_s

    #dp_ccdA_e = sigma_ccdB*1500 - gamma_pccdB*p_ccdB_e - delta_ccdAccdB*p_ccdB_e*p_ccdA_e
    #dp_ccdA_s = sigma_ccdB*1700 - gamma_pccdB*p_ccdB_s - delta_ccdAccdB*p_ccdB_e*p_ccdA_s

    # proteína de Trar y Bjal de S. oneidensis
    dp_Trar = sigma_Trar*Ar_Trar/gamma_rTrar - gamma_pTrar*p_Trar
    dp_Bjal = sigma_Bjal*Ar_Bjal/gamma_rBjal - gamma_pBjal*p_Bjal

    # proteína de Bjar y Esal de E. coli
    dp_Bjar = sigma_Bjar*Ar_Bjar/gamma_rBjar - gamma_pBjar*p_Bjar
    dp_Esal = sigma_Esal*Ar_Esal/gamma_rEsal - gamma_pEsal*p_Esal

    # complejo Trar-Esal en S. oneidensis y complejo Bjar-Bjal en E. coli
    dc_BjarBjal = s_BjarBjal*p_Bjar*p_Bjal
    dc_TrarEsal = s_TrarEsal*p_Trar*p_Esal

    #Bacterias
    alpha_ec = np.exp(-0.05*p_ccdB_e) * sigma_Ec
    alpha_so = np.exp(-0.35*p_ccdB_s) * sigma_So
    dEc = alpha_ec*Ec*(1-(Ec+p_So*So)/K)
    dSo = alpha_so*So*(1-(So+p_Ec*Ec)/K)


    dXdt = [dp_ccdB_e, dp_ccdB_s, dp_ccdA_e, dp_ccdA_s, dp_Bjal, dp_Bjar, dp_Esal, dp_Trar, dc_TrarEsal, dc_BjarBjal, dEc, dSo]
    #dXdt = [dp_ccdB_e,dp_ccdB_s,dp_ccdA_e,dp_ccdA_s]
    return dXdt


main_path = os.getcwd()
parameters_path = 'parameters'
results_path = 'results'


if os.path.exists(os.path.join(main_path,results_path)) == False:
    os.mkdir(os.path.join(main_path, results_path))

if os.path.exists(os.path.join(main_path, parameters_path)) == False:
    os.mkdir(os.path.join(main_path, parameters_path))

df_params = pd.read_csv(os.path.join(main_path, parameters_path, 'params_biosensor.csv'))

df_params2 = pd.read_csv(os.path.join(main_path, parameters_path, 'params_toxin.csv'))

params_vals = list(df_params['value'])

params_vals2 = list(df_params2['value'])

Tmax = 150
nums = Tmax*5
vecTime = np.linspace(0, Tmax, nums)

#variable_names = ['p_LuxI', 'p_LuxR', 'p_mtrC']
variable_names = ['p_ccdB_e','p_ccdB_s','p_ccdA_e','p_ccdA_s','p_Trar','p_Bjal','p_Bjar','p_Esal','c_TrarEsal','c_BjarBjal','Ec','So']
#variable_names = ['p_ccdB_e','p_ccdB_s','p_ccdA_e','p_ccdA_s']

cond_init = [0.1]*len(variable_names)

#simu = odeint(model_biosensor, cond_init, vecTime, args=(params_vals,))
simu = odeint(model_ToxAnti, cond_init, vecTime, args=(params_vals2,))

#fig, axes = plt.subplots(1,len(variable_names), figsize=(10,10))

#for idx, name in enumerate(variable_names):
#    axes[idx].plot(vecTime, simu[:,idx])
#    axes[idx].grid()
#    axes[idx].set_ylabel(name)
#    axes[idx].set_xlabel('Time')


# Concentratino of toxin-antitoxin in E. coli
plt.plot(vecTime,simu[:,0], label='ccdB protein')
plt.plot(vecTime,simu[:,2], label='ccdA protein')
plt.title("Toxin-Antitoxin System in E. coli")
plt.grid()
plt.legend()
plt.savefig(os.path.join(main_path, results_path, 'simulation_toxin_E.jpeg'), dpi=500)
plt.close()

# Concentration of toxin-antitoxina in S. oneidensis
plt.plot(vecTime,simu[:,1], label='ccdB protein')
plt.plot(vecTime,simu[:,3], label='ccdA protein')
plt.title("Toxin-Antitoxin System in S. oneidensis")
plt.grid()
plt.legend()
plt.savefig(os.path.join(main_path, results_path, 'simulation_toxin_S.jpeg'), dpi=500)
plt.close()

#
plt.plot(vecTime,simu[:,10], label='E. coli')
plt.plot(vecTime,simu[:,11], label='S. oniendesis')
plt.title("Bacteria Population")
plt.grid()
plt.legend()
plt.savefig(os.path.join(main_path, results_path, 'simulation_bacteria_population.jpeg'), dpi=500)
plt.close()


