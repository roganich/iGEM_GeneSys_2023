import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def model(variables, t, params):
    p_NahR, C_mNahR, p_LuxI, AHL, p_LuxR, C_LuxRAHL, p_mtrC = variables
    sigma_pNahR, sigma_rNahR, gamma_rNahR, gamma_pNahR, beta_mNahR, K_mNahR, sigma_pLuxI, gamma_rLuxI, alpha_LuxI, beta_LuxI, K_LuxI, h_LuxI, gamma_pLuxI, eta_pLuxI, beta_LuxRAHL, K_LuxRAKL, sigma_pLuxR, sigma_rLuxR, gamma_rLuxR, gamma_pLuxR, sigma_pmtrC, gamma_rmtrC, alpha_mtrC, beta_mtrC, K_mtrC, h_LuxRAHL, gamma_mtrC = params

    dp_Nahrdt = sigma_pNahR*(sigma_rNahR/gamma_rNahR) - gamma_pNahR*p_NahR - beta_mNahR/(K_mNahR + p_NahR)
    dC_mNahRdt = beta_mNahR/(K_mNahR + p_NahR)
    dp_LuxIdt = sigma_pLuxI/gamma_rLuxI*(alpha_LuxI+ (beta_LuxI*C_mNahR/(K_LuxI+C_LuxRAHL**h_LuxI))) - gamma_pLuxI*p_LuxI - eta_pLuxI*p_LuxI
    dAHLdtdt = eta_pLuxI*p_LuxI - beta_LuxRAHL*AHL/(K_LuxRAKL + AHL)
    dp_LuxRdt = sigma_pLuxR*(sigma_rLuxR/gamma_rLuxR) - gamma_pLuxR*p_LuxR
    dC_luxRAHLdt = beta_LuxRAHL*AHL/(K_LuxRAKL + AHL)
    dp_mtrCdt = (sigma_pmtrC/gamma_rmtrC)*(alpha_mtrC + (beta_mtrC*C_LuxRAHL/(K_mtrC + C_LuxRAHL**h_LuxRAHL))) - gamma_mtrC*p_mtrC

    dXdt = [dp_Nahrdt, dC_mNahRdt, dp_LuxIdt, dAHLdtdt, dp_LuxRdt, dC_luxRAHLdt, dp_mtrCdt]

    return dXdt


