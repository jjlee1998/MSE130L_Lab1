import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import gamma
from lmfit import minimize, Parameters
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# import data as pandas DataGrames:

gd_df = pd.read_csv('./output/grain_densities.csv')
ii_10C_df = pd.read_csv('./output/ii_10C.csv')
ii_15C_df = pd.read_csv('./output/ii_15C.csv')
ii_20C_df = pd.read_csv('./output/ii_20C.csv')
ii_25C_df = pd.read_csv('./output/ii_25C.csv')
ii_30C_df = pd.read_csv('./output/ii_30C.csv')
ii_35C_df = pd.read_csv('./output/ii_35C.csv')

# data preprocessing:
#   - isolate fitting regions
#   - add time column (2.87 frames per second)
#   - add alpha column, scaled into the range [0, 1]
#   - (0 set from min either before or after isolation)

min_image_id = [2258, 887, 283, 215, 147, 45]
absolute_zero = [True, True, True, False, True, False]
ii_df_tmp = [ii_10C_df, ii_15C_df, ii_20C_df, ii_25C_df, ii_30C_df, ii_35C_df]
ii_df_all = []

for n in range(6):
    ii_df_n = ii_df_tmp[n].copy()
    min_ii = ii_df_n['integrated_intensity'].min()
    ii_df_n = ii_df_n[ii_df_n['image_id'] > min_image_id[n]]
    if not absolute_zero[n]:
        min_ii = ii_df_n['integrated_intensity'].min()
    max_ii = ii_df_n['integrated_intensity'].max()
    ii_df_n['time_s'] = (ii_df_n['image_id'] - 1) / 2.87
    ii_df_n['alpha'] = (ii_df_n['integrated_intensity']-min_ii)/(max_ii-min_ii)
    ii_df_all.append(ii_df_n)

# fit each integrated intensity curve to the Becker-Doering result:

def bd_alpha(params, t):
    t0 = params['t0']
    nv2 = params['nv2']
    alpha = 1 - np.exp(-np.pi/3*nv2*(t-t0)**3)
    return np.where(t > t0, alpha, 0)

def bd_alpha_resid(params, t, alpha):
    alpha_pred = bd_alpha(params, t)
    return alpha_pred - alpha

fitted_params = []

for ii_df in ii_df_all:

    t = ii_df['time_s'].values
    alpha = ii_df['alpha'].values
    ii = ii_df['integrated_intensity'].values
    params = Parameters()
    params.add('t0', value=t[0], min=0, vary=True)
    params.add('nv2', value=1, min=0.0, vary=True)
    result = minimize(bd_alpha_resid, params, args=(t, alpha))
    params_out = result.params
    fitted_params.append(params_out)

    #ii_df.plot.scatter(x='time_s', y='alpha')
    #plt.plot(t, bd_alpha(params_out, t))
    #plt.show()

# merge Nv^2 (1/s^3) results with grain densities, add absolute temperature,
# then calculate N/v (1/m^3), and calculate N (1/s*m^2) and v (m/s):

nv2 = np.asarray([params['nv2'] for params in fitted_params])
nv2 = np.insert(nv2, 0, np.nan)
gd_df['Nv^2'] = nv2

nv2_se = np.asarray([params['nv2'].stderr for params in fitted_params])
nv2_se = np.insert(nv2_se, 0, np.nan)
gd_df['Nv^2_se'] = nv2_se

gd_df['T_K'] = gd_df['T'] + 273.15
gd_df['grains/m^2'] = gd_df['grain_density_um^-2'] * 1e12
n_v = (gd_df['grains/m^2'] * (np.pi/3)**(1/3) / gamma(4/3))**(3/2)
gd_df['N/v'] = n_v
gd_df['N'] =  (gd_df['Nv^2'] * gd_df['N/v']**2)**(1/3)
gd_df['v'] = (gd_df['Nv^2'] / gd_df['N/v'])**(1/3)
print()
print(gd_df)

# fit N and v to Arrhenius to determine energy barriers (eV):

kb = 8.617e-5 # eV/K
omega = 35.031e-30 # 35.031 cubic angstroms per atom, materials project
#lattice_param = np.power(omega, 1/3) # approximate as cubic
lattice_param = 4.512e-10 #4.512 angstrom, materials project
temp_k = gd_df['T_K'].values[1:].reshape(-1, 1)
n = gd_df['N'].values[1:]
v = gd_df['v'].values[1:]

ols_n_res = sm.OLS(np.log(n), sm.add_constant(1/temp_k)).fit()
ols_v_res = sm.OLS(np.log(v), sm.add_constant(1/temp_k)).fit()
n_params = ols_n_res.params
n_se = ols_n_res.bse
v_params = ols_v_res.params
v_se = ols_v_res.bse
print()
print(f'n-regression: R^2 = {ols_n_res.rsquared}')
print(f'v-regression: R^2 = {ols_v_res.rsquared}')

#print(ols_n_res.summary())

delta_G_nucleation = n_params[1] * -kb
delta_G_nucleation_se = n_se[1] * kb
delta_E_attach = v_params[1] * -kb
delta_E_attach_se = v_se[1] * kb
n_0 = np.exp(n_params[0])
n_0_se = n_0 * n_se[0]
v_0 = np.exp(v_params[0])
v_0_se = v_0 * v_se[0]
attempt_freq = v_0 / lattice_param
attempt_freq_se = v_0_se / lattice_param

#'''
print()
print(f'total free energy of nucleation: {delta_G_nucleation} eV/at')
print(f'\t\tsigma: {delta_G_nucleation_se}')
print(f'free energy of attachment: {delta_E_attach} eV/at')
print(f'\t\tsigma: {delta_E_attach_se}')
print(f'nuclation rate prefactor: {n_0} nuclei*s^-1*m^-2')
print(f'\t\tsigma: {n_0_se}')
print(f'growth velocity prefactor: {v_0} m/s')
print(f'\t\tsigma: {v_0_se}')
print(f'attempt frequency: {attempt_freq} 1/s*at')
print(f'\t\tsigma: {attempt_freq_se}')
print()
#'''

# determine free energy of critical nucleus:

delta_G_ic = delta_G_nucleation - delta_E_attach
delta_G_ic_se = np.sqrt(delta_G_nucleation_se**2 + delta_E_attach**2)

# assume avg gamma_s is 0.11 J/m^2...
# assume spherical critical nucleus to calculate ic, g(c->a):

gamma_s = 0.11 #J/m^2 << play around with this!
gamma = gamma_s * (3*omega)**(2/3) * (4*np.pi)**(1/3)
gamma_ev = gamma / 1.602e-19
g_c_a = np.sqrt(4*gamma_ev**3/27/delta_G_ic)
ic = (2*gamma_ev/3/g_c_a)

# print results
print(f'free energy per atom of critical nucleus: {delta_G_ic} eV') # eV
print(f'\t\tsigma: {delta_G_ic_se}')
print(f'gamma: {gamma_ev} eV/at^(2/3)')
print(f'g(c->a): {g_c_a} eV/at')
print(f'ic: {ic} atoms')
print()
