import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from lmfit import minimize, Parameters

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

# merge Nv^2 (nv2) results with grain densities and add absolute temperature:
nv2 = np.asarray([params['nv2'] for params in fitted_params])
nv2 = np.insert(nv2, 0, np.nan)
gd_df['nv2'] = nv2
gd_df['T_K'] = gd_df['T'] + 273.15

# fit density model to estimate N/v (denoted n_v):
