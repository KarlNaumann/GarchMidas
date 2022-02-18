"""
GARCH-MIDAS SIMPLE ANALYSIS & PLOTS
-------------------------------------------------------
This file tests the GARCH-MIDAS-X estimations of the GM-estimation
"""

__author__ = "Karl Naumann-Woleske"
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Karl Naumann-Woleske"

import numpy as np
import pandas as pd

import result_analysis as ra
import data
import thesis_settings as tset

# Settings
v = True  # verbose
p = True  # plotting
alt = False
load_se = True

names = tset.var_names()
figures = tset.gm_fig()
result_files = tset.gm_res()
output_files = tset.gm_analysis_csv()

if alt:
    alt = '_Att3.csv'
    for i in range(len(result_files)):
        result_files[i] = result_files[i][:-4] + alt
    for i in range(len(output_files)):
        output_files[i] = output_files[i][:-4] + alt[5:]
    for i in range(len(figures)):
        figures[i] = figures[i][:-4] + '.png'

# Obtain the results as estimated
res_rest = pd.read_csv(result_files[0], index_col=0)
res_unrest = pd.read_csv(result_files[1], index_col=0)
se_rest = pd.read_csv(result_files[2], index_col=0)
se_unrest = pd.read_csv(result_files[3], index_col=0)
se_rest_n = pd.read_csv(result_files[2][:-4] + '_n.csv', index_col=0)
se_unrest_n = pd.read_csv(result_files[3][:-4] + '_n.csv', index_col=0)
print(res_rest)
# Get the data
ret, mvs = data.dataInput()
K = tset.midasLags()
p_num = res_rest.shape[1] - 1

if not load_se:
    print('--- Standard Errors ---')
    for macro in mvs.columns:
        print('Var: ', macro)
        args = [K, ret, mvs[macro]]
        p_r = res_rest.loc[macro, 'mu':'w2']
        p_u = res_unrest.loc[macro, 'mu':'w2']
        se_rest_n.loc[macro, :] = ra.standardErrors(p_r, args, step=1e-4)
        se_unrest_n.loc[macro, :] = ra.standardErrors(p_u, args)
        if macro == 'RV':
            se_rest_n.loc[macro, :] = ra.standardErrors(p_r, args, step=1e-6)

    se_rest_n.to_csv('results/se_rest_n.csv')
    se_unrest_n.to_csv('results/se_unrest_n.csv')
    print(se_rest_n)
    print(se_unrest_n)

# t-statistics and p-values for all the 
t_st_rest, p_rest = ra.tstat_df(res_rest.loc[:, 'mu':'w2'], se_rest_n)
t_st_unrest, p_unrest = ra.tstat_df(res_unrest.loc[:, 'mu':'w2'], se_unrest_n)
p_rest.loc[:, 'w1'] = None
p_rest.to_csv(output_files[0])
t_st_rest.loc[:, 'w1'] = None
t_st_rest.to_csv(output_files[1])
p_unrest.to_csv(output_files[2])
t_st_unrest.to_csv(output_files[3])
# If verbose: display stats
if v:
    print("T-stat restricted p-values:\n", p_rest)
    print("STATS:\n", t_st_rest)
    print("T-stat unrestricted p-values:\n", p_unrest)
    print("STATS:\n", t_st_unrest)

# Likelihood ratio tests to determine restricted v unrestricted
lrt = ra.lrt_weight(res_rest.loc[:, 'Log-Likelihood'],
                    res_unrest.loc[:, 'Log-Likelihood'])
lrt.to_csv(output_files[4])
if v: print("Likelihood Ratio Test:\n", lrt)

# Results dataframe
results = ra.resultsFrame(res_rest, res_unrest)
print(se_rest_n)
print(se_unrest_n)
# Recreate volatility series and calculate BIC / VAR(X)
g_list = []
g_r_list = []
tau_list = []
tau_r_list = []
innov_list = []
innov_r_list = []
prv = results.loc['RV', 'mu':'w2'].values[0, :]
grv, _ = ra.recreate_asym(prv, [K, ret, mvs['RV']])
for num, var in enumerate(results.index):
    # Unrestricted case
    if np.mod(num, 2) == 1:
        g, tau = ra.recreate_asym(results.iloc[num, :-1], [K, ret, mvs[var]])
        g_list.append(g)
        tau_list.append(tau)
        vol = np.multiply(g.iloc[:, 0].values, g.iloc[:, 1].values)
        innovations = np.divide(ret - results.iloc[num, 0],
                                np.sqrt(vol[:, np.newaxis]))
        innov_list.append(pd.DataFrame(innovations, index=g.index))
        print("unrest {}".format(var))
        print("coeff: ", results.iloc[num, -7])
        print("se: ", se_unrest_n.loc[var, 'theta'])
        results.iloc[num, -1] = ra.identified(results.iloc[num, -7],
                                              se_unrest_n.loc[var, 'theta'],
                                              vol.shape[0])
    # Restricted case
    else:
        g, tau = ra.recreate_asym(results.iloc[num, :-1], [K, ret, mvs[var]])
        g_r_list.append(g)
        tau_r_list.append(tau)
        vol = np.multiply(g.iloc[:, 0].values, g.iloc[:, 1].values)
        innovations = np.divide(ret - results.iloc[num, 0],
                                np.sqrt(vol[:, np.newaxis]))
        innov_r_list.append(pd.DataFrame(innovations, index=g.index))
        print("rest {}".format(var))
        print("coeff: ", results.iloc[num, -7])
        print("se: ", se_rest_n.loc[var, 'theta'])
        results.iloc[num, -1] = ra.identified(results.iloc[num, -7],
                                              se_rest_n.loc[var, 'theta'],
                                              vol.shape[0])
    results.iloc[num, -3] = ra.bic(p_num, results.iloc[num, -4], g.shape[0])
    results.iloc[num, -2] = ra.varRatio(g)

if v: print(results)

results.to_csv(output_files[5])

# Determine the desired models based on restricted v unrestricted
des_g = []
des_ix = []

des_coef = ra.desired_df(lrt, res_rest, res_unrest, results.iloc[:, -1])
des_coef.to_csv(output_files[6])
des_se = ra.desired_df(lrt, se_rest, se_unrest, results.iloc[:, -1])
des_se.to_csv(output_files[7])
for i, reg in enumerate(lrt.index.tolist()):
    # If llf different enough - take the unrestricted
    if lrt.loc[reg, 'p-val'] < 0.1:
        des_ix.append(reg + '_u')
        des_g.append(g_list[i])
    else:
        des_ix.append(reg)
        des_g.append(g_r_list[i])

# Plot all volatilities
ra.plotAllAnnVol(g_r_list, names, block=False, save=figures[0])
ra.plotAllAnnVol(g_list, names, block=False, save=figures[1])
ra.plotAllAnnVol(des_g, names, block=False, save=figures[2])
# Plot all innovations 
ra.plotAllInnov(innov_r_list, names, block=False, save=figures[3])
ra.plotAllInnov(innov_list, names, block=False, save=figures[4])
# Plot all the weighting schemes
w1_r = res_rest.loc[:, 'w1'].tolist()
w2_r = res_rest.loc[:, 'w2'].tolist()
w1_u = res_unrest.loc[:, 'w1'].tolist()
w2_u = res_unrest.loc[:, 'w2'].tolist()
ra.plotAllWeights(w1_r, w2_r, w1_u, w2_u, K, names, block=False, save=figures[5])
