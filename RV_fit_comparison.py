"""
Direct Model Comparison
-------------------------------------------------------
"""

__author__ = "Karl Naumann-Woleske"
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Karl Naumann-Woleske"

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
from matplotlib import pyplot as plt
from matplotlib import rc

import data
import basin_settings as bs
import forecasting as fc
import result_analysis as ra
import thesis_settings as tset
import ema_actual as ema

# Get data and "true" RV
ret, mvs = data.dataInput()
rv = data.realizedVariance()

gm_res = pd.read_csv('results/GM_Analysis_desired_coef.csv', index_col=0)
ms_res = pd.read_csv('markovOutput/MS_Analysis_desired_coef.csv',
                     index_col=0)
columns = ['MSPE', 'QLIKE', 'B0', 'B1', 'T_B0', 'T_B1', 'R2']
results_gm = pd.DataFrame(np.zeros((gm_res.index.shape[0], len(columns))),
                          index=gm_res.index, columns=columns)
results_ms = pd.DataFrame(np.zeros((ms_res.index.shape[0], len(columns))),
                          index=ms_res.index, columns=columns)

gm = False
ms = False
spreadPlot = True

rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)

if spreadPlot:
    macro = 'SPREAD'
    args = [tset.midasLags(), ret, mvs[macro]]
    g, tau = ra.recreate_asym(gm_res.loc[macro, :'w2'], args)
    spreadSingle = g.iloc[:, 0] * g.iloc[:, 1]
    spreadMS, _, _ = ra.recreate_ms(ms_res.loc[macro, :'p1'], args)
    ix = pd.DatetimeIndex(spreadSingle.index.tolist())
    res = pd.DataFrame(spreadSingle.values,
                       index=ix.round('d'), columns=['volSingle'])
    match1 = pd.concat([res, spreadMS.loc[res.index.tolist()]], axis=1)
    match1.columns = ['volSingle', 'volMS']
    match = pd.concat([match1, rv.loc[match1.index.tolist()]], axis=1)
    match = match.dropna(0)
    plt.figure(figsize=(7, 4))
    plt.plot(np.sqrt(252 * match['IntraRV']), linewidth=1.0,
             color='navy', label='Realised Variance')
    plt.plot(np.sqrt(252 * match['volSingle']), linewidth=2.0,
             color='firebrick', label='Single-Regime Spread')
    plt.plot(np.sqrt(252 * match['volMS']), linewidth=1.0, linestyle='--',
             color='gold', label='Two-Regime Spread')
    plt.ylim(bottom=0, top=100)
    plt.minorticks_on()
    plt.legend(fontsize=14)
    plt.legend()
    plt.savefig('figures/fig:spreadComp.png', bbox_inches='tight')
    plt.show()

if gm:
    for i, macro in enumerate(gm_res.index):
        print("Macro: ", macro)
        args = [tset.midasLags(), ret, mvs[macro]]
        g, tau = ra.recreate_asym(gm_res.loc[macro, :'w2'], args)
        vol = g.iloc[:, 0] * g.iloc[:, 1]
        ix = pd.DatetimeIndex(vol.index.tolist())
        res = pd.DataFrame(vol.values, index=ix.round('d'), columns=['vol'])
        match = pd.concat([res, rv.loc[res.index.tolist()]], axis=1)
        match = match.dropna(0)
        rv_true = match.loc[:, 'IntraRV']
        actual = match.loc[:, 'vol']
        results_gm.loc[macro, 'MSPE'] = fc.mspe(actual, rv_true)
        results_gm.loc[macro, 'QLIKE'] = fc.qlike(actual, rv_true)
        p, t, r2 = fc.mincerZarnowitz(actual, rv_true)
        results_gm.loc[macro, 'B0':'B1'] = p
        results_gm.loc[macro, 'T_B0':'T_B1'] = t
        results_gm.loc[macro, 'R2'] = r2
        results_gm.to_csv('results/GM_InSampleFit.csv')
        # plt.figure()
        # plt.plot(np.sqrt(252*rv_true),label='RV')
        # plt.plot(np.sqrt(252*actual),label='Model Results')
        # plt.legend()
        # plt.show()
        print(results_gm)

if ms:
    for i, macro in enumerate(ms_res.index):
        print("Macro: ", macro)
        args = [tset.midasLags(), ret, mvs[macro]]
        vol, _, _ = ra.recreate_ms(ms_res.loc[macro, :'p1'], args)
        ix = pd.DatetimeIndex(vol.index.tolist())
        res = pd.DataFrame(vol.values, index=ix.round('d'), columns=['vol'])
        match = pd.concat([res, rv.loc[res.index.tolist()]], axis=1)
        match = match.dropna(0)
        rv_true = match.loc[:, 'IntraRV']
        actual = match.loc[:, 'vol']
        results_ms.loc[macro, 'MSPE'] = fc.mspe(actual, rv_true)
        results_ms.loc[macro, 'QLIKE'] = fc.qlike(actual, rv_true)
        p, t, r2 = fc.mincerZarnowitz(actual, rv_true)
        results_ms.loc[macro, 'B0':'B1'] = p
        results_ms.loc[macro, 'T_B0':'T_B1'] = t
        results_ms.loc[macro, 'R2'] = r2
        results_ms.to_csv('markovOutput/MS_InSampleFit.csv')

        print(results_ms)
