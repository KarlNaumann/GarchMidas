"""
MODULE FOR FORECASTING WITH THE BASIC GARCH-MIDAS-X
-------------------------------------------------------
"""

__author__ = "Karl Naumann-Woleske"
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Karl Naumann-Woleske"

import numpy as np
import pandas as pd

import data
import forecasting as fc

forecast = False
analysis = True

# Get data and "true" RV
ret, mvs = data.dataInput()
rv = data.realizedVariance()

# Obtain the results as estimated
res_coef = pd.read_csv('markovOutput/MS_Analysis_desired_coef.csv', index_col=0)
res_se = pd.read_csv('markovOutput/MS_Analysis_desired_se.csv', index_col=0)

# Select the identified models
for var in res_coef.index:
    ident = np.abs(res_coef.loc[var, 'theta']) / res_se.loc[var, 'theta']
    thresh = np.sqrt(np.log(ret.shape[0]))
    if ident < thresh: res_coef = res_coef.drop(var, axis=0)

horizons = [15, 75, 125]

# Obtain the forecasts for quareterly RV
print("-- Forecasting Days --")
if forecast:
    for macro in res_coef.index:
        print('---- {} ----'.format(macro))
        params = res_coef.loc[macro, 'mu':'p1'].tolist()
        print("Params: ", params)
        args = [12, ret, mvs[macro]]
        if res_coef.loc[macro, 'w1'] == 1:
            rest = True
        else:
            rest = False
        res = fc.forecastMSD(params,
                             args,
                             est='asym',
                             horizons=[15, 75, 125],
                             rest=rest,
                             v=True,
                             save='forecasting/' + macro + '_Day_2.csv')

# Results files
columns = ['MSPE', 'QLIKE', 'B0', 'B1', 'T_B0', 'T_B1', 'R2']
for15 = pd.DataFrame(np.zeros((res_coef.index.shape[0], len(columns))),
                     index=res_coef.index, columns=columns)
for75 = for15.copy(deep=True)
for125 = for15.copy(deep=True)
res_dict = {15: for15, 75: for75, 125: for125}

print('Analysis of daily point-forecasts')
for hi, h in enumerate(horizons):
    for macro in res_coef.index:
        res = pd.read_csv('forecasting/' + macro + '_Day.csv',
                          index_col=0)
        ix = pd.DatetimeIndex(res.index.tolist())
        res = pd.DataFrame(res.iloc[:, hi].values,
                           index=ix.round('d'), columns=[h])
        match = pd.concat([res, rv.loc[res.index.tolist()]], axis=1)
        match = match.dropna(0)
        actual = match.loc[:, 'IntraRV']
        forecast = match.loc[:, h]
        res_dict[h].loc[macro, 'MSPE'] = fc.mspe(forecast, actual)
        res_dict[h].loc[macro, 'QLIKE'] = fc.qlike(forecast, actual)
        p, t, r2 = fc.mincerZarnowitz(forecast, actual)
        res_dict[h].loc[macro, 'B0':'B1'] = p
        res_dict[h].loc[macro, 'T_B0':'T_B1'] = t
        res_dict[h].loc[macro, 'R2'] = r2
        res_dict[h].to_csv('forecasting/MS_' + str(h) + '_Day.csv')
    print(res_dict[h])
