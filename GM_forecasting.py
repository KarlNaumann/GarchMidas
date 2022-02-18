"""MODULE FOR FORECASTING WITH THE BASIC GARCH-MIDAS-X"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize,Bounds
from matplotlib import pyplot as plt

import data
import basin_settings as bs
import forecasting as fc
import result_analysis as ra
import thesis_settings as tset

forecast = False
quarterly_analysis = False
analysis = True

#Get data and "true" RV
ret,mvs = data.dataInput()
rv = data.realizedVariance()

# Obtain the results as estimated
res_coef = pd.read_csv('results/GM_Analysis_desired_coef.csv',index_col=0)
horizons = [15,75,125]

# Obtain the forecasts for quareterly RV
print("-- Forecasting Days --")
if forecast:
    for macro in res_coef.index[4:]:
        print('---- {} ----'.format(macro))
        params = res_coef.loc[macro,'mu':'w2'].tolist()
        print("Params: ",params)
        data = [12,ret,mvs[macro]]
        res = fc.forecastAsymD(params,
                    data,
                    est = 'asym',
                    horizons = [15,75,125],
                    rest=True,
                    v=True,
                    save='forecasting_rep/'+macro+'_Day.csv')

forecast_files = ['forecasting_rep/'+m+'_Q_2.csv' 
                        for m in res_coef.index.tolist() for h in horizons]

# Results files
columns = ['MSPE','QLIKE','B0','B1','T_B0','T_B1','R2']
for15 = pd.DataFrame(np.zeros((res_coef.index.shape[0],len(columns))),
          index=res_coef.index,columns=columns)
for75 = for15.copy(deep=True)
for125 = for15.copy(deep=True)
res_dict = {'15':for15,'75':for75,'125':for125}

res1Q = pd.DataFrame(np.zeros((res_coef.index.shape[0],len(columns))),
          index=res_coef.index,columns=columns)
res2Q = pd.DataFrame(np.zeros((res_coef.index.shape[0],len(columns))),
          index=res_coef.index,columns=columns)
res3Q = pd.DataFrame(np.zeros((res_coef.index.shape[0],len(columns))),
          index=res_coef.index,columns=columns)
res4Q = pd.DataFrame(np.zeros((res_coef.index.shape[0],len(columns))),
          index=res_coef.index,columns=columns)
resQD = {'1Q':res1Q,'2Q':res2Q,'3Q':res3Q,'4Q':res4Q}

print('Analysis of Quarterly RV')
if quarterly_analysis:
    for macro in res_coef.index:
        res = pd.read_csv('forecasting_rep/'+macro+'_Q_2.csv',
                            index_col=0)
        ix = pd.DatetimeIndex(res.index.tolist())
        res = pd.DataFrame(res.values, 
                                index=ix.round('d'),columns=res.columns)
        qrv = fc.quarterlyRV(rv,res.index)
        print(qrv)
        res = res.iloc[2:,:]
        qrv = qrv.iloc[2:,:]
        #plt.figure()
        #plt.plot(res,label='res')
        #plt.plot(qrv,label='qrv')
        #plt.title(macro)
        #plt.legend()
        #plt.show()
        for q in res.columns:
            resQD[q].loc[macro,'MSPE'] = fc.mspe(res.loc[:,q],qrv)
            resQD[q].loc[macro,'QLIKE'] = fc.qlike(res.loc[:,q],qrv)
            p,t,r2 = fc.mincerZarnowitz(res.loc[:,q],qrv)
            resQD[q].loc[macro,'B0':'B1'] = p
            resQD[q].loc[macro,'T_B0':'T_B1'] = t
            resQD[q].loc[macro,'R2'] = r2
            print("Macro: {}, Quarter: {}".format(macro,q))
            print("QLIKE:",resQD[q].loc[macro,'MSPE'])
            print("MSPE:",resQD[q].loc[macro,'QLIKE'])
            print("MZ:",p,t)
    
    print(res1Q)
    print(res2Q)
    print(res3Q)
    print(res4Q)

print('Analysis for daily point-forecasts')
if analysis:
    for h in horizons:  
        print("horizon: {}".format(h))       
        for macro in res_coef.index:
            res = pd.read_csv('forecasting_rep/'+macro+'_Day.csv',
                                index_col=0)
            ix = pd.DatetimeIndex(res.index.tolist())
            res = pd.DataFrame(res.loc[:,h].values, 
                        index=ix.round('d'),columns=[h])
            match = pd.concat([res,rv.loc[res.index.tolist()]],axis=1)
            match = match.dropna(0)
            actual = match.loc[:,'IntraRV']
            forecast = match.loc[:,h]
            res_dict[h].loc[macro,'MSPE'] = fc.mspe(forecast,actual)
            res_dict[h].loc[macro,'QLIKE'] = fc.qlike(forecast,actual)
            p,t,r2 = fc.mincerZarnowitz(forecast,actual)
            res_dict[h].loc[macro,'B0':'B1'] = p
            res_dict[h].loc[macro,'T_B0':'T_B1'] = t
            res_dict[h].loc[macro,'R2'] = r2
            res_dict[h].to_csv('forecasting/GM_'+str(h)+'_Day.csv')
        print(res_dict[h])