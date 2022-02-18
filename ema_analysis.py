import numpy as np
import pandas as pd

import result_analysis as ra
import latexExport as ltx
import data
import thesis_settings as tset

""" REGIME SWITCHING GARCH-MIDAS SIMPLE ANALYSIS & PLOTS """
""" This file tests the GARCH-MIDAS-X estimations of the GM-estimation"""

# Settings
v = True # verbose
p = False # plotting
testing = True
alt = False
# Load SE
load_se_rest = True
load_se_unrest = True

# Files 
names = tset.var_names()     
figures = tset.gm_fig()
result_files = tset.ms_results()
output_files = tset.ms_analysis_csv()
result_files[0] = 'markovResults/EMA_restricted_direct_X.csv'
# Results
res_rest    = pd.read_csv(result_files[0],index_col=0)
res_unrest  = pd.read_csv(result_files[1],index_col=0)
se_rest = pd.DataFrame(np.zeros((res_rest.shape[0],res_rest.shape[1]-1)),
                        index=res_rest.index,columns=res_rest.columns[:-1])
se_unrest = pd.DataFrame(np.zeros((res_rest.shape[0],res_rest.shape[1]-1)),
                        index=res_rest.index,columns=res_rest.columns[:-1])

# Data
ret,mvs = data.dataInput()
K = tset.midasLags()
p_num = res_rest.shape[1]-1

# Determine the standard errors via inverse Hessian (numerically approx.)
if load_se_rest: se_rest = pd.read_csv('markovOutput/se_rest.csv',
                                                        index_col=0)
else:
    print(' -  Calculating SE  - ')
    for macro in mvs.columns:
        if v: print('Variable: ',macro)
        args = [K,ret,mvs[macro]]
        p = res_rest.loc[macro,'mu':'p1']
        se_rest.loc[macro,:] = ra.standardErrorsMS(p,args,step=1e-5)
        # in case the precision isnt good enough i.e. errors due to 
        # identification issues
        if se_rest.loc[macro,:].isnull().any(): 
            alt = ra.standardErrorsMS(p,args,step=1e-7)
            for i,item in enumerate(se_rest.columns):
                if np.isnan(se_rest.loc[macro,item]):
                    se_rest.loc[macro,item] = alt[i]
        if se_rest.loc[macro,:].isnull().any(): 
            alt = ra.standardErrorsMS(p,args,step=1e-8)
            for i,item in enumerate(se_rest.columns):
                if np.isnan(se_rest.loc[macro,item]):
                    se_rest.loc[macro,item] = alt[i]
        if v: print('Restricted:\n',se_rest.loc[macro,:]) 
    se_rest.to_csv('markovOutput/se_rest_X.csv')
if v: print("Restricted SE:\n",se_rest)

if load_se_unrest: se_unrest = pd.read_csv('markovOutput/se_unrest.csv',
                                                        index_col=0)
else:
    print(' -  Calculating SE  - ')
    for macro in mvs.columns:
        if v: print('Variable: ',macro)
        args = [K,ret,mvs[macro]]
        p = res_unrest.loc[macro,'mu':'p1']
        se_unrest.loc[macro,:] = ra.standardErrorsMS(p,args,step=1e-5)
        # in case the precision isnt good enough i.e. errors due to 
        # identification issues
        if se_unrest.loc[macro,:].isnull().any(): 
            alt = ra.standardErrorsMS(p,args,step=1e-7)
            for i,item in enumerate(se_unrest.columns):
                if np.isnan(se_unrest.loc[macro,item]):
                    se_unrest.loc[macro,item] = alt[i]
        if se_unrest.loc[macro,:].isnull().any(): 
            alt = ra.standardErrorsMS(p,args,step=1e-8)
            for i,item in enumerate(se_unrest.columns):
                if np.isnan(se_unrest.loc[macro,item]):
                    se_unrest.loc[macro,item] = alt[i]
        if v: print('Unrestricted:\n',se_unrest.loc[macro,:])   
    se_unrest.to_csv('markovOutput/se_unrest.csv')
if v: print('Unrestricted SE:\n',se_unrest)
    
# t-statistics and p-values for all parameters
t_st_rest,p_rest = ra.tstat_df(res_rest.loc[:,'mu':'p1'],se_rest)
t_st_unrest,p_unrest = ra.tstat_df(res_unrest.loc[:,'mu':'p1'],se_unrest)
p_rest.loc[:,'w1'] = None
p_rest.to_csv(output_files[0])
t_st_rest.loc[:,'w1'] = None
t_st_rest.to_csv(output_files[1])
p_unrest.to_csv(output_files[2])
t_st_unrest.to_csv(output_files[3])
if v: 
    print("T-stat restricted p-values:\n",p_rest)
    print("STATS:\n",t_st_rest)
    print("T-stat unrestricted p-values:\n",p_unrest)
    print("STATS:\n",t_st_unrest)

# Likelihood ratio tests to determine restricted v unrestricted
lrt = ra.lrt_weight(res_rest.loc[:,'LLF'],
                    res_unrest.loc[:,'LLF'])
lrt.to_csv(output_files[4])
if v: print(lrt)

# Results dataframe
results = ra.resultsFrame(res_rest,res_unrest)

# Recreate volatility series and calculate BIC / VAR(X)
v_list,v_r_list = [],[]
tau_list,tau_r_list = [],[]
innov_list,innov_r_list = [],[]
p_list,p_r_list = [],[]
des_names1,des_names2 = [],[]
des_v,des_t,des_p,des_ix = [],[],[],[]
des_param_u,des_param_r,des_param_name = [],[],[]
grv,_,drv = ra.recreate_ms(res_rest.loc['RV','mu':'p1'],[K,ret,mvs['RV']])
grv = pd.DataFrame(np.hstack([np.array(grv),np.array(drv)]))
for num,var in enumerate(results.index):
    #Unrestricted case
    if np.mod(num,2) == 1: 
        v,p,daily_tau = ra.recreate_ms(results.iloc[num,:-4],
                                                        [K,ret,mvs[var]])
        v_list.append(v)
        tau_list.append(daily_tau)
        innovations = np.divide(ret-results.iloc[num,0],v)
        innov_list.append(pd.DataFrame(innovations,index=v.index)) 
        p_list.extend([p])
        results.iloc[num,-1] = ra.identified(results.iloc[num,6],
                                        se_unrest.loc[var,'theta'],v.shape[0])
        df = pd.DataFrame(np.hstack([np.array(v),np.array(daily_tau)]))
        if results.iloc[num,-1] and lrt.loc[var,'p-val']<0.1:
            des_v.append(v)
            des_names1.append(var+' Unrestricted')
            des_names2.append(var)
            des_t.append(daily_tau)
            des_p.append(p)

        if results.iloc[num,-1] or results.iloc[num-1,-1]:
            des_param_u.append(results.iloc[num,:-4])
            des_param_r.append(results.iloc[num-1,:-4])
            des_param_name.append(var)
            
    #Restricted case
    else: 
        v_r,p_r,daily_tau_r = ra.recreate_ms(results.iloc[num,:-4],
                                                        [K,ret,mvs[var]])
        v_r_list.append(v_r)
        tau_r_list.append(daily_tau_r)
        innovations = np.divide(ret-results.iloc[num,0],v_r)
        innov_r_list.append(pd.DataFrame(innovations,index=v_r.index))
        p_r_list.extend([p_r])
        results.iloc[num,-1] = ra.identified(results.iloc[num,6],
                                        se_rest.loc[var,'theta'],v_r.shape[0])
        df = pd.DataFrame(np.hstack([np.array(v_r),np.array(daily_tau_r)]))
        if results.iloc[num,-1] and lrt.loc[var,'p-val']>=0.1:
            des_v.append(v_r)
            des_names1.append(var)
            des_names2.append(var)
            des_t.append(daily_tau_r)
            des_p.append(p_r)
    results.iloc[num,-3] = ra.bic(p_num,results.iloc[num,-4],v_r.shape[0])
    results.iloc[num,-2] = ra.varRatio(df)
    
print(results)

results.to_csv(output_files[5])

# Determine the desired models based on restricted v unrestricted LRT

des_coef = ra.desired_df(lrt,res_rest,res_unrest,results.iloc[:,-1])
des_coef.to_csv(output_files[6])
des_se = ra.desired_df(lrt,se_rest,se_unrest,results.iloc[:,-1])
des_se.to_csv(output_files[7])

# Plot all volatilities
#ra.plotMSAnnVol(v_list,tau_list,p_list,names,block=False,
#                                        save='figures/fig:MS_unrestricted.png')
#ra.plotMSAnnVol(v_r_list,tau_r_list,p_r_list,names,block=False,
#                                        save='figures/fig:MS_restricted.png')
#ra.plotMSAnnVol(des_v,des_t,des_p,des_names2,block=False,
#                                        save='figures/fig:MS_des.png')

for i in range(len(des_v)):
    ra.plotMSAnnVol_s(des_v[i],des_t[i],des_p[i],des_names1[i],block=False,
                save='figures/fig:MS_{}_vol.png'.format(des_names2[i]))
for i in range(len(des_param_r)):    
    w1_r = des_param_r[i].iloc[-4]
    w2_r = des_param_r[i].iloc[-3]
    w1_u = des_param_u[i].iloc[-4]
    w2_u = des_param_u[i].iloc[-3]
    ra.plotAllWeights_s(w1_r,w2_r,w1_u,w2_u,K,des_param_name[i],block=False,
                save='figures/fig:MS_{}_w.png'.format(des_param_name[i]))
    
# Plot all innovations 
ra.plotAllInnov(innov_r_list,names,block=False,save=figures[3])
ra.plotAllInnov(innov_list,names,block=False,save=figures[4])

# Plot all the weighting schemes
w1_r = res_rest.loc[:,'w1'].tolist()
w2_r = res_rest.loc[:,'w2'].tolist()
w1_u = res_unrest.loc[:,'w1'].tolist()
w2_u = res_unrest.loc[:,'w2'].tolist()
ra.plotAllWeights(w1_r,w2_r,w1_u,w2_u,K,names,
                                    block=False,save='figures/fig:MS_weights')