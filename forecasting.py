import numpy as np
import pandas as pd
from scipy.optimize import minimize,SR1,basinhopping
import statsmodels.api as sm
from dateutil import relativedelta as rd
from datetime import datetime

import basin_settings as bs
import likelihood_functions as llf
import data
import thesis_settings as tset
import ema_actual as ema

"""Methods for the forecasting models
    The series is first recreated, then a direct horizon-day forecast is
    computed, these are then saved"""

def horizonForecast(p,f_g_1,tau,horizon,est):
    """Select the correct forecast formula based on model type"""
    if est == 'gm':
        return gm_h(p,f_g_1,tau,horizon)
    elif est == 'asym':
        return asym_h(p,f_g_1,tau,horizon)
    elif est == 'arch':
        return arch_h(p,ret,tau,arg)
    elif est == 'arch_asym':
        return arch_asym_h(p,ret,tau,arg)
    elif est == 'msgm':
        return ms_gm_h(self, arg)
    else: print("No valid input")

def estimation(x0,data,est,rest=True,disp=False,full=False):
    """Local estimation algorithm for the forecast subsample. It is initialised
        at the full-sample estimates to reduce estimation time"""
    
    if rest: bnds,_ = bs.bounds_Simple()
    else: _,bnds = bs.bounds_Simple()
    if full is False:
        if est == 'gm':
            results = minimize(
                llf.simpleGM, x0, args=data,
                jac = '2-point',
                bounds = bnds, options={'disp':disp})
        elif est == 'asym':
            rest,unrest = bs.bounds_Simple_asym()
            if rest: bnds = rest
            else: bnds = unrest
            results = minimize(
                llf.simpleGM_asym, x0, args=data,
                jac = '2-point',
                bounds = bnds,  options={'disp':disp})
        elif est == 'arch':
            results = minimize(
                llf.simpleGM_ARCH, x0, args=data,
                jac = '2-point',
                bounds = bnds, options={'disp':disp})
        elif est == 'arch_asym':
            rest,unrest = bs.bounds_Simple_asym()
            if rest: bnds = rest
            else: bnds = unrest
            results = minimize(
                llf.simpleGM_ARCH_asym, x0, args=data,
                jac = '2-point',
                bounds = bnds, options={'disp':disp})
        elif est == 'msgm':
            rest,unrest = bs.bounds_Switch()
            if rest: bnds = rest
            else: bnds = unrest
             #TBD
        else: print("please specify underlying model")
    if full:
        if est == 'asym':
            if x0[-2]==1.0:
                bnds,_ = bs.bounds_Simple_asym()
                step = bs.StepAsym(rest=True,max=2.5)
            else: 
                _,bnds = bs.bounds_Simple_asym()
                step = bs.StepAsym(rest=False,max=2.5)    
            
            results = basinhopping(llf.simpleGM_asym,x0,\
                                minimizer_kwargs=bs.min_arg(data,bnds),\
                                niter=6,
                                take_step=step,
                                callback=bs.print_fun)
    return results.x

def findPersistence(est_param,est='asym'):
    if est == 'asym':
        a = est_param[1]
        b = est_param[2]
        gamma = est_param[3]
        return a + b + (gamma/2)

def forecastingQ(params,data,horizon,num_pred,est,rest=True,v=False,save=''):
    """Method replicating the forecasting approach of Conrad & Loch: training
    sample ranging from 1973Q1 - 1998Q4 - First 4qrtr forecast is for 2000Q1
    """
    K,ret,macro = data[0],data[1],data[2]
    # Explicity use numerical locations as dates are non-consecutive
    samp_start = ret.index[0]
    samp_end = ret.index[-1]
    est_end = ret.index[ret.index.get_loc('1998-12-31',method='bfill')]
    est_span = ret.index[0:ret.index.get_loc('1998-12-31',method='bfill')+1]
    
    # Todo:
    # estimate
    # get starting vals for 1q,2q,3q,4q
    # get RV estimate from formula for 1q,2q,3q,4q
    # Do this recursively????
    # Return this RV estimate
    
    f_samp_end = ret.index[-2 - horizon]
    f_samp_start = ret.index[-1 - num_pred - horizon]
    f_samp_span = ret.index[-1 - num_pred - horizon:-1 - horizon]
    if v: print("Samp start & end: ",samp_start,samp_end)
    if v: print("F Samp start & end: ",f_samp_start,f_samp_end)
    end = ret.index.get_loc('1998-12-31',method='bfill')+1
    ix = [end+(1+i)*63 for i in range(44)]
    f_res = pd.DataFrame(np.zeros((44,4)),
                        index=ret.index[ix],columns=['1Q','2Q','3Q','4Q'])
    # Make sure we start at 2000 for the equal sample
    f_res = f_res.iloc[3:,:]
    
    
    for i,dat in enumerate(f_res.index.tolist()):
        print('\n# - - Iteration: {} - - Forecast: {} - - #'.format(i,dat))
        # move the training start and end in quarters aka 63 days
        start = i*63
        end = ret.index.get_loc('1998-12-31',method='bfill')+1+(i*63)
        span = ret.index[start:end]
        
        print("Estimation span: ",span[0]," to ",span[-1])
        
        # log-likelihood function only selects data in range of returns
        est_param = estimation(params,[K,ret.loc[span.tolist()],macro],
                                        est=est,rest=rest,disp=False)
        if v: print("Est param: ",est_param)
        
        # recreate the g and tau series from the estimated parameters
        span = ret.index[start:end+1]
        args = [K,ret.loc[span.tolist()],macro]
        g,tau_l = recreateSimple(est_param,args,est=est)
        
        # fix the tau for all other quarters at the earliest option
        curr = ret.index[end].to_period('Q').to_timestamp()
        curr = tau_l.index.get_loc(curr,method='pad')
        fix_tau = tau_l.iloc[curr+1,0]
        tau_1q = tau_l.iloc[curr,0]
        
        # results frame = this is the end of the quarter for RV to be estimated
        est_target = ret.index.get_loc(dat)
        
        # approximate the starting dates of quarters for the target
        f_1q = est_target - 63
        f_2q = est_target - 126
        f_3q = est_target - 189
        f_4q = est_target - 252
        
        # 4q ahead forecast is earliest start - for this it is given
        f_4q_1 = g.iloc[-1,0]
        # estimate other g1,1 from f_4q_1
        f_3q_1 = horizonForecast(est_param,f_4q_1,fix_tau,63,est)
        f_2q_1 = horizonForecast(est_param,f_4q_1,fix_tau,126,est)
        f_1q_1 = horizonForecast(est_param,f_4q_1,fix_tau,189,est)
        
        # Forecasts for XQ ahead are all RV up to then minues previous Q
        rv_1q = asym_RV(est_param,f_1q_1,tau_1q,63)
        rv_2q = asym_RV(est_param,f_2q_1,fix_tau,126)
        rv_3q = asym_RV(est_param,f_3q_1,fix_tau,189)
        rv_4q = asym_RV(est_param,f_4q_1,fix_tau,252)
        # Save the forecasts
        f_res.loc[dat,'1Q'] = rv_1q
        f_res.loc[dat,'2Q'] = rv_2q-rv_1q
        f_res.loc[dat,'3Q'] = rv_3q-rv_2q
        f_res.loc[dat,'4Q'] = rv_4q-rv_3q
        
        if v:print("Forecast:",f_res.loc[dat,:])
    if save: f_res.to_csv(save)
    
    return f_res
  
def forecastAsymD(params,data,est,horizons=[15],rest=True,v=False,save=''):
    K,ret,macro = data[0],data[1],data[2]
    
    total_quarters = ret.index.to_period('Q').to_timestamp().unique()
    first_f = ret.index.get_loc('2000-01-01',method='bfill')
    all_q = ret.index[first_f:].to_period('Q').to_timestamp().unique()
    
    horizons = [15,75,125]
    f_res = pd.DataFrame(np.zeros((ret.index[first_f:].shape[0],len(horizons))),
                        index=ret.index[first_f:],columns=horizons)
    # Explicity use numerical locations as dates are non-consecutive
    samp_start = ret.index[0]
    est_end = ret.index[first_f - max(horizons)]
    est_span = ret.index[0:first_f - max(horizons)]

    # log-likelihood function only selects data in range of returns
    est_param = estimation(params,[K,ret.loc[est_span.tolist()],macro],
                                    est=est,rest=rest,disp=False,full=True)
    persistence = findPersistence(est_param,est=est)
    if v: 
        print("Original param: ",params)
        print("Est param: ",est_param)
    
    for h,hor in enumerate(f_res.columns):
        for i,date in enumerate(f_res.index.tolist()):
            est_target = ret.index.get_loc(date,method='bfill') # Targeted date
            end = ret.index.get_loc(date,method='pad')-hor
            end_q = total_quarters.get_loc(ret.index[end],method='pad')
            span = ret.index[:end+2] # include one-step g as it is predetermined
            args = [K,ret.loc[span.tolist()],macro]
            # Recreate the sample with data until forecast start point
            g,tau_l = recreateSimple(est_param,args,est=est)
            
            if end_q != total_quarters.get_loc(date,method='pad'):
                # Determine tau by current date + 1 period of tau
                tau_choice = tau_l.index.get_loc(total_quarters[end_q+1],
                                                            method='pad')
            else:
                # We remain in the same quarter - thus use the current tau
                tau_choice = tau_l.index.get_loc(total_quarters[end_q],
                                                            method='pad')
            fix_tau = tau_l.iloc[tau_choice].values
            g_1 = g.iloc[-1,0]
            f_res.iloc[i,h] = horizonForecast(est_param,g_1,fix_tau,hor,est)
        if v:print("Forecast:",f_res.loc[date,:])
    if save: f_res.to_csv(save)
    
    return f_res

def forecastMSD(params,args,est,horizons=[15],rest=True,v=False,save=''):
    K,ret,macro = args[0],args[1],args[2]
    
    total_quarters = ret.index.to_period('Q').to_timestamp().unique()
    first_f = ret.index.get_loc('2000-01-01',method='bfill')
    all_q = ret.index[first_f:].to_period('Q').to_timestamp().unique()
    
    horizons = [15,75,125]
    f_res = pd.DataFrame(np.zeros((ret.index[first_f:].shape[0],len(horizons))),
                        index=ret.index[first_f:],columns=horizons)
    # Explicity use numerical locations as dates are non-consecutive
    samp_start = ret.index[0]
    est_end = ret.index[first_f - max(horizons)]
    est_span = ret.index[0:first_f - max(horizons)]
    
    args_new = [K,ret[:first_f],macro]
    
    if rest: bnds,_ = ema.boundsUncond()
    else: _,bnds = ema.boundsUncond()
    # log-likelihood function only selects data in range of returns
    result = minimize(ema.llfHaasUncondDirect,params,
                                    args = args_new, 
                                    bounds = bnds, 
                                    jac = '2-point',
                                    options={'disp':2,
                                                'ftol':1e-10,
                                                'eps':1e-6,
                                                'gtol':1,
                                                'maxls': 30,
                                                'maxiter':150})
    est_param = result.x
    if v: 
        print("Original param: ",params)
        print("Est param: ",est_param)
        
    # need the list of tau to determine fixed one
    f0,f1,v0,v1,daily_tau,tau_l = ema.volatilityDensityUncond(est_param,args)
    
    for h,hor in enumerate(f_res.columns):
        for i,date in enumerate(f_res.index.tolist()):
            est_target = ret.index.get_loc(date,method='bfill') # Targeted date
            end = ret.index.get_loc(date,method='pad')-hor # End of estimation
            end_q = total_quarters.get_loc(ret.index[end],method='pad')
            span = ret.index[:end+2] # include one-step g as it is predetermined
            if end_q != total_quarters.get_loc(date,method='pad'):
                # Determine tau by current date + 1 period of tau
                tau_choice = tau_l.index.get_loc(total_quarters[end_q+1],
                                                            method='pad')
            else:
                # We remain in the same quarter - thus use the current tau
                tau_choice = tau_l.index.get_loc(total_quarters[end_q],
                                                            method='pad')
            fix_tau = tau_l.iloc[tau_choice].values
            args = [K,ret[:end],macro]
            f_res.iloc[i,h] = horizonUncond(est_param,args,fix_tau,hor,end)
            if save: f_res.to_csv(save)
        if v:print("Forecast:",f_res.loc[date,:])
    if save: f_res.to_csv(save)
    return f_res

def qlike(forecast, actual):
    """Method for the QLIKE evaluation function"""
    error = np.zeros(forecast.shape)
    for i,date in enumerate(forecast.index):
        a = actual.loc[date]
        f = forecast.loc[date]
        error[i] = (a/f) - np.log(a/f)-1
    return np.mean(error)

def mspe(forecast, actual):
    """Method for mean squared prediction error"""
    error2 = np.zeros(forecast.shape)
    for i,date in enumerate(forecast.index):
        error2[i] = (forecast.loc[date]-actual.loc[date])**2
    return np.mean(error2)
    
def mincerZarnowitz(forecast,actual):
    """Method to run the MZ regression for evaluation"""
    Y = actual.values
    constant = np.ones(forecast.shape)
    X = np.hstack((constant[:,np.newaxis],forecast.values[:,np.newaxis]))
    model = sm.OLS(Y,X)
    results = model.fit()
    print(results.summary())
    p = results.params
    t = results.tvalues
    r2 = results.rsquared
    return p,t,r2

def matchRV(forecast, rv):
    """ Matches the dates of the forecasts and the realized variance
        Returns: dataframe with column for forecasts and for corresponding RV
    """
    ix = pd.DatetimeIndex(forecast.index.tolist())
    x = pd.DataFrame(forecast.values,index=ix.round('d'),
                        columns=forecast.columns)
    match = pd.concat([x,rv.loc[x.index.tolist()]],axis=1)
    match = match1.dropna(axis=0)
    match.columns=['forecast','RV']
    return match

def quarterlyRV(rv,index):
    result = pd.DataFrame(np.zeros(index.shape),index=index,columns=['IntraRV'])
    for i,dat in enumerate(index):
        loc = rv.index.get_loc(dat,method='bfill')
        result.iloc[i,0] = np.sum(rv.iloc[loc-63:loc+1,0])
    return result

###############################
# Direct forecasting formulas
# Based on the Quantitative Methods for Finance lectures by Dick van Dijk,
# Lecture 3 - forecasting with GARCH
# Assumption: stationary series (enforced in the model, thus safe assumption)

def gm_h(p,f_g_1,tau,horizon):
    """GARCH(1,1) direct forecast, given tau"""
    mu,a1,b1,m,theta,w1,w2 = p[0],p[1],p[2],p[3],p[4],p[5],p[6]
    fore = 1 + ((a1+b1)**(horizon-1))*f_g_1
    return fore * tau

def asym_h(p,f_g_1,tau,horizon):
    """TGARCH(1,1) direct forecast, given tau"""
    mu,a1,b1,gamma,m,theta,w1,w2 = p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]
    fore = 1 + ((a1+b1+gamma/2)**(horizon-1))*(f_g_1-1)
    return fore * tau

def arch_h(p,ret,tau,arg,horizon):
    """ARCH(inf) direct forecast, given tau"""
    mu,a1,b1,m,theta,w1,w2 = p[0],p[1],p[2],p[3],p[4],p[5],p[6]
    # first available datapoint ... to match series
    first_tau = ret.index.get_loc(macro.index[K],method='bfill')
    shocks = ((ret.values[first_tau:] - mu)**2)/d_tau
    # add zeros for length of horizon forecast, as E(e) = 0
    shocks = np.vstack((shocks,np.zeros((horizon,1))))
    i,j = 1,0
    bpow = np.zeros(shocks.shape)
    while i>1e-8 and j<d_tau.shape[0]:
        bpow[j]=b1**j
        i = bpow[j]
        j=j+1
    # forecast = sum of constant, alpha term, gamma term
    const = (1-a1-b1-gamma/2)/(1-b1)
    p_a = a1*np.sum(np.flipud(bpow)*shocks)
    return const + p_a

def arch_asym_h(p,ret,tau,arg,horizon):
    """Asymmetric ARCH(inf) direct forecast, given tau"""
    mu,a1,b1,gamma,m,theta,w1,w2 = p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]
    # first available datapoint ... to match series
    first_tau = ret.index.get_loc(macro.index[K],method='bfill')
    shocks = ((ret.values[first_tau:] - mu)**2)/d_tau
    # add zeros for length of horizon forecast, as E(e) = 0
    shocks = np.vstack((shocks,np.zeros((horizon,1))))
    # indicator for gamma
    ind = 1*((ret.iloc[first_tau:,:].values - mu)<0)
    i,j = 1,0
    bpow = np.zeros(shocks.shape)
    while i>1e-8 and j<d_tau.shape[0]:
        bpow[j]=b1**j
        i = bpow[j]
        j=j+1
    # forecast = sum of constant, alpha term, gamma term
    const = (1-a1-b1-gamma/2)/(1-b1)
    p_a = a1*np.sum(np.flipud(bpow)*shocks)
    p_g = gamma*np.sum(np.flipud(bpow)*shocks*ind)
    return const + p_a + p_g

def ms_gm_h(self, arg):
    """Markov-Switching GARCH recursive forecast, given tau"""
    pass

def asym_RV(p,f_g_1,tau,Nt):
    mu,a1,b1,gamma,m,theta,w1,w2 = p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]
    temp = a1+b1+gamma/2
    fore = (f_g_1-1)*((1-(temp**Nt))/temp)
    return tau * (Nt+fore)
   
def horizonUncond(p,args,ftau,horizon,start):
    """Horizon forecasting formula for the regime-switching unconditional
        variance
    """
    mu,a0_0,a0_1,a,b,gamma = p[0],p[1],p[2],p[3],p[4],p[5]
    theta,w1,w2 = p[6],p[7],p[8]
    p00,p11 = p[9],p[10]
    K,ret,macro = args[0],args[1],args[2]

    # Step 1 - create tau to calculate the prior shocks
    first_tau = ret.index.get_loc(macro.index[K],method='bfill')
    midas = llf.lagMatrix(macro,K).values @ llf.betaWeight(w1,w2,K)
    # we have set m=0 due to the identification issues
    tau = pd.DataFrame(np.exp(theta*midas),index=macro.index)
    resampler = ret.index.to_period('Q').to_timestamp()
    d_tau = tau.loc[resampler].values[first_tau:,:]
    r = ret.values[first_tau:,:]
    r2 = (r - mu)**2
    shocks = r2/d_tau
    ind = 1*((r - mu)<0)
    
    forecast = np.zeros((horizon,1))
    f0_p,f1_p,v0,v1,daily_tau,_ = ema.volatilityDensityUncond(p,args)
    inf,fore = ema.filterProbability(p[-2],p[-1],f0_p,f1_p)
    
    linf = inf[-1,:] # P(st = 1|t)
    lfore = fore[-1,:]
    
    bpow = np.zeros(shocks.shape)
    i = 1
    j = 0
    while i>1e-8 and j<bpow.shape[0]:
        bpow[j] = b**(j+horizon)
        i = bpow[j,0]
        j+=1
    s0 = np.sum(np.flipud(bpow)*shocks)
    s1 = np.sum(np.flipud(bpow)*shocks*ind)
    g0 = a0_0/(1-b) + a*s0 + gamma*s1
    g1 = a0_1/(1-b) + a*s0 + gamma*s1
    
    for i in range(horizon):
        linf[0] = p00*linf[0] + (1-p11)*linf[1]
        linf[1] = (1-p00)*linf[0] + p11*linf[1]
    return ftau*(linf[0]*g0 + linf[1]*g1)
    
###############################
# Recreating the different types of series
def recreateSimple(p,data,est):
    """ Recreate the tau and g series based on the ARCH specification with which
        it was originally estimated - this is the simple GARCH-MIDAS
        model without regime-switching component
    """
    K,ret,macro = data[0],data[1],data[2]
    if est in ['gm','arch']:
        mu,a1,b1,m,theta,w1,w2 = p[0],p[1],p[2],p[3],p[4],p[5],p[6]
    elif est in ['asym','arch_asym']:
        mu,a1,b1,gamma,m,theta,w1,w2 = p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]
    elif est in ['ms_gm','ms_asym']: return recreateMS(p,data)
    else: print('please specify mode')
    
    # Step 1 - create tau
    first_tau = ret.index.get_loc(macro.index[K],method='bfill')
    midas = llf.lagMatrix(macro,K).values @ llf.betaWeight(w1,w2,K)
    tau = pd.DataFrame(np.exp(m+(theta*midas)),
                    index=macro.index,columns=['tau'])
    resample = ret.index.to_period('Q').to_timestamp()
    d_tau = tau.loc[resample].values[first_tau:,:]
    
    # Step 2 - determine g
    r2 = (ret.iloc[first_tau:,:].values - mu)**2
    shocks = r2/d_tau
    ind = 1*((ret.iloc[first_tau:,:].values - mu)<0)
    g = np.zeros(d_tau.shape)
    g[0] = 1
    for i in range(1,g.shape[0]):
        g[i] = (1-a1-b1-(gamma/2)) + (a1+gamma*ind[i-1])*shocks[i-1] + b1*g[i-1]
    # Create G dependent on the specification chosen
    if est == 'gm':
        for i in range(1,g.shape[0]):
            g[i] = (1-a1-b1) + a1*shocks[i-1] + b1*g[i-1]
    elif est == 'asym':
        for i in range(1,g.shape[0]):
            g[i] = (1-a1-b1-(gamma/2))
            g[i] += (a1+gamma*ind[i-1])*shocks[i-1] + b1*g[i-1]
    elif est == 'arch':
        g = ((1-a1-b1)/(1-b1))*np.ones(d_tau.shape)
        for i in range(g.shape[0]):
            g[i] += a1*np.sum(np.flipud(bpow[:i])*shocks[:i,:])
    elif est == 'arch_asym':
        g = ((1-a1-b1-gamma/2)/(1-b1))*np.ones(d_tau.shape)
        for i in range(g.shape[0]):
            g[i] += a1*np.sum(np.flipud(bpow[:i])*shocks[:i,:])
            g[i] += gamma*np.sum(np.flipud(bpow[:i])*shocks[:i,:]*ind[:i,:])
    
    g = pd.DataFrame(np.hstack([g,d_tau]),index=ret.index[first_tau:],
                            columns=['g','d_tau'])
    return g,tau

    
