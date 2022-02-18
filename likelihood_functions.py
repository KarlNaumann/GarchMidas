import numpy as np
import pandas as pd

#####################################################
# Auxiliary Functions #
def betaWeight(w1,w2,K):
    """Function to calculate the beta weights
        Input: omega 1 and 2, midas lag years
        Returns: vector containing the weight at each lag K"""
    tot = np.sum([((k/K)**(w1-1))*((1-(k/K))**(w2-1)) for k in range(1,K+1)])
    array = np.array([((k/K)**(w1-1))*((1-(k/K))**(w2-1))*(1/tot) for k in range(1,K+1)])
    return array[:,np.newaxis]

def lagMatrix(df,K):
    """Function to help create lag matrix for tau:
        Input: macroeconomic data & desired lags (K)
        Output: dataframe with all lags required for given date"""
    array = np.zeros((df.shape[0],K))
    for i in range(K,df.shape[0]):
        choice = np.flip(df.iloc[i-K:i].values,axis=0)
        array[i,:] = choice[:].T
    return pd.DataFrame(array,index=df.index)

#####################################################
# Likelihood Functions #

def garch11(p,data):
    """negative Log-likelihood function of the GARCH-MIDAS of Engle et. Al. 2013
        Can be found in their original paper on page 782
        Input:  vector p of parameters to be optimized, must have length 7
                list data, which contains MIDAS lag years K, 
                        returns DataFrame (mean 0 ), and macro DataFrame
        Note: the dataframes should have a date-time index, with macro only having monthly
                entries, and returns having d entries. For example: macro has '2019-01-01' 
                and '2019-02-01', returns has '2019-01-01','2019-01-02',...,'2019-02-01'
        Returns: negative log-likelihood value"""
    
    a1,b1,m,theta,w1,w2 = p[0],p[1],0,1,1,1
    K,ret,macro = data[0],data[1],data[2]
    
    # Step 1 - create tau 
    first_tau = ret.index.get_loc(macro.index[K-1],method='bfill')
    midas = lagMatrix(macro,K).values @ betaWeight(w1,w2,K)
    tau = pd.DataFrame(np.exp(m+(theta*midas)),index=macro.index)
    d_tau = tau.loc[ret.index.to_period('M').to_timestamp()].values[first_tau:,:]
    
    # Step 2 - determine g
    r2 = ret.values[first_tau:]**2
    g = np.zeros(d_tau.shape)
    g[0] = 1
    for i in range(1,g.shape[0]):
        g[i] = (1-a1-b1) + a1*r2[i-1] + b1*g[i-1]
    
    # Step 3 - Log likelihood function
    lpi = np.log(2*np.pi*np.ones(g.shape))
    nllf=0.5*np.sum(lpi+np.log(g)+(r2/g))
    if np.isnan(nllf) or np.isinf(nllf): nllf= 9e15
    return nllf

# Engle et Al. (2013) - based GM function
def simpleGM(p,data):
    """negative Log-likelihood function of the GARCH-MIDAS of Engle et. Al. 2013
        Can be found in their original paper on page 782
        Input:  vector p of parameters to be optimized, must have length 7
                list data, which contains MIDAS lag years K, 
                        returns DataFrame (mean 0 ), and macro DataFrame
        Note: the dataframes should have a date-time index, with macro only having monthly
                entries, and returns having d entries. For example: macro has '2019-01-01' 
                and '2019-02-01', returns has '2019-01-01','2019-01-02',...,'2019-02-01'
        Returns: negative log-likelihood value"""
    
    mu,a1,b1,m,theta,w1,w2 = p[0],p[1],p[2],p[3],p[4],p[5],p[6]
    K,ret,macro = data[0],data[1],data[2]
    
    # Step 1 - create tau 
    first_tau = ret.index.get_loc(macro.index[K],method='bfill')
    midas = lagMatrix(macro,K).values @ betaWeight(w1,w2,K)
    tau = pd.DataFrame(np.exp(m+(theta*midas)),index=macro.index)
    resample = ret.index.to_period('Q').to_timestamp()
    d_tau = tau.loc[resample].values[first_tau:,:]
    
    # Step 2 - determine g
    r2 = (ret.values[first_tau:] - np.mean(ret.values[first_tau:]))**2
    shocks = r2/d_tau
    g = np.zeros(d_tau.shape)
    g[0] = 1
    for i in range(1,g.shape[0]):
        g[i] = (1-a1-b1) + a1*shocks[i-1] + b1*g[i-1]
    
    # Step 3 - Log likelihood function
    vol = d_tau*g
    lpi = np.log(2*np.pi*np.ones(vol.shape))
    lvol = np.log(vol)
    nllf=0.5*np.sum(lpi+lvol+(r2/vol))
    if np.isnan(nllf) or np.isinf(nllf): nllf= 9e15
    return nllf

# Conrad & Loch (2015) - asymmetric GARCH MIDAS
def simpleGM_asym(p,data):
    """negative Log-likelihood function of the GARCH-MIDAS of Engle et. Al. 2013
        Can be found in their original paper on page 782
        Input:  vector p of parameters to be optimized, must have length 7
                list data, which contains MIDAS lag years K, 
                        returns DataFrame (mean 0 ), and macro DataFrame
        Note: the dataframes should have a date-time index, with macro only having monthly
                entries, and returns having d entries. For example: macro has '2019-01-01' 
                and '2019-02-01', returns has '2019-01-01','2019-01-02',...,'2019-02-01'
        Returns: negative log-likelihood value"""
    
    mu,a1,b1,gamma,m,theta,w1,w2 = p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]
    K,ret,macro = data[0],data[1],data[2]
    
    # Step 1 - create tau 
    first_tau = ret.index.get_loc(macro.index[K],method='bfill')
    midas = lagMatrix(macro,K).values @ betaWeight(w1,w2,K)
    tau = pd.DataFrame(np.exp(m+(theta*midas)),index=macro.index)
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
    # Step 3 - Log likelihood function
    vol = d_tau*g
    lpi = np.log(2*np.pi*np.ones(vol.shape))
    lvol = np.log(vol)
    nllf=0.5*np.sum(lpi+lvol+(r2/vol))
    if np.isnan(nllf) or np.isinf(nllf): nllf= 9e15
    return nllf

# ARCH(inf) version of Engle et Al. GM
def simpleGM_ARCH(p,data):
    """negative Log-likelihood function of the GARCH-MIDAS of Engle et. Al. 2013
        Can be found in their original paper on page 782
        Input:  vector p of parameters to be optimized, must have length 7
                list data, which contains MIDAS lag years K, 
                        returns DataFrame (mean 0 ), and macro DataFrame"""

    a1,b1,m,theta,w1,w2 = p[0],p[1],p[2],p[3],p[4],p[5]
    K,ret,macro = data[0],data[1],data[2]
    # Step 1 - create tau 
    first_tau = ret.index.get_loc(macro.index[K],method='bfill')
    midas = lagMatrix(macro,K).values @ betaWeight(w1,w2,K)
    tau = pd.DataFrame(np.exp(m**2+(theta**2*midas)),index=macro.index)
    resampler = ret.index.to_period('Q').to_timestamp()
    d_tau = tau.loc[resampler].values[first_tau:,:]
    
    # Step 2 - determine g
    i = 1
    j=0
    bpow = np.zeros(d_tau.shape)
    while i>1e-8 and j<d_tau.shape[0]:
        bpow[j]=b1**j
        i = bpow[j]
        j=j+1
    r2 = (ret.values[first_tau:] - np.mean(ret.values[first_tau:]))**2
    shocks = r2/d_tau
    g = ((1-a1-b1)/(1-b1))*np.ones(d_tau.shape)
    for i in range(g.shape[0]):
        g[i] += a1*np.sum(np.flipud(bpow[:i])*shocks[:i,:])

    # Step 3 - Log likelihood function
    vol = d_tau*g
    lpi = np.log(2*np.pi*np.ones(vol.shape))
    lvol = np.log(vol)
    nllf=0.5*np.sum(lpi+lvol+(r2/vol))
    if np.isnan(nllf) or np.isinf(nllf): nllf= 9e15
    return nllf

# ARCH(inf) version of Conrad & Loch asymmetric GM
def simpleGM_ARCH_asym(p, data):
    """docstring for simpleGM_ARCH_asym"""
    """negative Log-likelihood function of the GARCH-MIDAS of Engle et. Al. 2013
        Can be found in their original paper on page 782
        Input:  vector p of parameters to be optimized, must have length 7
                list data, which contains MIDAS lag years K, 
                        returns DataFrame (mean 0 ), and macro DataFrame"""

    mu,a1,b1,gamma,m,theta,w1,w2 = p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]
    K,ret,macro = data[0],data[1],data[2]
    
    # Step 1 - create tau 
    first_tau = ret.index.get_loc(macro.index[K],method='bfill')
    midas = lagMatrix(macro,K).values @ betaWeight(w1,w2,K)
    tau = pd.DataFrame(np.exp(m+(theta*midas)),index=macro.index)
    resampler = ret.index.to_period('Q').to_timestamp()
    d_tau = tau.loc[resampler].values[first_tau:,:]
    
    # Step 2 - determine g
    i = 1
    j=0
    bpow = np.zeros(d_tau.shape)
    while i>1e-8 and j<d_tau.shape[0]:
        bpow[j]=b1**j
        i = bpow[j]
        j=j+1
    r2 = (ret.values[first_tau:] - mu)**2
    shocks = r2/d_tau
    ind = 1*((ret.iloc[first_tau:,:].values - mu)<0)
    g = ((1-a1-b1)/(1-b1))*np.ones(d_tau.shape)
    for i in range(g.shape[0]):
        g[i] += a1*np.sum(np.flipud(bpow[:i])*shocks[:i,:])
        g[i] += gamma*np.sum(np.flipud(bpow[:i])*shocks[:i,:]*ind[:i,:])

    # Step 3 - Log likelihood function
    vol = d_tau*g
    lpi = np.log(2*np.pi*np.ones(vol.shape))
    lvol = np.log(vol)
    nllf=0.5*np.sum(lpi+lvol+(r2/vol))
    if np.isnan(nllf) or np.isinf(nllf): nllf= 9e15
    return nllf