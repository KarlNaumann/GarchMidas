import numpy as np
import pandas as pd
from random import sample as randomSample
from matplotlib import pyplot as plt
from scipy.optimize import minimize,SR1,basinhopping,Bounds
import numdifftools as ndt

import thesis_settings as tset
import data
import likelihood_functions as llf
import basin_settings as bs
import thesis_settings as tset
files = tset.ms_results()

np.random.seed(42)

########## Basic Settings ##########
def boundsFull():
    # Order: 
    # Restricted case lb & ub
    #           mu,  a0,  a1,  b0,  b1,  g0,  g1,   m,   theta,w1, w2, p0, p1
    lb_res =  [-1,1e-8,1e-8,1e-8,1e-8,-1e2,-1e2,-1e15,-1e15, 1,   1, 0, 0]
    ub_res =  [ 1,1,1,   1,   1,   2,   2, 1e15, 1e15, 1,1e15, 1, 1]
    # unrestricted case lb & ub
    #           mu,  a0,  a1,  b0,  b1,  g0,  g1,    m,   theta, w1, w2,  p0, p1
    lb_free = [-1,1e-8,1e-8,1e-8,1e-8,-1e2,-1e2,-1e15,-1e15,-1e15,   1, 0, 0]
    ub_free = [ 1,1,1,1,   1,   2,   2, 1e15, 1e15, 1e15,1e15, 1, 1]
    # make sure chosen x always within bounds
    feasible = [True for i in range(len(lb_res))]
    # create bounds objects
    rest = Bounds(lb_res,ub_res,keep_feasible=feasible)
    unrest = Bounds(lb_free,ub_free,keep_feasible=feasible)
    return rest,unrest

def boundsUncond():
    # Order: 
    # Restricted case lb & ub
    [0.03,0.1,1,0.03,0.9,0.1,0.2,-0.26,1,1.437,0.8,0.8]
    #          mu,a0_0,a0_1,   a,   b,   g,theta,w1,  w2,p0,p1
    lb_res =  [-1,1e-8,1e-8,1e-8,1e-8,-1e2,-1e15, 1,   1, 0, 0]
    ub_res =  [ 1,1e15,1e15,   1,   1,   2, 1e15, 1,1e15, 1, 1]
    # unrestricted case lb & ub
    #          mu,a0_0,a0_1,   a,   b,   g,theta,    w1,  w2,p0,p1
    lb_free = [-1,1e-8,1e-8,1e-8,1e-8,-1e2,-1e15,-1e15,   1, 0, 0]
    ub_free = [ 1,1e15,1e15,   1,   1,   2, 1e15, 1e15,1e15, 1, 1]
    # make sure chosen x always within bounds
    feasible = [True for i in range(len(lb_res))]
    # create bounds objects
    rest = Bounds(lb_res,ub_res,keep_feasible=feasible)
    unrest = Bounds(lb_free,ub_free,keep_feasible=feasible)
    return rest,unrest

def boundsGARCH():
    # Bounds for the GARHC(1,1) model
    #           mu,  a0,  a1,  a,  b1,  g0, p0,p1
    lb_res =  [-1,1e-8,1e-8,1e-8,1e-8,1e-8, 0, 0]
    ub_res =  [ 1,1e15,1e15,   1,   1,   1, 1, 1]
    # make sure chosen x always within bounds
    feasible = [True for i in range(len(lb_res))]
    return Bounds(lb_res,ub_res,keep_feasible=feasible)

class StepUncondUR(object):
    """Class that generates custom step-sizes for the parameters in the 
        Markov-switching GM models - larger steps are taken for the weights 
        w1,w2 than for others"""
    def __init__(self, stepsize=1,maximum=2,rest=True,maxTry=1000):
        self.stepsize = stepsize
        self.max = maximum
        self.rest = rest
        self.maxTry=maxTry
    def __call__(self, x):
        # Order: a0,b0,a1,b1,m,theta,w1,w2,p00,p11
        s = self.stepsize
        prior = x
        for i in range(self.maxTry):
            x[-4]  += np.random.uniform(1-x[-4], self.max*s)  # w1
            x[-3]  += np.random.uniform(1-x[-3], self.max*s)  # w2
            if x[-4]>0 and x[-3]>1: break
            else: 
                x[-4]=prior[-4]
                x[-3]=prior[-3]
            
        print("Attempt: ", x)
        return np.array(x)

class StepUncondR(object):
    """Class that generates custom step-sizes for the parameters in the 
        Markov-switching GM models - larger steps are taken for the weights 
        w1,w2 than for others"""
    def __init__(self, stepsize=1,maximum=2,rest=True,maxTry=1000):
        self.stepsize = stepsize
        self.max = maximum
        self.rest = rest
        self.maxTry = maxTry
    def __call__(self, x):
        # Order: a0,b0,a1,b1,m,theta,w1,w2,p00,p11
        s = self.stepsize
        prior = x
        for i in range(self.maxTry):
            x[1] += np.random.uniform(-0.5*s, 0.5*s)
            x[2] += np.random.uniform(-0.5*s, 0.5*s)
            x[-4]  = 1  # w1
            x[-3]  += np.random.uniform(1-x[-3], self.max*s)  # w2
            if x[-4]>0 and x[-3]>1: break
            else: 
                x[-4]=prior[-4]
                x[-3]=prior[-3]
            if x[1] < 0: x[1] = 1e-10
            if x[2] < 0: x[2] = 0.1
        print("Attempt: ", x)
        return np.array(x)

class StepGarch(object):
    """Class that generates custom step-sizes for the parameters in the 
        Markov-switching GM models - larger steps are taken for the weights 
        w1,w2 than for others"""
    def __init__(self, stepsize=1,maximum=2,rest=True,maxTry=1000):
        self.stepsize = stepsize
        self.max = maximum
        self.rest = rest
        self.maxTry = maxTry
    def __call__(self, x):
        # Order: a0,b0,a1,b1,m,theta,w1,w2,p00,p11
        s = self.stepsize
        x[1] += np.random.uniform(-0.02*s,0.02*s)
        x[2] += np.random.uniform(-0.5*s,0.5*s)
        x[-2] = 0.95
        x[-1] = 0.95
        print("Attempt: ", x)
        return np.array(x)
        
########## Probability Calculations ##########
def filterProbability(p00,p11,f0,f1): # fore(t) is e(t|t-1)
    P = np.array([[p00,1-p00],[1-p11,p11]]) # transition probabilities
    inf = np.zeros((f0.shape[0],2)) # Matrix to e(t|t)
    fore = np.zeros((f0.shape[0],2)) # Matrix for e(t|t-1)
    
    # Develop starting point
    inf[0,0] = (1-p11)/(2-p00-p11) # p(s00 = 0 | R00)
    inf[0,1] = (1-p00)/(2-p00-p11) # p(s00 =1 | R00)
    fore[0,0] = inf[0,0] # since nothing prior exists, this is also p(s0|s-1)
    fore[0,1] = inf[0,1]
    
    # Recursion:
    for i in range(1,fore.shape[0]):
        fore[i,0] = p00*inf[i-1,0] + (1-p11)*inf[i-1,1]
        fore[i,1] = (1-p00)*inf[i-1,0] + p11*inf[i-1,1]
        #fore[i,:] = (P @ inf[i-1,:][:,np.newaxis]).T
        inf[i,0] = f0[i]*fore[i,0] / (f0[i]*fore[i,0] + f1[i]*fore[i,1])
        inf[i,1] = f1[i]*fore[i,1] / (f0[i]*fore[i,0] + f1[i]*fore[i,1])
    return inf,fore
    
def smoothProbability(p00,p11,fore,inf): # fore(t) is e(t|t-1)
    smooth = np.zeros(inf.shape) # e(t|T)
    smooth[-1,0] = inf[-1,0]
    smooth[-1,1] = inf[-1,1]
    for i in range(inf.shape[0]-2,-1,-1):
        p_0_T = smooth[i+1,0] # One-step given all info
        p_1_T = smooth[i+1,1]
        p0_t = inf[i,0] # Current given current info
        p1_t = inf[i,1]
        p_0_t = fore[i+1,0] # one-step given current
        p_1_t = fore[i+1,1]
        p00_s = (p_0_T * p0_t * p00) / p_0_t
        p01_s = (p_1_T * p0_t * (1-p00)) / p_1_t
        p10_s = (p_0_T * p1_t * (1-p11)) / p_0_t
        p11_s = (p_1_T * p1_t * p11) / p_1_t
        smooth[i,0] = p00_s + p01_s
        smooth[i,1] = p10_s + p11_s
    return smooth

########## Likelihood Functions #########
# regime in unconditional variance only
def llfHaasUncond(p,args):
    """Log-Likelihood for MS in unconditional variance aka. a0"""
    mu,a0_0,a0_1,a,b,gamma = p[0],p[1],p[2],p[3],p[4],p[5]
    theta,w1,w2 = p[6],p[7],p[8]
    p00,p11 = p[9],p[10]
    eta = (1-p11)/(2-p00-p11)
    K,ret,macro,expt = args[0],args[1],args[2],args[3]
    
    # Step 1 - create tau 
    first_tau = ret.index.get_loc(macro.index[K],method='bfill')
    midas = llf.lagMatrix(macro,K).values @ llf.betaWeight(w1,w2,K)
    # we have set m=0 due to the identification issues
    tau = pd.DataFrame(np.exp(theta*midas),index=macro.index)
    resampler = ret.index.to_period('Q').to_timestamp()
    d_tau = tau.loc[resampler].values[first_tau:,:]
    
    # Step 2 - create the different g series
    bpow,g0,g1 = [np.zeros(d_tau.shape) for i in range(3)]
    i = 1
    j = 0
    while i>1e-8 and j<d_tau.shape[0]:
        bpow[j]=b**j
        i = bpow[j,0]
        j=j+1
    
    r = ret.values[first_tau:,:]
    r2 = (r - mu)**2
    shocks = r2/d_tau
    ind = 1*((r - mu)<0)
        
    for i in range(g0.shape[0]):
        s0 = np.sum(np.flipud(bpow[:i])*shocks[:i,:])
        s1 = np.sum(np.flipud(bpow[:i])*shocks[:i,:]*ind[:i,:])
        g0[i] = a0_0/(1-b) + a*s0 + gamma*s1
        g1[i] = a0_1/(1-b) + a*s0 + gamma*s1
    
    v0 = g0*d_tau
    v1 = g1*d_tau
    f0 = (1/np.sqrt(2*np.pi*v0))*np.exp(-1*(r2/(2*v0)))
    f1 = (1/np.sqrt(2*np.pi*v1))*np.exp(-1*(r2/(2*v1)))
    f0[f0==0] = 1e-323 # Avoid 0 division log error (exp(z)==0 for -z extreme)
    f1[f1==0] = 1e-323
    
    coef_00 = 1 - expt[1:] - expt[:-1] + p11*expt[:-1]
    coef_01 = expt[1:] - p11*expt[:-1]
    coef_10 = expt[:-1] - p11*expt[:-1]
    coef_11 = p11*expt[:-1]
    
    llfunction = np.sum([
        np.sum((1-expt)*np.log(f0)),
        np.sum(expt*np.log(f1)),
        np.sum(coef_00*np.log(p00)),
        np.sum(coef_01*np.log(1-p00)),
        np.sum(coef_10*np.log(1-p11)),
        np.sum(coef_11*np.log(p11)),
        expt[0]*eta,
        (1-expt[0])*(1-eta)
    ])
    if np.isnan(llfunction) or np.isinf(llfunction): llfunction= -9e15
    return -llfunction

def llfHaasUncondDirect(p,args):
    """Log-Likelihood for MS in unconditional variance aka. a0"""
    mu,a0_0,a0_1,a,b,gamma = p[0],p[1],p[2],p[3],p[4],p[5]
    theta,w1,w2 = p[6],p[7],p[8]
    p00,p11 = p[9],p[10]
    P = np.array([[p00,1-p00],[1-p11,p11]])
    K,ret,macro = args[0],args[1],args[2]
        
    # Step 1 - create tau 
    first_tau = ret.index.get_loc(macro.index[K],method='bfill')
    midas = llf.lagMatrix(macro,K).values @ llf.betaWeight(w1,w2,K)
    # we have set m=0 due to the identification issues
    tau = pd.DataFrame(np.exp(theta*midas),index=macro.index)
    resampler = ret.index.to_period('Q').to_timestamp()
    d_tau = tau.loc[resampler].values[first_tau:,:]
    
    # Step 2 - create the different g series
    bpow,g0,g1 = [np.zeros(d_tau.shape) for i in range(3)]
    i = 1
    j = 0
    while i>1e-8 and j<d_tau.shape[0]:
        bpow[j]=b**j
        i = bpow[j,0]
        j=j+1
    
    r = ret.values[first_tau:,:]
    r2 = (r - mu)**2
    shocks = r2/d_tau
    ind = 1*((r - mu)<0)
        
    for i in range(g0.shape[0]):
        s0 = np.sum(np.flipud(bpow[:i])*shocks[:i,:])
        s1 = np.sum(np.flipud(bpow[:i])*shocks[:i,:]*ind[:i,:])
        g0[i] = a0_0/(1-b) + a*s0 + gamma*s1
        g1[i] = a0_1/(1-b) + a*s0 + gamma*s1
    
    v0 = g0*d_tau
    v1 = g1*d_tau

    f0 = (1/np.sqrt(2*np.pi*v0))*np.exp(-1*(r2/(2*v0)))
    f1 = (1/np.sqrt(2*np.pi*v1))*np.exp(-1*(r2/(2*v1)))
    f0[f0==0]=1e-323
    f1[f1==0]=1e-323
    
    inf = np.zeros((f0.shape[0],2)) # Matrix to e(t|t)
    fore = np.zeros((f0.shape[0],2)) # Matrix for e(t|t-1)
    inf[0,0] = (1-p11)/(2-p00-p11) # p(s00 = 0 | R00)
    inf[0,1] = (1-p00)/(2-p00-p11) # p(s00 = 1 | R00)
    fore[0,0] = inf[0,0] # since nothing prior exists, this is also p(s0|s-1)
    fore[0,1] = inf[0,1]
    
    # Recursion:
    for i in range(1,fore.shape[0]):
        fore[i,0] = p00*inf[i-1,0] + (1-p11)*inf[i-1,1] # p(st=0|st-1)
        fore[i,1] = (1-p00)*inf[i-1,0] + p11*inf[i-1,1] # p(st=1|st-1)
        inf[i,0] = f0[i]*fore[i,0] / (f0[i]*fore[i,0] + f1[i]*fore[i,1])
        inf[i,1] = f1[i]*fore[i,1] / (f0[i]*fore[i,0] + f1[i]*fore[i,1])
    
    pt0 = fore[:,0][:,np.newaxis]*f0
    pt1 = fore[:,1][:,np.newaxis]*f1
    llfunction = np.sum(np.log(pt0 + pt1))
    
    if np.isnan(llfunction) or np.isinf(llfunction): llfunction= -9e15
    return -llfunction

def llfHaasFullDirect(p,args):
    """Log-Likelihood for MS in unconditional variance aka. a0"""
    mu,a0_0,a0_1,a0,a1,b0,b1 = p[0],p[1],p[2],p[3],p[4],p[5],p[6]
    gamma0,gamma1 = p[7],p[8]
    theta,w1,w2 = p[9],p[10],p[11]
    p00,p11 = p[12],p[13]
    P = np.array([[p00,1-p00],[1-p11,p11]])
    K,ret,macro = args[0],args[1],args[2]
        
    # Step 1 - create tau 
    first_tau = ret.index.get_loc(macro.index[K],method='bfill')
    midas = llf.lagMatrix(macro,K).values @ llf.betaWeight(w1,w2,K)
    # we have set m=0 due to the identification issues
    tau = pd.DataFrame(np.exp(theta*midas),index=macro.index)
    resampler = ret.index.to_period('Q').to_timestamp()
    d_tau = tau.loc[resampler].values[first_tau:,:]
    
    # Step 2 - create the different g series
    bpow0,bpow1,g0,g1 = [np.zeros(d_tau.shape) for i in range(4)]
    i = 1
    j = 0
    while i>1e-8 and j<d_tau.shape[0]:
        bpow0[j]=b0**j
        i = bpow0[j,0]
        j=j+1
    i = 1
    j = 0
    while i>1e-8 and j<d_tau.shape[0]:
        bpow1[j]=b1**j
        i = bpow1[j,0]
        j=j+1
    
    r = ret.values[first_tau:,:]
    r2 = (r - mu)**2
    shocks = r2/d_tau
    ind = 1*((r - mu)<0)
        
    for i in range(g0.shape[0]):
        s0_0 = np.sum(np.flipud(bpow0[:i])*shocks[:i,:])
        s1_0 = np.sum(np.flipud(bpow0[:i])*shocks[:i,:]*ind[:i,:])
        s0_1 = np.sum(np.flipud(bpow1[:i])*shocks[:i,:])
        s1_1 = np.sum(np.flipud(bpow1[:i])*shocks[:i,:]*ind[:i,:])
        g0[i] = a0_0/(1-b0) + a0*s0_0 + gamma0*s1_0
        g1[i] = a0_1/(1-b1) + a1*s0_1 + gamma1*s1_1
    
    v0 = g0*d_tau
    v1 = g1*d_tau

    f0 = (1/np.sqrt(2*np.pi*v0))*np.exp(-1*(r2/(2*v0)))
    f1 = (1/np.sqrt(2*np.pi*v1))*np.exp(-1*(r2/(2*v1)))
    f0[f0==0]=1e-323
    f1[f1==0]=1e-323
    
    inf = np.zeros((f0.shape[0],2)) # Matrix to e(t|t)
    fore = np.zeros((f0.shape[0],2)) # Matrix for e(t|t-1)
    inf[0,0] = (1-p11)/(2-p00-p11) # p(s00 = 0 | R00)
    inf[0,1] = (1-p00)/(2-p00-p11) # p(s00 = 1 | R00)
    fore[0,0] = inf[0,0] # since nothing prior exists, this is also p(s0|s-1)
    fore[0,1] = inf[0,1]
    
    # Recursion:
    for i in range(1,fore.shape[0]):
        fore[i,0] = p00*inf[i-1,0] + (1-p11)*inf[i-1,1] # p(st=0|st-1)
        fore[i,1] = (1-p00)*inf[i-1,0] + p11*inf[i-1,1] # p(st=1|st-1)
        inf[i,0] = f0[i]*fore[i,0] / (f0[i]*fore[i,0] + f1[i]*fore[i,1])
        inf[i,1] = f1[i]*fore[i,1] / (f0[i]*fore[i,0] + f1[i]*fore[i,1])
    
    pt0 = fore[:,0][:,np.newaxis]*f0
    pt1 = fore[:,1][:,np.newaxis]*f1
    llfunction = np.sum(np.log(pt0 + pt1))
    
    if np.isnan(llfunction) or np.isinf(llfunction): llfunction= -9e15
    return -llfunction

# GARCH(1,1) regime switching
def llfgarch11(p,args):
    mu,a0_0,a0_1,a,b,gamma = p[0],p[1],p[2],p[3],p[4],p[5]
    p00,p11 = p[6],p[7]
    P = np.array([[p00,1-p00],[1-p11,p11]])
    ret = args
    eta = (1-p11)/(2-p00-p11)
    
    # Step 2 - create the different g series
    bpow,g0,g1 = [np.zeros(ret.shape) for i in range(3)]
    i = 1
    j = 0
    while i>1e-8 and j<ret.shape[0]:
        bpow[j]=b**j
        i = bpow[j,0]
        j=j+1
    
    r = ret.values
    r2 = (r - mu)**2
    shocks = r2
    ind = 1*((r - mu)<0)
        
    for i in range(g0.shape[0]):
        s0 = np.sum(np.flipud(bpow[:i])*shocks[:i,:])
        s1 = np.sum(np.flipud(bpow[:i])*shocks[:i,:]*ind[:i,:])
        g0[i] = a0_0/(1-b) + a*s0 #+ gamma*s1
        g1[i] = a0_1/(1-b) + a*s0 #+ gamma*s1
    
    f0 = (1/np.sqrt(2*np.pi*g0))*np.exp(-1*(r2/(2*g0)))
    f1 = (1/np.sqrt(2*np.pi*g1))*np.exp(-1*(r2/(2*g1)))
    f0[f0==0]=1e-323
    f1[f1==0]=1e-323
    
    inf = np.zeros((f0.shape[0],2)) # Matrix to e(t|t)
    fore = np.zeros((f0.shape[0],2)) # Matrix for e(t|t-1)
    inf[0,0] = (1-p11)/(2-p00-p11) # p(s00 = 0 | R00)
    inf[0,1] = (1-p00)/(2-p00-p11) # p(s00 = 1 | R00)
    fore[0,0] = inf[0,0] # since nothing prior exists, this is also p(s0|s-1)
    fore[0,1] = inf[0,1]
    
    # Recursion:
    for i in range(1,fore.shape[0]):
        fore[i,0] = p00*inf[i-1,0] + (1-p11)*inf[i-1,1] # p(st=0|st-1)
        fore[i,1] = (1-p00)*inf[i-1,0] + p11*inf[i-1,1] # p(st=1|st-1)
        inf[i,0] = f0[i]*fore[i,0] / (f0[i]*fore[i,0] + f1[i]*fore[i,1])
        inf[i,1] = f1[i]*fore[i,1] / (f0[i]*fore[i,0] + f1[i]*fore[i,1])
    
    pt0 = fore[:,0][:,np.newaxis]*f0
    pt1 = fore[:,1][:,np.newaxis]*f1
    llfunction = np.sum(np.log(pt0 + pt1))
    
    if np.isnan(llfunction) or np.isinf(llfunction): llfunction= -9e15
    return -llfunction
      
########## Utility Functions ##########
def volatilityDensityUncond(p,args):
    """Calculate the densities of the volatility under each regime"""
    mu,a0_0,a0_1,a,b,gamma = p[0],p[1],p[2],p[3],p[4],p[5]
    theta,w1,w2 = p[6],p[7],p[8]
    p00,p11 = p[9],p[10]
    P = np.array([[p00,1-p00],[1-p11,p11]])
    K,ret,macro = args[0],args[1],args[2]
    eta = (1-p11)/(2-p00-p11)
        
    # Step 1 - create tau 
    first_tau = ret.index.get_loc(macro.index[K],method='bfill')
    midas = llf.lagMatrix(macro,K).values @ llf.betaWeight(w1,w2,K)
    # we have set m=0 due to the identification issues
    tau = pd.DataFrame(np.exp(theta*midas),index=macro.index)
    resampler = ret.index.to_period('Q').to_timestamp()
    d_tau = tau.loc[resampler].values[first_tau:,:]
    
    # Step 2 - create the different g series
    bpow,g0,g1 = [np.zeros(d_tau.shape) for i in range(3)]
    i = 1
    j = 0
    while i>1e-8 and j<d_tau.shape[0]:
        bpow[j]=b**j
        i = bpow[j,0]
        j=j+1
    r = ret.values[first_tau:,:]
    r2 = (r - mu)**2
    shocks = r2/d_tau
    ind = 1*((r - mu)<0)
    for i in range(g0.shape[0]):
        s0 = np.sum(np.flipud(bpow[:i])*shocks[:i,:])
        s1 = np.sum(np.flipud(bpow[:i])*shocks[:i,:]*ind[:i,:])
        g0[i] = a0_0/(1-b) + a*s0 + gamma*s1
        g1[i] = a0_1/(1-b) + a*s0 + gamma*s1
        
    # Volatility process under each regime
    v0 = pd.DataFrame(g0*d_tau, index=ret.index[first_tau:])
    v1 = pd.DataFrame(g1*d_tau, index=ret.index[first_tau:])
    daily_tau = pd.DataFrame(d_tau, index=ret.index[first_tau:])
    # Under the assumption of normal distribution
    f0 = (1/np.sqrt(2*np.pi*v0.values))*np.exp(-1*(r2/(2*v0.values)))
    f1 = (1/np.sqrt(2*np.pi*v1.values))*np.exp(-1*(r2/(2*v1.values)))
    f0[f0==0]=1e-323
    f1[f1==0]=1e-323
    return f0,f1,v0,v1,daily_tau,tau

def volatilityDensityGARCH11(p,args):
    mu,a0,b0,gamma0 = p[0],p[1],p[2],p[3]
    a1,b1,gamma1 = p[4],p[5],p[6]
    K,ret,macro = args[0],args[1],args[2]
    
    # Create the different g series
    bpow0,bpow1,g0,g1 = [np.zeros(ret.shape) for i in range(4)]
    i = 1
    j = 0
    while i>1e-8 and j<ret.shape[0]:
        bpow0[j]=b0**j
        i = bpow0[j,0]
        j=j+1
    i = 1
    j = 0
    while i>1e-8 and j<ret.shape[0]:
        bpow1[j]=b1**j
        i = bpow1[j,0]
        j=j+1
    
    r = ret.values
    r2 = (r - mu)**2
    shocks = r2
    ind = 1*((r - mu)<0)
        
    for i in range(g0.shape[0]):
        s0_0 = np.sum(np.flipud(bpow0[:i])*shocks[:i,:])
        s0_1 = np.sum(np.flipud(bpow1[:i])*shocks[:i,:])
        s1_0 = np.sum(np.flipud(bpow0[:i])*shocks[:i,:]*ind[:i,:])
        s1_1 = np.sum(np.flipud(bpow1[:i])*shocks[:i,:]*ind[:i,:])
        g0[i] = (1-a0-b0-(gamma0/2))/(1-b0) + a0*s0_0 + gamma0*s1_0
        g1[i] = (1-a1-b1-(gamma1/2))/(1-b1) + a1*s0_1 + gamma1*s1_1
    
    f0 = (1/np.sqrt(2*np.pi*g0))*np.exp(-1*(r2/(2*g0)))
    f1 = (1/np.sqrt(2*np.pi*g1))*np.exp(-1*(r2/(2*g1)))
    
    g0 = pd.DataFrame(g0,index=ret.index)
    g1 = pd.DataFrame(g1,index=ret.index)
    d_tau = pd.DataFrame(np.zeros(g0.shape),index=ret.index)
    return f0,f1,g0,g1,d_tau
    
def plotStep(p,v0,v1,fore,d_tau,smooth,iteration=None,ret=None):    
    pt0 = fore[:,0][:,np.newaxis]*v0.values
    pt1 = fore[:,1][:,np.newaxis]*v1.values
    vT = pd.DataFrame(pt0+pt1,index=v0.index,columns=['volatility'])
    plt.figure()
    # Plot the total volatility
    ax1 = plt.subplot(211)
    if iteration: 
        plt.title('Total volatility at iteration {}'.format(iteration))
    else: plt.title('Total volatility')
    if ret is not None:
        ax1.plot(
            np.sqrt(252*(ret**2)),
            linewidth=0.5,
            color='orange')
    ax1.plot(
        np.sqrt(252*vT),
        linewidth = 0.5,
        color = 'navy')
    ax1.plot(
        np.sqrt(252*d_tau),
        linewidth=1,
        color='firebrick')
    
    ax1.autoscale(enable=True, axis='x', tight=True)
    ax1.minorticks_on()
    ax1.set_ybound(lower=0,upper=100)
    # Plot smoothed probabilities
    ax2 = plt.subplot(212,sharex=ax1)
    s = pd.DataFrame(smooth[:,1],index=vT.index)
    ax2.plot(
        s,
        linewidth=0.5,
        color = 'firebrick')
    plt.tight_layout()
    plt.show()

########## EMA Function ##########
def ema(x0,args0,llfunc,bounds,vd,cols,save='',rest=True,options=None):
    """Expectation Maximisation Algorithm by Dempster (1977)
    options = {'eps':1e-6,'maxiter':100,'disp':False,'verbose':True}
    """
    # General options
    default = {'eps':1e-5,'maxiter':25,'disp':False,
                    'verbose':True,'plot':False}
    if options is not None:
        for i in options: 
            default[i] = options[i]
    v = default['verbose']
    
    # Set up the bounds
    if rest: 
        bnds,_ = bounds()
        step = StepFull(maximum=0.5,rest=True)
    else: 
        _,bnds = bounds()
        step = StepFull(maximum=0.5,rest=False)
    
    # Set up of items to save:
    res = pd.DataFrame(np.zeros((1,len(cols))),index = [0],columns=cols)
    res.loc[0,cols[0]:cols[-2]] = x0
    res.loc[0,cols[-1]] = 5e4
    llf_list = []
    param_list = [x0]
    niter = 5
    resX = x0
    for it in range(default['maxiter']):
        if v: print('- - - Iteration {} - - -'.format(it))
        # Expectation step
        f0,f1,v0,v1,d_tau = vd(resX,args0)
        inf,fore = filterProbability(resX[-2],resX[-1],f0,f1)
        smooth = smoothProbability(resX[-2],resX[-1],fore,inf)
        if default['plot']: plotStep(x0,v0,v1,fore,d_tau,smooth,iteration=it)
        argsE = args0
        argsE.extend([smooth[:,1][:,np.newaxis]])
        if it == 0: llf_list.extend([llfunc(x0,argsE)])
        
        result = minimize(
            llfunc,resX,
            args = argsE, 
            bounds = bnds, 
            jac = '2-point',
            options={'disp':default['disp'],
                        'ftol':1e-19,
                        'eps':1e-10,
                        'gtol':1,
                        'maxls': 30,
                        'maxiter':100})
        print(result)
        print("Iteration start: \n",resX)
        print("Iteration X: \n",result.x)
        print("Derivative: \n",result.jac)
        print("P00: ",result.x[-2])
        print("P11: ",result.x[-1])
        resX = np.copy(result.x)
        #jac = result.jac[1]
        #if np.abs(jac)>100.0: 
        #    resX[1] += -np.sign(jac)*0.0001
        #    if resX[2]<0: resX[2] = 1e-5
        #elif np.abs(jac)>50.0:
        #    resX[2] += -np.sign(jac)*0.00001
        #    if resX[2]<0: resX[2] = 1e-5
        #jac = result.jac[2]
        #if np.abs(jac)>50.0: 
        #    resX[2] += -(jac/1000)
        #    if resX[2]<0: resX[2] = 1e-5
        #elif np.abs(jac)>15.0:
        #    resX[2] += -np.sign(jac)*0.000001
        #    if resX[2]<0: resX[2] = 1e-5
        #
        #if result.jac[-1]>0:resX[-1]=0.99
        #if result.jac[-2]>0:resX[-2]=0.99
        print("EDIT: \n",resX)
        res.loc[it,cols[0]:cols[-2]] = result.x
        res.loc[it,cols[-1]] = result.fun
        llf_list.extend([result.fun])
        param_list.extend([result.x])
        print("LLF List: ",llf_list)
        print("LLF Direct: ",llfHaasUncondDirect(result.x,args0))
        delta = llf_list[-2]/llf_list[-1]-1
        if delta < default['eps'] and delta>0:
            print(result)
            break
        
    if default['plot']: 
        p = param_list[-1]
        f0,f1,v0,v1,d_tau = vd(p,args0)
        inf,fore = filterProbability(p[-2],p[-1],f0,f1)
        smooth = smoothProbability(p[-2],p[-1],fore,inf)
        plt.figure()
        plt.title('Log-Likelihood Development')
        plt.plot(llf_list,linewidth=1,color='navy')
        plotStep(x0,v0,v1,fore,d_tau,smooth,iteration=it,ret=args0[1])    
    
    if save: res.to_csv(save)
    return llf_list[-1],param_list[-1]

if __name__ == "__main__":
    ########## Implementation ##########
    ret, mvs = data.dataInput()
    K = tset.midasLags()
    # EMA - applied first to get some starting values, but not continued
    uncond = False
    # Use of the direct log-likelihood estimation as it works
    restricted = False
    unrestricted = False
    full = True
    garch11 = False
    # Results of the original GM-Model -> Use of w1, w2 as starting values
    stationary_r = pd.read_csv('results/res_rest.csv',index_col=0)
    stationary_u = pd.read_csv('results/res_unrest.csv',index_col=0)
    
    # Implementation of a unconditional variance switch between regimes
    if uncond:
        s_start = [0.03,2e-3,1.5,3e-2,0.8,1e-5,-0.26,1,1.437,0.7,0.7]
        s1 = 'markovResults/EMA_'
        savesR = [s1+'{}_R.csv'.format(i) for i in mvs.columns]
        savesU = [s1+'{}_U.csv'.format(i) for i in mvs.columns]
        c_uncond = ['mu','a0_0','a0_1','a','b','g','theta',
                        'w1','w2','p0','p1','LLF']
        resR = pd.DataFrame(np.zeros((mvs.shape[1],len(c_uncond))),
                                index=mvs.columns,columns=c_uncond)
        unresR = resR.copy(deep=True)
        
        for item,macro in enumerate(mvs.columns):
            print("############ Starting {} ############".format(macro))
            args0 = [K,ret,mvs[macro]]
            print('- - - - - - - Restricted Case - - - - - - -')
            s_start[6:9] = stationary_r.loc[macro,'theta':'w2']
            loglikR,xR = ema(s_start,args0,
                            llfHaasUncond,boundsUncond,volatilityDensityUncond,
                            c_uncond,save=savesR[item],rest=True,options=None)
            resR.loc[macro,'mu':'p1'] = xR
            resR.loc[macro,'LLF'] = loglikR
            resR.to_csv('markovResults/EMA_restricted.csv')
            
            print('- - - - - - - Unestricted Case - - - - - - -')
            start_u = xR
            start_u[-2],start_u[-1] = 0.7,0.7
            loglikU,xU = ema(xR,args0,
                            llfHaasUncond,boundsUncond,volatilityDensityUncond,
                            c_uncond,save=savesU[item],rest=False,options=None)
            unresR.loc[macro,'mu':'p1'] = xU
            unresR.loc[macro,'LLF'] = loglikU
            unresR.to_csv('markovResults/EMA_unrestricted.csv')
            
            print("###### Results for {} ######".format(macro))
            print('Restricted: \n {}'.format(resR.loc[macro,:]))
            print('Unestricted: \n {}'.format(unresR.loc[macro,:]))
    
    # Compute direct minimisation of the LLF
    if restricted:
        bnds_r,bnds_u = boundsUncond()
        s1 = 'markovResults/EMA_direct_'
        c_uncond = ['mu','a0_0','a0_1','a','b','g','theta',
                        'w1','w2','p0','p1','LLF']
        # Results & standard errors frame
        resR = pd.DataFrame(np.zeros((mvs.shape[1],len(c_uncond))),
                                index=mvs.columns,columns=c_uncond)
        unresR = resR.copy(deep=True)
        resSE = pd.DataFrame(np.zeros((mvs.shape[1],len(c_uncond)-1)),
                                index=mvs.columns,columns=c_uncond[:-1])
        unresSE = resSE.copy(deep = True)
        
        resR = pd.read_csv('markovResults/EMA_restricted_direct.csv',
                            index_col=0)
        unresR = pd.read_csv('markovResults/EMA_unrestricted_direct.csv',
                            index_col=0)
        
        # Get EMA values
        a = 'markovResults/EMA_restricted.csv'
        b = 'markovResults/EMA_unrestricted.csv'
        ema_rest=pd.read_csv(a,index_col=0)
        ema_unrest=pd.read_csv(b,index_col=0)
        
        x0 = [0.03,1e-8,1,0.015,0.91,0.07,0.1,1,5,0.97,0.97]
        
        for item,macro in enumerate(mvs.columns):
            plot_direct = False
            print("############# Starting {} ############".format(macro))
            print('- - - - - - - Restricted Case - - - - - - -')
            # Setup
            args = [K,ret,mvs[macro]]
            # Restricted case
            r_start = ema_rest.iloc[item,:-1]
            r_start[1] = 0.001
            r_start[2] = 0.1
            r_start[-2] = 0.98
            r_start[-1] = 0.98
            bnds_r,_ = boundsUncond()
            min_dict = {'args':args, 
                        'bounds':bnds_r, 
                        'jac':'2-point',
                        'options':{'disp':2,
                                    'ftol':1e-20,
                                    'eps':1e-6,
                                    'gtol':1,
                                    'maxls': 25,
                                    'maxiter':220}}
            step = StepUncondR(maximum=2)
            """
            result_r = basinhopping(llfHaasUncondDirect,r_start,T=0,\
                                    minimizer_kwargs=min_dict,\
                                    niter_success = 3,
                                    niter=15,
                                    take_step=step,
                                    callback=bs.print_fun)
                                    
            """
            result_r = minimize(llfHaasUncondDirect,r_start,
                                    args = args, 
                                    bounds = bnds_r, 
                                    jac = '2-point',
                                    options={'disp':2,
                                                'ftol':1e-10,
                                                'eps':1e-6,
                                                'gtol':3,
                                                'maxls': 40,
                                                'maxiter':200})
            
            print(result_r.x)
            resR.loc[macro,'mu':'p1'] = result_r.x
            resR.loc[macro,'LLF'] = result_r.fun
            resR.to_csv('markovResults/EMA_restricted_direct_X.csv')
            
            if plot_direct:
                f0,f1,v0,v1,d_tau,_ = volatilityDensityUncond(p,args)
                inf,fore = filterProbability(p[-2],p[-1],f0,f1)
                smooth = smoothProbability(p[-2],p[-1],fore,inf)
                plotStep(p,v0,v1,fore,d_tau,smooth)
                
    # Unrestricted case of direct log-likelihood optimisation
    if unrestricted:
        bnds_r,bnds_u = boundsUncond()
        s1 = 'markovResults/EMA_direct_'
        c_uncond = ['mu','a0_0','a0_1','a','b','g','theta',
                        'w1','w2','p0','p1','LLF']
        # Results & standard errors frame
        resR = pd.DataFrame(np.zeros((mvs.shape[1],len(c_uncond))),
                                index=mvs.columns,columns=c_uncond)
        unresR = resR.copy(deep=True)
        resSE = pd.DataFrame(np.zeros((mvs.shape[1],len(c_uncond)-1)),
                                index=mvs.columns,columns=c_uncond[:-1])
        unresSE = resSE.copy(deep = True)
        
        resR = pd.read_csv('markovResults/EMA_restricted_direct.csv',
                            index_col=0)
        unresR = pd.read_csv('markovResults/EMA_unrestricted_direct.csv',
                            index_col=0)
        
        # Get EMA values
        a = 'markovResults/EMA_restricted.csv'
        b = 'markovResults/EMA_unrestricted.csv'
        ema_rest=pd.read_csv(a,index_col=0)
        ema_unrest=pd.read_csv(b,index_col=0)
        
        for item,macro in enumerate(['UNEMP','CONS']):#enumerate(mvs.columns):
            plot_direct = False
            
            print("############# Starting {} ############".format(macro))
            args = [K,ret,mvs[macro]]
            ## Restricted case
            print('- - - - - - - Unestricted Case - - - - - - -')
            # Unrestricted case
            u_start = resR.loc[macro,:'p1']
            print(u_start)
            min_dict = {'args':args, 
                        'bounds':bnds_u, 
                        'jac':'2-point',
                        'options':{'disp':2,
                                    'ftol':1e-20,
                                    'eps':1e-6,
                                    'gtol':1,
                                    'maxls': 25,
                                    'maxiter':220}}
            step = StepUncondUR(maximum=3)
            result_u = basinhopping(llfHaasUncondDirect,u_start,T=0,\
                                    minimizer_kwargs=min_dict,\
                                    niter_success = 3,
                                    niter=15,
                                    take_step=step,
                                    callback=bs.print_fun)
            print(result_u)
            unresR.loc[macro,'mu':'p1'] = result_u.x
            unresR.loc[macro,'LLF'] = result_u.fun
            unresR.to_csv('markovResults/EMA_unrestricted_direct_X.csv')
            
            if plot_direct:
                f0,f1,v0,v1,d_tau,_ = volatilityDensityUncond(p,args)
                inf,fore = filterProbability(p[-2],p[-1],f0,f1)
                smooth = smoothProbability(p[-2],p[-1],fore,inf)
                plotStep(p,v0,v1,fore,d_tau,smooth)

    # GARCH(1,1) version
    if garch11:
        bnds = boundsGARCH()
        first = ret.index.get_loc('01-01-1973',method='bfill')
        args = ret.iloc[first:]
        x0 = [0.03,0.01,0.1,0.05,0.90,0.035,0.95,0.95]
        #options = np.arange(0.1,1,0.1)
        #for i in options:
        #    x0[2] = i
        #    print("Choice: ",i," - - ",llfgarch11(x0,args))
        #print(llfgarch11(x0,args))
        min_dict = {'args':args, 
                    'bounds':bnds, 
                    'jac':'2-point',
                    'options':{'disp':2,
                                'ftol':1e-5,
                                'eps':1e-6,
                                'gtol':1,
                                'maxls': 25,
                                'maxiter':220}}
        step = StepGarch(maximum=3)
        result = basinhopping(llfgarch11,x0,T=0,\
                                minimizer_kwargs=min_dict,\
                                niter_success = 3,
                                niter = 15,
                                take_step = step,
                                callback = bs.print_fun)
        print(result)
        result = minimize(llfgarch11,x0,
                                args = args, 
                                bounds = bnds, 
                                jac = '2-point',
                                options={'disp':2,
                                            'ftol':1e-10,
                                            'eps':1e-6,
                                            'gtol':1,
                                            'maxls': 40,
                                            'maxiter':200})
        print(result)
    
    # Compute direct minimisation of the LLF
    if full:
        lb_res =  [-1,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,-1e2,-1e2,-1e15, 1, 1, 0, 0]
        ub_res =  [ 1,1e15,1e15,   1,   1,   1,   1,   2,   2, 1e15, 1,1e15,1,1]
        feasible = [True for i in range(len(lb_res))]
        # create bounds objects
        bnds_r = Bounds(lb_res,ub_res,keep_feasible=feasible)
        s1 = 'markovResults/EMA_direct_'
        c_uncond = ['mu','a0_0','a0_1','a1_0','a1_1','b0','b1','g0',
                        'g1','theta','w1','w2','p0','p1','LLF']
        # Results & standard errors frame
        resR = pd.DataFrame(np.zeros((mvs.shape[1],len(c_uncond))),
                                index=mvs.columns,columns=c_uncond)
        unresR = resR.copy(deep=True)
        resSE = pd.DataFrame(np.zeros((mvs.shape[1],len(c_uncond)-1)),
                                index=mvs.columns,columns=c_uncond[:-1])
        unresSE = resSE.copy(deep = True)
        
        # Get EMA values
        a = 'markovResults/EMA_restricted_direct.csv'
        b = 'markovResults/EMA_unrestricted_direct.csv'
        ema_rest=pd.read_csv(a,index_col=0)
        ema_unrest=pd.read_csv(b,index_col=0)
        
        x0 = [0.03,0.15,1,0.05,0.1,0.8,0.8,0.1,0.05,0.7,1,5,0.8,0.8]
        
        class StepFull(object):
            def __init__(self, stepsize=1,maximum=2,rest=True,maxTry=1000):
                self.stepsize = stepsize
                self.max = maximum
                self.rest = rest
                self.maxTry = maxTry
            def __call__(self, x):
                # Order: a0,b0,a1,b1,m,theta,w1,w2,p00,p11
                s = self.stepsize
                prior = x
                for i in range(self.maxTry):
                    x[-1] += np.random.uniform(-0.05*s, 0.05*s)
                    x[-2] += np.random.uniform(-0.05*s, 0.05*s)
                    x[-4]  = 1  # w1
                    x[-3]  += np.random.uniform(1-x[-3], self.max*s)  # w2
                    if x[-4]>0 and x[-3]>1: break
                    else: 
                        x[-4]=prior[-4]
                        x[-3]=prior[-3]
                    if x[1] < 0: x[1] = 1e-10
                    if x[2] < 0: x[2] = 1
                print("Attempt: ", x)
                return np.array(x)
        
        for item,macro in enumerate(mvs.columns):
            plot_direct = False
            print("############# Starting {} ############".format(macro))
            print('- - - - - - - Restricted Case - - - - - - -')
            # Setup
            args = [K,ret,mvs[macro]]
            min_dict = {'args':args, 
                        'bounds':bnds_r, 
                        'jac':'2-point',
                        'options':{'disp':2,
                                    'ftol':1e-10,
                                    'eps':1e-6,
                                    'gtol':1,
                                    'maxls': 25,
                                    'maxiter':220}}
            step = StepFull(maximum=2)
            result_r = basinhopping(llfHaasFullDirect,x0,T=0,\
                                    minimizer_kwargs=min_dict,\
                                    niter_success = 3,
                                    niter=15,
                                    take_step=step,
                                    callback=bs.print_fun)
            
            print(result_r.x)
            resR.loc[macro,'mu':'p1'] = result_r.x
            resR.loc[macro,'LLF'] = result_r.fun
            resR.to_csv('markovResults/EMA_restricted_direct_THETA.csv')
            
            if plot_direct:
                f0,f1,v0,v1,d_tau,_ = volatilityDensityUncond(p,args)
                inf,fore = filterProbability(p[-2],p[-1],f0,f1)
                smooth = smoothProbability(p[-2],p[-1],fore,inf)
                plotStep(p,v0,v1,fore,d_tau,smooth)