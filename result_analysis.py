import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as ss
from matplotlib import pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec
import seaborn as sns
import numdifftools as ndt

import likelihood_functions as llf
import ema_actual as ema
import data as data

""" ANALYSIS METHODS"""
""" This file outlines the statistical tests and analysis that are conducted for the 
    GARCH-MIDAS model. It includes: series recreation,t-test of significance, 
    identification simulation, BIC criterion, variance ratios
"""

def standardErrors(p,args,step=1e-4,v=False):
    h = ndt.Hessian(llf.simpleGM_asym,step=step,method='central')(p,args)
    T = args[1].shape[0]
    inv = np.linalg.inv(h/T)
    s_var = np.diag(inv/T)
    return np.sqrt(s_var)

def standardErrorsMS(p,args,step=1e-8,v=False):
    h = ndt.Hessian(ema.llfHaasUncondDirect,step=step,method='central')(p,args)
    T = args[1].shape[0]
    inv = np.linalg.inv(h/T)
    s_var = np.diag(inv/T)
    return np.sqrt(s_var)

def resultsFrame(rest,unrest):
    """Creates a new results DataFrame out of the existing restricted and
        unrestricted cases"""
    ix_orig,col_orig = rest.index,rest.columns
    shape = (2*ix_orig.shape[0],col_orig.shape[0]+3)
    #New frame
    x = pd.DataFrame(np.zeros(shape))
    #Double the index (rest & unrest case)
    x.index = sum([[i,i] for i in ix_orig],[])
    c = col_orig.tolist()
    #For next statistics
    c.append('BIC')
    c.append('VAR(X)')
    c.append('Identification')
    x.columns=c
    #Add all the old parameter estimates to new results frame
    for i in range(0,rest.shape[0]):
        x.iloc[2*i,:-2] = rest.iloc[i,:]
        x.iloc[2*i+1,:-2] = unrest.iloc[i,:]
    #x.loc[:,"Log-Likelihood"] *= -1
    return x

def recreate_asym(p,data):
    """ Recreate the tau and g series based on the asymmetric GARCH-MIDAS
        as in Conrad & Loch (2015)
    """
    mu,a1,b1,gamma,m,theta,w1,w2 = p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]
    K,ret,macro = data[0],data[1],data[2]
    
    # Step 1 - create tau 
    first_tau = ret.index.get_loc(macro.index[K],method='bfill')
    midas = llf.lagMatrix(macro,K).values @ llf.betaWeight(w1,w2,K)
    tau = pd.DataFrame(np.exp(m+(theta*midas)),index=macro.index)
    d_tau = tau.loc[ret.index.to_period('Q').to_timestamp()].values
    
    # Step 2 - determine g
    r2 = (ret.iloc[first_tau:,:].values - mu)**2
    shocks = r2/d_tau
    ind = 1*((ret.iloc[first_tau:,:].values - mu)<0)
    g = np.zeros(d_tau.shape)
    g[0] = 1
    for i in range(1,g.shape[0]):
        g[i] = (1-a1-b1-(gamma/2)) + (a1+gamma*ind[i-1])*shocks[i-1] + b1*g[i-1]
    
    g = pd.DataFrame(np.hstack([g,d_tau]),
                index=ret.index[first_tau:],columns=['g','d_tau'])
    return g,tau

def recreate_arch(p,data):
    """ Recreate the tau and g series based on the ARCH specification of 
        the asymmetric GARCH-MIDAS of Conrad & Loch (2015)
    """
    mu,a1,b1,gamma,m,theta,w1,w2 = p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]
    K,ret,macro = data[0],data[1],data[2]
    
    # Step 1 - create tau 
    first_tau = ret.index.get_loc(macro.index[K],method='bfill')
    midas = lagMatrix(macro,K).values @ betaWeight(w1,w2,K)
    tau = pd.DataFrame(np.exp(m+(theta*midas)),index=macro.index)
    d_tau = tau.loc[ret.index.to_period('Q').to_timestamp()].values
    
    # Step 2 - determine g
    r2 = (ret.iloc[first_tau:,:].values - mu)**2
    shocks = r2/d_tau
    ind = 1*((ret.iloc[first_tau:,:].values - mu)<0)
    g = np.zeros(d_tau.shape)
    g[0] = 1
    for i in range(1,g.shape[0]):
        g[i] = (1-a1-b1-(gamma/2)) + (a1+gamma*ind[i-1])*shocks[i-1] + b1*g[i-1]
    
    g = pd.DataFrame(np.hstack([g,d_tau]),
                index=ret.index[first_tau:],columns=['g','d_tau'])
    return g,tau

def recreate_ms(p,args):
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
    inf,fore = ema.filterProbability(p00,p11,f0,f1)
    smooth = ema.smoothProbability(p00,p11,fore,inf)
    pt0 = fore[:,0][:,np.newaxis]*v0
    pt1 = fore[:,1][:,np.newaxis]*v1
    v = pd.DataFrame(pt0+pt1,index=ret.index[first_tau:])
    inf = pd.DataFrame(inf[:,1],index=ret.index[first_tau:])
    return v,inf,daily_tau

def varRatio(g):
    """Variance ratio to quantify long-term variance account of volatility"""
    t = g.iloc[:,1].values
    g = g.iloc[:,0].values
    return 100*np.var(np.log(t))/np.var(np.log(t*g))

def varRatioRV(g,grv):
    """Variance ratio to quantify long-term variance account of volatility"""
    t = g.iloc[:,1].values
    trv = grv.iloc[:,1].values
    g = grv.iloc[:,0].values
    return 100*np.var(np.log(t))/np.var(np.log(trv*g))
  
def tstat(val,se):
    """Basic t-statistic and p-value for the component in question"""
    stat = np.abs(val/se)
    pval = 1-ss.norm.cdf(stat)
    return stat,pval
    
def tstat_df(est,se):
    """Calculate the t-statistics and p-values for a dataframe of estimates,
         given standard errors"""
    p_df = pd.DataFrame(np.zeros(est.shape),
                index=est.index,columns=est.columns)
    st_df = pd.DataFrame(np.zeros(est.shape),
                index=est.index,columns=est.columns)
    
    for reg in est.index:
        for var in est.columns:
            st_df.loc[reg,var] = np.abs(est.loc[reg,var] / se.loc[reg,var])
            p_df.loc[reg,var] = 1-ss.norm.cdf(st_df.loc[reg,var])
        
    return st_df,p_df

# Evaluation criteria
def bic(p_num,loglike,T):
    """Schwarz Information Criterion"""
    return (p_num*np.log(T) + 2*loglike)/T
    
def lrt_weight(ll_rest,ll_unrest):
    """Likelihood ratio test of the restrictgold vs. the unrestricted weights"""
    lrt = pd.DataFrame(np.zeros((ll_rest.shape[0],2)),
                columns=['stat','p-val'],index=ll_rest.index)
    for reg in lrt.index:
        lrt.loc[reg,'stat'] = -2*(ll_unrest.loc[reg] - ll_rest.loc[reg])
        lrt.loc[reg,'p-val'] = ss.chi2.sf(lrt.loc[reg,'stat'],1)
    return lrt

def desired_df(lrt,rest,unrest,ident):
    """Function to return dataframe of the models that 
            will be used in the forecasting exercise"""
    result = pd.DataFrame(columns=rest.columns)
    for i,reg in enumerate(lrt.index.tolist()):
        # If llf different enough AND Identified - take the unrestricted
            if lrt.loc[reg,'p-val']<0.1:
                if ident.iloc[(2*i)+1]: result=result.append(unrest.loc[reg,:])
            else:
                if ident.iloc[(2*i)]: result=result.append(rest.loc[reg,:])
            
    return result
    
def identified(coeff,se,T):
    """Andrews & Cheng Identification Category Selection"""
    threshold = np.sqrt(np.log(T))
    statistic = np.abs(coeff / se)
    if statistic > threshold: return True
    else: return False


# Update the plot fonts and allow usage of LaTeX
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

def plotAllWeights_s(w1_r,w2_r,w1_u,w2_u,K,name,block=False,save=''):
    """Function to create a 5x2 subplot of all the passed weights"""
    fig, ax = plt.subplots()
    fig.set_size_inches(5,3)
    print(ax)
    f={'fontsize': 12, 'fontweight': 'medium'}
    r = llf.betaWeight(w1_r,w2_r,K)
    u = llf.betaWeight(w1_u,w2_u,K)
    lr = r'$\omega_1$={:01.0f}, $\omega_2$={:.2f}'.format(w1_r,w2_r)
    lu = r'$\omega_1$={:.2f}, $\omega_2$={:.2f}'.format(w1_u,w2_u)
    ax.plot(np.arange(1,K),r[:-1],label=lr,
                linewidth=1.0,linestyle='solid',color='navy')
    ax.plot(np.arange(1,K),u[:-1],label=lu,
                linewidth=1.0,linestyle='dashed',color='firebrick')
    ax.set_title(name,
                fontdict={'fontsize': 16, 'fontweight': 'medium'})
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_ylabel('Optimal Weights',fontdict=f)
    ax.set_xlabel('Lag Months',fontdict=f)
    ax.set_ybound(lower=0)
    ax.set_xbound(lower=0)
    ax.minorticks_on()
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save: plt.savefig(save, bbox_inches='tight')
    plt.show(block=block)
    plt.close()
    
def plotAllWeights(w1_r,w2_r,w1_u,w2_u,K,names,block=False,save=''):
    """Function to create a 5x2 subplot of all the passed weights"""
    l = len(names)
    if np.remainder(l,3) == 1: unev,r=1,1
    elif np.remainder(l,3) == 2:unev,r=2,1
    else:unev,r=0,0
    rows = np.floor_divide(l,3)+r
    gs = gridspec.GridSpec(rows, 6)
    ax_lst = []
    for i in range(np.floor_divide(l,3)):
        ax1 = plt.subplot(gs[i, 0:2])
        ax2 = plt.subplot(gs[i, 2:4])
        ax3 = plt.subplot(gs[i, 4:])
        ax_lst.extend([ax1,ax2,ax3])
    if unev==1:
        ax = plt.subplot(gs[rows-1, 2:4])
        ax_lst.extend([ax])
    if unev==2:
        ax1 = plt.subplot(gs[rows-1, 1:3])
        ax2 = plt.subplot(gs[rows-1, 3:5])
        ax_lst.extend([ax1,ax2])
    fig = plt.gcf()
    fig.set_size_inches(18,14)
    gs.tight_layout(fig)
    f={'fontsize': 10, 'fontweight': 'medium'}
    
    for i in range(len(names)):
        r = llf.betaWeight(w1_r[i],w2_r[i],K)
        u = llf.betaWeight(w1_u[i],w2_u[i],K)
        lr = r'$\omega_1$={:01.0f}, $\omega_2$={:.2f}'.format(w1_r[i],w2_r[i])
        lu = r'$\omega_1$={:.2f}, $\omega_2$={:.2f}'.format(w1_u[i],w2_u[i])
        ax_lst[i].plot(np.arange(1,K),r[:-1],label=lr,
                    linewidth=0.8,linestyle='solid',color='navy')
        ax_lst[i].plot(np.arange(1,K),u[:-1],label=lu,
                    linewidth=0.8,linestyle='dashed',color='firebrick')
        ax_lst[i].set_title(names[i],
                    fontdict={'fontsize': 16, 'fontweight': 'medium'})
        ax_lst[i].autoscale(enable=True, axis='x', tight=True)
        ax_lst[i].tick_params(axis='both', which='major', labelsize=8)
        ax_lst[i].set_ylabel('Optimal Weights',fontdict=f)
        ax_lst[i].set_xlabel('Lag Months',fontdict=f)
        ax_lst[i].set_ybound(lower=0)
        ax_lst[i].set_xbound(lower=0)
        ax_lst[i].minorticks_on()
        ax_lst[i].legend(fontsize=7)
    plt.tight_layout()
    gs.tight_layout(fig)
    if save: plt.savefig(save, bbox_inches='tight')
    plt.show(block=block)
    plt.close()

def plotAllAnnVol(g_list,names,block=False,save=''):
    """Function to create a 5x2 subplot of all the passed volatilities"""
    r_s,r_e = data.recessionSE()
    l = len(names)
    if np.remainder(l,3) == 1: unev,r=1,1
    elif np.remainder(l,3) == 2:unev,r=2,1
    else:unev,r=0,0
    rows = np.floor_divide(l,3)+r
    gs = gridspec.GridSpec(rows, 6)
    ax_lst = []
    for i in range(np.floor_divide(l,3)):
        ax1 = plt.subplot(gs[i, 0:2])
        ax2 = plt.subplot(gs[i, 2:4])
        ax3 = plt.subplot(gs[i, 4:])
        ax_lst.extend([ax1,ax2,ax3])
    if unev==1:
        ax = plt.subplot(gs[rows-1, 2:4])
        ax_lst.extend([ax])
    if unev==2:
        ax1 = plt.subplot(gs[rows-1, 1:3])
        ax2 = plt.subplot(gs[rows-1, 3:5])
        ax_lst.extend([ax1,ax2])
    fig = plt.gcf()
    fig.set_size_inches(18,14)
    gs.tight_layout(fig)
    f={'fontsize': 10, 'fontweight': 'medium'}
    annVol_list=[]
    for g in g_list:
        annVol = np.sqrt(252*g.iloc[:,0]*g.iloc[:,1])
        annTau = np.sqrt(252*g.iloc[:,1])
        annVol_list.append([annVol,annTau])
    
    for i in range(len(names)):
        at = r'Annualised $\tau_t$'
        av = r'Annualised $\tau_{t}\:g_{i,t}$'
        ax_lst[i].plot(annVol_list[i][0],label=av,
                        linewidth=0.5,linestyle='solid',color='navy')
        ax_lst[i].plot(annVol_list[i][1],label=at,
                        linewidth=1.5,linestyle='solid',color='firebrick')
        for j in zip(r_s,r_e):
            ax_lst[i].axvspan(xmin=j[0], xmax=j[1],color='gainsboro')
        ax_lst[i].set_title(names[i],
                    fontdict={'fontsize': 16, 'fontweight': 'medium'})
        ax_lst[i].autoscale(enable=True, axis='x', tight=True)
        ax_lst[i].tick_params(axis='both', which='major', labelsize=8)
        ax_lst[i].set_ylabel('Ann. Volatility',fontdict=f)
        ax_lst[i].set_xlabel('Time',fontdict=f)
        ax_lst[i].set_ybound(lower=0,upper=100)
        ax_lst[i].minorticks_on()
        ax_lst[i].legend(fontsize=7)
    plt.tight_layout()
    gs.tight_layout(fig)
    if save: plt.savefig(save, bbox_inches='tight')
    plt.show(block=block)
    plt.close()

def plotMSAnnVol(v_list,t_list,p_list,names,block=False,save=''):
    """Function to create a 5x2 subplot of all the passed volatilities"""
    r_s,r_e = data.recessionSE()
    l = len(names)
    if np.remainder(l,3) == 1: unev,r=1,1
    elif np.remainder(l,3) == 2:unev,r=2,1
    else:unev,r=0,0
    rows = np.floor_divide(l,3)+r
    gs = gridspec.GridSpec(3*rows, 6)
    ax_lst = []
    for i in range(np.floor_divide(l,3)):
        ax1 = plt.subplot(gs[3*i:3*i+2, 0:2])
        ax1_p=plt.subplot(gs[3*i+2:3*i+3,0:2])
        ax2 = plt.subplot(gs[3*i:3*i+2, 2:4])
        ax2_p=plt.subplot(gs[3*i+2:3*i+3,2:4])
        ax3 = plt.subplot(gs[3*i:3*i+2, 4:])
        ax3_p=plt.subplot(gs[3*i+2:3*i+3,4:])
        ax_lst.extend([ax1,ax1_p,ax2,ax2_p,ax3,ax3_p])
    if unev==1:
        ax = plt.subplot(gs[3*rows-3:3*rows-1, 2:4])
        ax_p=plt.subplot(gs[3*rows-1:3*rows, 2:4])
        ax_lst.extend([ax,ax_p])
    if unev==2:
        ax1 = plt.subplot(gs[3*rows-3:3*rows-1, 1:3])
        ax1_p=plt.subplot(gs[3*rows-1:3*rows,0:2])
        ax2 = plt.subplot(gs[3*rows-3:3*rows-1, 3:5])
        ax1_p=plt.subplot(gs[3*rows-1:3*rows,0:2])
        ax_lst.extend([ax1,ax1_p,ax2,ax2_p])
    fig = plt.gcf()
    fig.set_size_inches(18.6,27)
    gs.tight_layout(fig)
    f={'fontsize': 18, 'fontweight': 'medium'}
    annVol_list=[]
    for i,v in enumerate(v_list):
        annVol = np.sqrt(252*v)
        annTau = np.sqrt(252*t_list[i])
        annVol_list.append([annVol,annTau])
    
    for i in range(2*len(names)):
        if np.mod(i,2) == 0:
            at = r'Annualised $\tau_t$'
            av = r'Annualised $\tau_{t}\:g_{i,t}$'
            prob = r'P$(s_{i,t}=1|R_{i-1,t})$'
            ax_lst[i].plot(annVol_list[int(i/2)][0],label=av,
                            linewidth=1.0,linestyle='solid',color='navy')
            ax_lst[i].plot(annVol_list[int(i/2)][1],label=at,
                            linewidth=2.0,linestyle='solid',color='firebrick')
            for j in zip(r_s,r_e):
                ax_lst[i].axvspan(xmin=j[0], xmax=j[1],color='gainsboro')
            ax_lst[i].set_title(names[int(i/2)],
                        fontdict={'fontsize': 24, 'fontweight': 'medium'})
            ax_lst[i].autoscale(enable=True, axis='x', tight=True)
            ax_lst[i].tick_params(axis='both', which='major', labelsize=8)
            ax_lst[i].set_ylabel('Ann. Volatility',fontdict=f)
            ax_lst[i].set_xlabel('Time',fontdict=f)
            ax_lst[i].set_ybound(lower=0,upper=100)
            ax_lst[i].minorticks_on()
            ax_lst[i].legend(fontsize=7)
        else: 
            p = r'Posterior Probability'
            ax_lst[i].plot(p_list[int(i/2)],label=p,
                            linewidth=1,linestyle='solid',color='firebrick')
            for j in zip(r_s,r_e):
                ax_lst[i].axvspan(xmin=j[0], xmax=j[1],color='gainsboro')
            ax_lst[i].autoscale(enable=True, axis='x', tight=True)
            ax_lst[i].tick_params(axis='both', which='major', labelsize=8)
            ax_lst[i].set_ylabel(prob,
                            fontdict={'fontsize': 14, 'fontweight': 'medium'})
            ax_lst[i].set_xlabel('Time',fontdict=f)
            ax_lst[i].set_ybound(lower=0,upper=1)
            ax_lst[i].minorticks_on()
    plt.tight_layout()
    gs.tight_layout(fig)
    if save: plt.savefig(save, bbox_inches='tight')
    plt.show(block=block)
    plt.close()

def plotMSAnnVol_s(v,t,p,name,block=False,save=''):
    """Function to create a 5x2 subplot of all the passed volatilities"""
    r_s,r_e = data.recessionSE()
    gs = gridspec.GridSpec(3, 1)
    ax_lst = []
    ax1 = plt.subplot(gs[0:2,:])
    ax2 = plt.subplot(gs[2,:])
    ax_lst.extend([ax1,ax2])
    fig = plt.gcf()
    fig.set_size_inches(9,9)
    gs.tight_layout(fig)
    f={'fontsize': 20, 'fontweight': 'medium'}
    annVol = np.sqrt(252*v)
    annTau = np.sqrt(252*t)
    at = r'Annualised $\tau_t$'
    av = r'Annualised $\tau_{t}\:g_{i,t}$'
    prob = r'P$(s_{i,t}=1|R_{i,t})$'
    p_label = r'Posterior Probability'
    ax_lst[0].plot(annVol,label=av,
                    linewidth=1.0,linestyle='solid',color='navy')
    ax_lst[0].plot(annTau,label=at,
                    linewidth=2.0,linestyle='solid',color='firebrick')
    for j in zip(r_s,r_e):
        ax_lst[0].axvspan(xmin=j[0], xmax=j[1],color='gainsboro')
    ax_lst[0].set_title(name,fontdict={'fontsize': 34, 'fontweight': 'medium'})
    ax_lst[0].autoscale(enable=True, axis='x', tight=True)
    ax_lst[0].tick_params(axis='both', which='major', labelsize=14)
    ax_lst[0].set_ylabel('Ann. Volatility',fontdict=f)
    #ax_lst[0].set_xlabel('Time',fontdict=f)
    ax_lst[0].set_ybound(lower=0,upper=100)
    ax_lst[0].minorticks_on()
    ax_lst[0].legend(fontsize=14)
    
    ax_lst[1].plot(p,label=p_label,
                    linewidth=1,linestyle='solid',color='firebrick')
    for j in zip(r_s,r_e):
        ax_lst[1].axvspan(xmin=j[0], xmax=j[1],color='gainsboro')
    ax_lst[1].autoscale(enable=True, axis='x', tight=True)
    ax_lst[1].tick_params(axis='both', which='major', labelsize=14)
    ax_lst[1].set_ylabel(prob,
                    fontdict={'fontsize': 18, 'fontweight': 'medium'})
    ax_lst[1].set_xlabel('Time',fontdict=f)
    ax_lst[1].set_ybound(lower=0,upper=1)
    ax_lst[1].minorticks_on()
    plt.tight_layout()
    gs.tight_layout(fig)
    if save: plt.savefig(save, bbox_inches='tight')
    plt.show(block=block)
    plt.close()

def plotAllInnov(inn_list,names,block=False,save=''):
    """Function to create a 5x2 subplot of all the passed innovations"""
    l = len(names)
    if np.remainder(l,3) == 1: unev,r=1,1
    elif np.remainder(l,3) == 2:unev,r=2,1
    else:unev,r=0,0
    rows = np.floor_divide(l,3)+r
    gs = gridspec.GridSpec(rows, 6)
    ax_lst = []
    for i in range(np.floor_divide(l,3)):
        ax1 = plt.subplot(gs[i, 0:2])
        ax2 = plt.subplot(gs[i, 2:4])
        ax3 = plt.subplot(gs[i, 4:])
        ax_lst.extend([ax1,ax2,ax3])
    if unev==1:
        ax = plt.subplot(gs[rows-1, 2:4])
        ax_lst.extend([ax])
    if unev==2:
        ax1 = plt.subplot(gs[rows-1, 1:3])
        ax2 = plt.subplot(gs[rows-1, 3:5])
        ax_lst.extend([ax1,ax2])
    fig = plt.gcf()
    fig.set_size_inches(18,14)
    gs.tight_layout(fig)
    f={'fontsize': 8, 'fontweight': 'medium'}
    annVol_list=[]
    bins = np.linspace(-4,4,100)
    for i in range(len(names)):
        jb,p = ss.jarque_bera(inn_list[i])
        inn = (inn_list[i]).values
        label1 = 'Skew:{:.2f},Kurt:{:.2f}\nJB: {:.2f} ({:.2f})'.format(
                    ss.skew(inn)[0],ss.kurtosis(inn,fisher=False)[0],jb,p)
        xx = np.arange(-4, +4, 0.001)                                                   
        yy = ss.norm.pdf(xx,np.mean(inn_list[i]),np.std(inn_list[i]))
        sns.distplot(inn_list[i],bins=bins,
                        kde=False,norm_hist=True,ax=ax_lst[i])
        ax_lst[i].plot(xx,yy,linewidth=1.0,linestyle='solid',color='firebrick')
        ax_lst[i].set_title(names[i],
                fontdict={'fontsize': 12, 'fontweight': 'medium'})
        ax_lst[i].autoscale(enable=True, axis='x', tight=True)
        ax_lst[i].tick_params(axis='both', which='major', labelsize=8)
        ax_lst[i].set_ylabel('Frequency',fontdict=f)
        ax_lst[i].set_xlabel('Innovation',fontdict=f)
        ax_lst[i].set_xbound(lower=-3,upper=3)
        ax_lst[i].minorticks_on()
        props = dict(boxstyle='round', facecolor='white', alpha=0.1)
        ax_lst[i].text(0.05, 0.95, label1, transform=ax_lst[i].transAxes, 
                        fontsize=7,verticalalignment='top', bbox=props)
    plt.tight_layout()
    gs.tight_layout(fig)
    if save: plt.savefig(save, bbox_inches='tight')
    plt.show(block=block)
    plt.close()