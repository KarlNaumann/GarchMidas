print('--------------------------------------------------')
import numpy as np
import pandas as pd
import sys
import warnings
from scipy.optimize import basinhopping#, minimize, Bounds, LinearConstraint,NonlinearConstraint,SR1
import numdifftools as ndt

import data
import basin_settings as bs
import likelihood_functions as llf
import latexExport as latex
import thesis_settings as tset

#Turn off warnings for final checks
if not sys.warnoptions:
    warnings.simplefilter("ignore")

save = True
files = tset.gm_results()

# Import the data that is used
ret,mvs = data.dataInput()

# Initialize the arrays for the results
columns=['mu',"a1","b1",'gamma',"m","theta","w1","w2","Log-Likelihood"]
index = mvs.columns
hess = []
res_rest = pd.DataFrame(np.zeros((mvs.columns.shape[0],9)),
                    columns=columns,index=mvs.columns)
res_unrest = pd.DataFrame(np.zeros((mvs.columns.shape[0],9)),
                    columns=columns,index=mvs.columns)
se_rest = pd.DataFrame(np.zeros((mvs.columns.shape[0],8)),
                    columns=columns[:-1],index=mvs.columns)
se_unrest = pd.DataFrame(np.zeros((mvs.columns.shape[0],8)),
                    columns=columns[:-1],index=mvs.columns)

# Basinhopping (Global minimization) setup
np.random.seed(40)
rest,unrest = bs.bounds_Simple_asym()
K = tset.midasLags()
niter = 20   
step_r = bs.StepAsym(rest=True,max=2.5)
step_u = bs.StepAsym(rest=False,max=2.5)
x0 = [0.03,0.02,0.9,0.1,0.05,0.1,1,5]
#####################################################
# Start of optimization for the regular GARCH-MIDAS-X #
print("----# BEGIN GARCH-MIDAS #----")
for macro in mvs.columns:
    print("- - - Starting Restricted {} - - -".format(macro))
    # Restricted case
    args = ([K,ret,mvs[macro]])
    rest_reg = basinhopping(llf.simpleGM_asym,x0,T=0,\
                            minimizer_kwargs=bs.min_arg(args,rest),\
                            niter=niter,
                            take_step=step_r,
                            callback=bs.print_fun)
    print(rest_reg)
    res_rest.loc[macro,'mu':'w2'] = rest_reg.x
    res_rest.loc[macro,'Log-Likelihood'] = rest_reg.fun
    hess.append(rest_reg.lowest_optimization_result.hess_inv.todense())
    se_rest.loc[macro,:] = np.sqrt(np.diag(hess[-1]/ret.shape[0]))
    print("SE:",se_rest.loc[macro,:])
    if save: res_rest.to_csv(files[0])
    if save: se_rest.to_csv(files[1])
    print("- - - Starting Unrestricted {} - - -".format(macro))
    # Unrestricted case
    unrest_reg = basinhopping(llf.simpleGM_asym,rest_reg.x,\
                            minimizer_kwargs=bs.min_arg(args,unrest),\
                            niter=niter,
                            take_step=step_u,
                            callback=bs.print_fun)
    print(unrest_reg)
    res_unrest.loc[macro,'mu':'w2'] = unrest_reg.x
    res_unrest.loc[macro,'Log-Likelihood'] = unrest_reg.fun
    hess.append(unrest_reg.lowest_optimization_result.hess_inv.todense())
    se_unrest.loc[macro,:] = np.sqrt(np.diag(hess[-1]/ret.shape[0]))
    print("SE:",se_rest.loc[macro,:])
    if save: se_unrest.to_csv(files[3])
    if save: res_unrest.to_csv(files[2])
    print("- - - Complete - - -")

if save: res_rest.to_csv(files[0])
if save: res_unrest.to_csv(files[2])
if save: se_rest.to_csv(files[1])
if save: se_unrest.to_csv(files[3])
