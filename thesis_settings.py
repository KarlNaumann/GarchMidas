import os
#Make subdirectory for our results
if not os.path.exists('results'):
    os.mkdir('results')
if not os.path.exists('figures'):
    os.mkdir('figures')
if not os.path.exists('forecasting'):
    os.mkdir('forecasting')
if not os.path.exists('forecasting_rep'):
    os.mkdir('forecasting_rep')
if not os.path.exists('markovOutput'):
    os.mkdir('markovOutput')
# General variable settings:
def midasLags():
    return 12    
# Ordered variable names
def var_names():
    """returns list of variable names"""
    names=[ r'$\Delta$ Real Consumption',
            r'$\Delta$ Consumer Sentiment Index',
            r'$\Delta$ GDP',
            r'$\Delta$ Housing Starts',
            r'Inflation',
            r'$\Delta$ Industrial Production Index',
            r'National Activity Index',
            r'New Orders Index',
            r'$\Delta$ Corporate Profits',
            r'Realized Variance',
            r'Term Spread',
            r'$\Delta$ Unemployment']
    return names
def gm_results():
    files = ['results/res_rest.csv',
                'results/se_rest.csv',
                'results/res_unrest.csv',
                'results/se_unrest.csv']
    return files
def ms_results():
    files = ['markovResults/EMA_restricted_direct.csv',
                'markovResults/EMA_unrestricted_direct.csv']
    return files   
def forecastingReplication_results():
    files = ['forecasting_rep/res_rest.csv',
                'forecasting_rep/se_rest.csv',
                'forecasting_rep/res_unrest.csv',
                'forecasting_rep/se_unrest.csv']
    return files
def gm_fig():
    """returns GM analysis figure designation"""
    figures = [ 'figures/fig:annVol_GM_r.png',
                'figures/fig:annVol_GM_u.png',
                'figures/fig:annVol_GM_des.png',
                'figures/fig:innovHist_GM_r.png',
                'figures/fig:innovHist_GM_u.png',
                'figures/fig:allWeights_GM.png']
    return figures
def gm_res():
    """returns GM results file designation"""
    result_files = ['results/res_rest.csv',
                    'results/res_unrest.csv',
                    'results/se_rest.csv',
                    'results/se_unrest.csv']
    return result_files
def gm_analysis_csv():
    """Return GM analysis output file designation"""
    output_files = ['results/GM_Analysis_pval_rest.csv',
                    'results/GM_Analysis_tstat_rest.csv',
                    'results/GM_Analysis_pval_unrest.csv',
                    'results/GM_Analysis_tstat_unrest.csv',
                    'results/GM_Analysis_lrt.csv',
                    'results/GM_Analysis_results.csv',
                    'results/GM_Analysis_desired_coef.csv',
                    'results/GM_Analysis_desired_se.csv']
    return output_files
def ms_analysis_csv():
    """Return GM analysis output file designation"""
    output_files = ['markovOutput/MS_Analysis_pval_rest.csv',
                    'markovOutput/MS_Analysis_tstat_rest.csv',
                    'markovOutput/MS_Analysis_pval_unrest.csv',
                    'markovOutput/MS_Analysis_tstat_unrest.csv',
                    'markovOutput/MS_Analysis_lrt.csv',
                    'markovOutput/MS_Analysis_results.csv',
                    'markovOutput/MS_Analysis_desired_coef.csv',
                    'markovOutput/MS_Analysis_desired_se.csv']
    return output_files