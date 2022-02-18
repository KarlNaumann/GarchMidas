"""
Cleaning of the data
-------------------------------------------------------
Takes data from the RawData/ folder and transforms it
"""

__author__ = "Karl Naumann-Woleske"
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "Karl Naumann-Woleske"


import numpy as np
import pandas as pd
import latexExport as latex
import matplotlib.gridspec as gridspec

from matplotlib import pyplot as plt
from matplotlib import rc
from statsmodels.tsa.stattools import adfuller
from scipy.stats import skew, kurtosis


# Update the plot fonts and allow usage of LaTeX
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)


def autocorr(x):
    """First-order autocorrelation of array x

    Parameters
    ----------
    x : np.ndarray

    Returns
    -------
    autocorr : float
    """
    mu = np.mean(x)
    top = np.sum(np.multiply(x[:-1] - mu, x[1:] - mu))
    bottom = np.sum((x[:-1] - mu)**2)
    return top / bottom


def dataInput(retIx='cut'):
    """Loads the returns and macro data of Conrad & Loch (2015) - Sets the date
       range to 1973-01-01 -> 2010-12-31 for equality in the replication
       exercise

    Parameters
    ----------
    retIx : str

    Returns
    -------
    ret : pd.DataFrame
    mvs : pd.DataFrame
    """
    # RETURNS DATA - Centered around 0
    df1 = pd.read_excel('data/cl_returndata.xls', index_col=0)
    ix = pd.DatetimeIndex(df1.index.tolist())
    x = ix.get_loc('1973-01-01', method='bfill')

    if retIx == 'cut':
        y = ix.get_loc('2011-01-01', method='pad')
        ret = pd.DataFrame(df1.values[x:y, :], index=ix[x:y], columns=['Returns'])
    else:
        ret = pd.DataFrame(df1.values[x:, :], index=ix[x:], columns=['Returns'])

    print("NaN check returns (should be False): ", df1.isnull().values.any())

    # MACRO VARIABLES
    df2 = pd.read_excel('data/cl_macrodata.xls', index_col=0, header=0)
    ix = pd.DatetimeIndex(df2.index.tolist())
    cols = ['RV', 'GDPD', 'IPI', 'UNEMP', 'HOUSE', 'PROFIT', 'INF', 'NAI',
            'NOI', 'CSI', 'CONS', 'SPREAD']

    raw = pd.DataFrame(df2.values, index=ix, columns=cols, dtype=np.float64)
    print("NaN check macro (should be False): ", raw.isnull().values.any())
    mvs = raw.reindex(columns=sorted(raw.columns))
    return ret, mvs


def descriptiveStats(ret: pd.DataFrame, mvs: pd.DataFrame, verbose=True):
    """ Descriptive stats for the returns and the macroeconomic variables
        returns two latex tables of descriptive stats

    Parameters
    ----------
    ret : pd.DataFrame
    mvs : pd.DataFrame
    verbose : bool
    """
    if verbose:
        print("Data Analysis")

    cols = ['Obs.', 'Max', 'Mean', 'Median', 'Min', 'Std. Deviation', 'Skew',
            'Kurtosis']
    dataAnalysis1 = pd.DataFrame(np.zeros((1, 8)), index=['Returns'],
                                 columns=cols)
    dataAnalysis2 = pd.DataFrame(np.zeros((len(mvs.columns.tolist()), 8)),
                                 index=mvs.columns.tolist(), columns=cols)

    for r in ret.columns.tolist():
        dataAnalysis1.loc[r, 'Obs.'] = ret.loc[:, r].values.shape[0]
        dataAnalysis1.loc[r, 'Max'] = np.max(ret.loc[:, r])
        dataAnalysis1.loc[r, 'Mean'] = np.mean(ret.loc[:, r])
        dataAnalysis1.loc[r, 'Median'] = np.median(ret.loc[:, r])
        dataAnalysis1.loc[r, 'Min'] = np.min(ret.loc[:, r])
        dataAnalysis1.loc[r, 'Std. Deviation'] = np.std(ret.loc[:, r])
        dataAnalysis1.loc[r, 'Skew'] = skew(ret.loc[:, r])
        dataAnalysis1.loc[r, 'Kurtosis'] = kurtosis(ret.loc[:, r])
        dataAnalysis1.loc[r, 'AC'] = autocorr(ret.loc[:, r])

    if verbose:
        print(dataAnalysis1)

    latex.latex_table(dataAnalysis1,
                      'Descriptive Statistics of the Returns',
                      'results/tab:descriptives_ret.tex',
                      indexTitle='',
                      save=True, precision=4,
                      index=True, verbose=False)

    for mv in mvs.columns.tolist():
        dataAnalysis2.loc[mv, 'Obs.'] = mvs.loc[:, mv].values.shape[0]
        dataAnalysis2.loc[mv, 'Max'] = np.max(mvs.loc[:, mv])
        dataAnalysis2.loc[mv, 'Mean'] = np.mean(mvs.loc[:, mv])
        dataAnalysis2.loc[mv, 'Median'] = np.median(mvs.loc[:, mv])
        dataAnalysis2.loc[mv, 'Min'] = np.min(mvs.loc[:, mv])
        dataAnalysis2.loc[mv, 'Std. Deviation'] = np.std(mvs.loc[:, mv])
        dataAnalysis2.loc[mv, 'Skew'] = skew(mvs.loc[:, mv])
        dataAnalysis2.loc[mv, 'Kurtosis'] = kurtosis(mvs.loc[:, mv])
        dataAnalysis2.loc[mv, 'AC'] = autocorr(mvs.loc[:, mv])

    if verbose:
        print(dataAnalysis2)

    latex.latex_table(dataAnalysis2,
                      'Descriptive Statistics of the Macroeconomic Variables',
                      'results/tab:descriptives_mc.tex',
                      indexTitle='',
                      save=True, precision=4,
                      index=True, verbose=False)
    if verbose:
        print("---complete")


def recessions():
    """Imports the NBER Recession data, sets the period to 1970-01-01
        until 2010-12-31. This is used for graph-shading purposes

    Returns
    -------
    recession : pd.DataFrame
    """
    df = pd.read_csv('data/USREC.csv', index_col=0)
    ix = pd.DatetimeIndex(df.index.tolist())
    x = ix.get_loc('1973-01-01', method='bfill')
    y = ix.get_loc('2011-01-01', method='bfill')
    recession = pd.DataFrame(df.values[x:y, :], index=ix[x:y])
    recession.columns = ['recession']
    print("NaN check Recession (should be false):", df.isnull().values.any())
    return recession


def recessionSE():
    """Returns list for the start and end of recessions

    Returns
    -------
    start : list
    end : list
    """
    r = recessions()
    start = []
    end = []
    # Check to see if we start in recession
    if r.iloc[0, 0] == 1:
        start.extend([r.index[0]])

    # add recession start and end dates
    for i in range(1, r.shape[0]):
        a = r.iloc[i - 1, 0]
        b = r.iloc[i, 0]
        if a == 0 and b == 1:
            start.extend([r.index[i]])
        elif a == 1 and b == 0:
            end.extend([r.index[i - 1]])

    # if there is a recession at the end, add the last date
    if len(start) > len(end):
        end.extend([r.index[-1]])
    return start, end


def realizedVariance():
    """Import the realized variance data and return its series, also
       saves latex document with descriptive stats

    Returns
    -------
    rv : pd.DataFrame
    """
    df = pd.read_excel('data/cl_rvintradata.xls', index_col=0, usecols=[0])
    ix = pd.DatetimeIndex(df.index.tolist())
    x = ix.get_loc('1973-01-01', method='bfill')
    y = ix.get_loc('2010-12-31', method='bfill')
    # Rescale to match the % scale of the returns series
    v = df.values[x:y, :] * 100 * 100
    rv = pd.DataFrame(v, index=ix[x:y])
    rv = rv.dropna(axis=0)
    rv.columns = ['IntraRV']
    print("NaN check returns intra-RV (should be false):",
          rv.isnull().values.any())
    return rv


if __name__ == "__main__":
    ret, mvs = dataInput()
    print(descriptiveStats(ret, mvs))

    # ADF for stationarity
    print('Augmented Dickey Fuller Test')
    for col in mvs.columns:
        res = adfuller(mvs.loc[:, col].values)[1]
        print("ADF for {} is {}".format(col, res))

    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)
    names = [
        r'$\Delta$ Real Consumption',
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
        r'$\Delta$ Unemployment'
    ]

    r_s, r_e = recessionSE()
    l = len(names)

    if np.remainder(l, 4) == 1:
        unev, r = 1, 1
    elif np.remainder(l, 4) == 2:
        unev, r = 2, 1
    elif np.remainder(l, 4) == 3:
        unev, r = 3, 1
    else:
        unev, r = 0, 0

    rows = np.floor_divide(l, 3) + r
    gs = gridspec.GridSpec(rows, 8)
    ax_lst = []
    for i in range(np.floor_divide(l, 4)):
        ax1 = plt.subplot(gs[i, 0:2])
        ax2 = plt.subplot(gs[i, 2:4])
        ax3 = plt.subplot(gs[i, 4:6])
        ax4 = plt.subplot(gs[i, 6:])
        ax_lst.extend([ax1, ax2, ax3, ax4])
    if unev == 1:
        ax = plt.subplot(gs[rows - 1, 3:5])
        ax_lst.extend([ax])
    if unev == 2:
        ax1 = plt.subplot(gs[rows - 1, 2:4])
        ax2 = plt.subplot(gs[rows - 1, 4:6])
        ax_lst.extend([ax1, ax2])
    if unev == 3:
        ax1 = plt.subplot(gs[rows - 1, 1:3])
        ax2 = plt.subplot(gs[rows - 1, 3:5])
        ax3 = plt.subplot(gs[rows - 1, 5:7])
        ax_lst.extend([ax1, ax2, ax3])

    fig = plt.gcf()
    fig.set_size_inches(14, 10)
    gs.tight_layout(fig)
    f = {'fontsize': 8, 'fontweight': 'medium'}
    for i in range(len(names)):
        ax_lst[i].plot(mvs.iloc[:, i], label=mvs.columns[i],
                       linewidth=0.5, linestyle='solid', color='blue')
        ax_lst[i].axhline(linewidth=0.5, color='black')
        for j in zip(r_s, r_e):
            ax_lst[i].axvspan(xmin=j[0], xmax=j[1], color='gainsboro')
        ax_lst[i].set_title(names[i],
                            fontdict={'fontsize': 12, 'fontweight': 'medium'})
        ax_lst[i].autoscale(enable=True, axis='x', tight=True)
        ax_lst[i].tick_params(axis='both', which='major', labelsize=8)
        ax_lst[i].set_xlabel('Year', fontdict=f)
        ax_lst[i].minorticks_on()
    plt.tight_layout()
    gs.tight_layout(fig)
    plt.savefig('figures/fig:TS_rawData', bbox_inches='tight')
    plt.show()
